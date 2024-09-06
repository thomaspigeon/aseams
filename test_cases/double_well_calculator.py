import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from ase.stress import full_3x3_to_voigt_6_stress


class DoubleWell(Calculator):
    """Simple double well potential calculator"""

    implemented_properties = ["energy", "energies", "forces", "free_energy"]
    implemented_properties += ["stress", "stresses"]  # bulk properties
    default_parameters = {
        "a": 1.0,
        "d1": 1.0,
        "d2": 2.0,
        "rc": None,
    }
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        d1,d2: float
          The potential minima
        a: float
          The potential depth, default 1.0

        rc: float, None
          Cut-off for the NeighborList is set to 3 * sigma if None.
          The energy is upshifted to be continuous at rc.

        """

        Calculator.__init__(self, **kwargs)

        if self.parameters.rc is None:
            self.parameters.rc = 3 * max(self.parameters.d1, self.parameters.d2)

        self.nl = None

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)

        d1 = self.parameters.d1
        d2 = self.parameters.d2
        a = self.parameters.a * 16 * d1**2 * d2**2 / (d1 - d2) ** 4  # To have height of the barrier = a
        rc = self.parameters.rc

        if self.nl is None or "numbers" in system_changes:
            self.nl = NeighborList([rc / 2] * natoms, self_interaction=False, bothways=True)

        self.nl.update(self.atoms)

        positions = self.atoms.positions
        cell = self.atoms.cell

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        # potential value at rc
        e0 = a * ((rc - d1) * (rc - d2) / rc**2) ** 2
        for ii in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(ii)
            cells = np.dot(offsets, cell)
            # pointing *towards* neighbours
            distance_vectors = positions[neighbors] + cells - positions[ii]

            r = np.sqrt((distance_vectors**2).sum(1))

            pairwise_energies = a * (((r - d1) * (r - d2) / r**2) ** 2) * (r <= rc)
            pairwise_forces = 2 * a * ((d1 + d2) * r - 2 * d1 * d2) * ((r - d1) * (r - d2)) / r**5 * (r <= rc)  # du_ij

            pairwise_energies -= e0
            pairwise_forces = (pairwise_forces / r)[:, np.newaxis] * distance_vectors

            energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies
            forces[ii] += pairwise_forces.sum(axis=0)

            stresses[ii] += 0.5 * np.dot(pairwise_forces.T, distance_vectors)  # equivalent to outer product

        # no lattice, no stress
        if self.atoms.cell.rank == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results["stress"] = stresses.sum(axis=0) / self.atoms.get_volume()
            self.results["stresses"] = stresses / self.atoms.get_volume()

        energy = energies.sum()
        self.results["energy"] = energy
        self.results["energies"] = energies

        self.results["free_energy"] = energy

        self.results["forces"] = forces

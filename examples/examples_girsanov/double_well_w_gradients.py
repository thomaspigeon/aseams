import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList

class DoubleWell(Calculator):
    """Double Well with gradient of the descriptors. Requires torch.
    """
    implemented_properties = ["energy", "energies", "forces", "free_energy"]
    implemented_properties += ["stress", "stresses", "grad_descriptors", "descriptors"]
    default_parameters = {
        "a": 1.0,
        "d1": 1.0,
        "d2": 2.0,
        "rc": None,
    }
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

        if self.parameters.rc is None:
            self.parameters.rc = 3 * max(self.parameters.d1, self.parameters.d2)
        
        self.nl = None

    def _compute_pair_descriptor(self, r, d1, d2, rc):
        r_filtered = r[r <= rc]
        
        if r_filtered.shape[0] == 0:
            return torch.zeros(1, dtype=r.dtype, device=r.device)

        scaling = 16 * d1**2 * d2**2 / (d1 - d2) ** 4
        e0 = (((rc - d1) * (rc - d2) / rc**2) ** 2)
        
        val = scaling * (((r_filtered - d1) * (r_filtered - d2) / r_filtered**2) ** 2 - e0)
        return torch.sum(val)

    def get_descriptors(self, positions_tensor):
        natoms = positions_tensor.shape[0]
        d1 = self.parameters.d1
        d2 = self.parameters.d2
        rc = self.parameters.rc
        
        cell = torch.tensor(self.atoms.cell.array, dtype=torch.float64, device=positions_tensor.device)

        if self.nl is None:
            self.nl = NeighborList([rc / 2] * natoms, self_interaction=False, bothways=True)
            self.nl.update(self.atoms)

        total_descriptor = torch.zeros(1, dtype=torch.float64, device=positions_tensor.device)

        for ii in range(natoms):
            neighbors, offsets_np = self.nl.get_neighbors(ii)
            if len(neighbors) == 0:
                continue
            
            offsets = torch.tensor(offsets_np, dtype=torch.float64, device=positions_tensor.device)
            
            ri = positions_tensor[ii]
            rj = positions_tensor[neighbors]
            
            cells_offset = torch.matmul(offsets, cell)
            distance_vectors = rj + cells_offset - ri
            
            r = torch.norm(distance_vectors, dim=1)
            
            local_val = self._compute_pair_descriptor(r, d1, d2, rc)
            total_descriptor[0] += 0.5 * local_val

        return total_descriptor

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
        rc = self.parameters.rc
        
        if self.nl is None or "numbers" in system_changes:
            self.nl = NeighborList([rc / 2] * natoms, self_interaction=False, bothways=True)

        self.nl.update(self.atoms)
        
        positions = torch.tensor(self.atoms.positions, dtype=torch.float64, requires_grad=True)
        theta_val = self.parameters.a
        
        descriptors = self.get_descriptors(positions)
        
        grad_descriptors_list = []
        for i in range(descriptors.shape[0]):
            grad_d = torch.autograd.grad(
                descriptors[i], 
                positions, 
                retain_graph=(i < descriptors.shape[0] - 1)
            )[0]
            grad_descriptors_list.append(grad_d.detach().numpy())
            
        grad_descriptors_np = np.array(grad_descriptors_list)
        
        forces = -theta_val * grad_descriptors_np[0]
        total_energy = theta_val * descriptors.item()
        
        self.results["energy"] = total_energy
        self.results["free_energy"] = total_energy
        self.results["forces"] = forces
        self.results["descriptors"] = descriptors.detach().numpy()
        self.results["grad_descriptors"] = grad_descriptors_np
        self.results["energies"] = np.zeros(natoms) 
        self.results["stress"] = np.zeros(6)
        self.results["stresses"] = np.zeros((natoms, 6))
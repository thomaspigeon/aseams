import numpy as np
import os, shutil, time
from ase.io import Trajectory, read, write
from ase.parallel import world, paropen, barrier
from ams import NumpyEncoder
import json
from ase.constraints import FixCom


class InitialConditionsSampler:
    """Class to sample initial conditions to run AMS later"""

    def __init__(self, dyn, xi, cv_interval=1, fixcm=True, rng=None):
        """An initial conditions sampler with a single replica

        Parameters:

        dyn: MolecularDynamics object
            Should be a stochastic dynamics

        xi: CollectiveVariable object
            Object that allows to measure whether the dynamics is in reactant (R) state, in product (P) state and the
            progress on the transitions

        cv_interval: int
            The CV is evaluated every cv_interval time steps
        """
        if type(xi).__name__ != "CollectiveVariables":
            raise ValueError("""xi must be a CollectiveVariables object""")
        #if type(dyn).__name__ != "Langevin":
            #raise ValueError("""dyn must be a Langevin object""")
        if isinstance(cv_interval, int) and cv_interval >= 0:
            self.cv_interval = cv_interval
        else:
            raise ValueError("""cv_interval must be an int >= 0""")
        xi.test_the_collective_variables(dyn.atoms)
        self.xi = xi
        self.dyn = dyn
        self.dyn.nsteps = 0
        self.calc = dyn.atoms.calc
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng
        self.run_dir = None
        self.ini_cond_dir = None
        self.t_sigma_r = None
        self.t_r_sigma = None
        self.first_in_r = False
        self.going_to_sigma = True
        self.going_back_to_r = False
        self.last_r_visited = None
        self.t_r_sigma_out = None
        self.t_sigma_out = None
        self.n_ini_conds_already = None
        self.trajfile = None
        self.fixcm = fixcm

    def set_run_dir(self, run_dir="./ini_conds_md_logs", append_traj=False):
        """Where the md logs will be written, if the directory does not exist, it will create it."""
        self.run_dir = run_dir
        if not os.path.exists(self.run_dir) and world.rank == 0:
            os.mkdir(self.run_dir)
        n_traj_already = len([fi for fi in os.listdir(self.run_dir) if fi.endswith(".traj")])
        if not append_traj:
            self.trajfile = self.run_dir + "/md_traj_{}.traj".format(n_traj_already)
            traj = self.dyn.closelater(Trajectory(filename=self.trajfile,
                                                  mode="a",
                                                  atoms=self.dyn.atoms,
                                                  properties=['energy', 'stress', 'forces']))
            self.dyn.attach(traj.write, interval=self.cv_interval)

    def set_ini_cond_dir(self, ini_cond_dir="./ini_conds"):
        """Where the initial conditions for AMS will be written, if the directory does not exist, it will create it."""
        self.ini_cond_dir = ini_cond_dir
        if not os.path.exists(self.ini_cond_dir) and world.rank == 0:
            os.mkdir(self.ini_cond_dir)

    def sample(self, n_conditions=100, n_steps=None):
        """Sampling initial conditions

        Parameters:

        n_conditions: int
            Number of initial conditions to sample, default is 100. If this argument is given, n_steps should be kept to
            None
        n_steps: int
            Number of integration time steps that should be performed. If this argument is given, n_conditions should be
            set to None
        """
        if self.run_dir is None:
            raise ValueError("""The running directory is not set ! Call initialconditionssampler.set_run_dir""")
        if self.ini_cond_dir is None:
            raise ValueError("""The directory to store initial conditions is not defined ! Call initialconditionssampler.set_ini_cond_dir""")
        if n_steps is None:
            n_steps = -1
            if not isinstance(n_conditions, int) or n_conditions <= 0:
                raise ValueError("""n_conditions should be an int > 0 if n_steps is left to None""")
        elif n_conditions is None:
            n_conditions = -1
            if not isinstance(n_steps, int) or n_steps <= 0:
                raise ValueError("""n_steps should be an int > 0 if n_conditions is set to None""")
        else:
            raise ValueError(
                """n_conditions should be an int > 0 if n_steps is left to None or n_steps should be an
                                int > 0 if n_conditions is set to None"""
            )
        n_cdt, n_stp = 0, 0
        if self.fixcm:
            self.dyn.atoms.set_constraint(FixCom())
        self.n_ini_conds_already = len([ini for ini in os.listdir(self.ini_cond_dir) if ini.endswith(".extyxz")])
        if isinstance(self.xi.cv_r, list):
            self.t_r_sigma = [[] for i in range(len(self.xi.cv_r))]
            self.t_sigma_r = [[] for i in range(len(self.xi.cv_r))]
            self.t_r_sigma_out = [[] for i in range(len(self.xi.cv_r))]
            self.t_sigma_out = [[] for i in range(len(self.xi.cv_r))]
        else:
            self.t_r_sigma = [[]]
            self.t_sigma_r = [[]]
            self.t_r_sigma_out = [[]]
            self.t_sigma_out = [[]]
        self.first_in_r = False
        while not self.first_in_r:
            self.dyn.run(self.cv_interval)
            n_stp += self.cv_interval
            if self.xi.is_out_of_r_zone(self.dyn.atoms):
                list_atoms = read(self.trajfile, index=":")
                at = list_atoms[self.rng.choice(len(list_atoms))]
                self._set_initialcond_dyn(at)
                self.dyn.call_observers()
                self.first_in_r = False
            if self.xi.in_r(self.dyn.atoms):
                self.first_in_r = True
        while n_cdt < n_conditions or n_stp < n_steps:
            self.last_r_visited = np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
            self.t_r_sigma[self.last_r_visited].append(0)
            self.t_sigma_r[self.last_r_visited].append(0)
            self.going_back_to_r = False
            self.going_to_sigma = True
            while not self.xi.above_sigma(self.dyn.atoms):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                self.t_r_sigma[self.last_r_visited][-1] += self.cv_interval
            fname = self.ini_cond_dir + "/" + str(self.n_ini_conds_already + n_cdt + 1) + ".extxyz"
            write(fname, self.dyn.atoms, format='extxyz')
            self.going_back_to_r = True
            self.going_to_sigma = False
            self._write_checkpoint()
            while not self.xi.in_r(self.dyn.atoms):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                if self.first_in_r:
                    self.t_sigma_r[self.last_r_visited][-1] += self.cv_interval
                if self.xi.is_out_of_r_zone(self.dyn.atoms):
                    self.first_in_r = False
                    list_atoms = read(self.trajfile, index=":")
                    at = list_atoms[self.rng.choice(range(len(list_atoms)))]
                    self._set_initialcond_dyn(at)
                    self.dyn.call_observers()
                    if self.first_in_r:
                        t_sigma_out = self.t_sigma_r[self.last_r_visited].pop(-1)
                        t_r_sigma_out = self.t_r_sigma[self.last_r_visited].pop(-1)
                        self.t_sigma_out[self.last_r_visited].append(t_sigma_out)
                        self.t_r_sigma_out[self.last_r_visited].append(t_r_sigma_out)
                if self.xi.in_r(self.dyn.atoms):
                    self.first_in_r = True
            n_cdt += 1
            self._write_checkpoint()
        self.n_ini_conds_already = self.n_ini_conds_already + n_cdt
        self._write_checkpoint()

    def _write_checkpoint(self):
        """write checkpoint data for step by step run"""
        checkpoint_data = {"run_dir": self.run_dir,
                           "ini_cond_dir": self.ini_cond_dir,
                           "nsteps": self.dyn.nsteps,
                           "cv_interval": self.cv_interval,
                           "first_in_r": self.first_in_r,
                           "last_r_visited": self.last_r_visited,
                           "going_to_sigma": self.going_to_sigma,
                           "going_back_to_r": self.going_back_to_r,
                           "trajfile": self.trajfile,
                           "n_ini_conds_already" : self.n_ini_conds_already,
                           "t_sigma_r": self.t_sigma_r,
                           "t_r_sigma": self.t_r_sigma,
                           "t_sigma_out": self.t_sigma_out,
                           "t_r_sigma_out": self.t_r_sigma_out}
        json_file = paropen(self.run_dir + "/ini_checkpoint.txt", "w")
        json.dump(checkpoint_data, json_file, indent=4, cls=NumpyEncoder)
        json_file.close()

    def _read_checkpoint(self):
        """Read the necessary information to restart sampler from the checkpoint file"""
        json_file = paropen(self.run_dir + "/ini_checkpoint.txt", "r")
        checkpoint_data = json.load(json_file)
        json_file.close()
        self.run_dir = checkpoint_data["run_dir"]
        self.ini_cond_dir = checkpoint_data["ini_cond_dir"]
        self.dyn.nsteps = checkpoint_data["nsteps"]
        self.cv_interval = checkpoint_data["cv_interval"]
        self.n_ini_conds_already = checkpoint_data["n_ini_conds_already"]
        self.trajfile = checkpoint_data["trajfile"]
        self.t_sigma_r = checkpoint_data["t_sigma_r"]
        self.t_r_sigma = checkpoint_data["t_r_sigma"]
        self.t_sigma_out = checkpoint_data["t_sigma_out"]
        self.t_r_sigma_out = checkpoint_data["t_r_sigma_out"]
        self.first_in_r = checkpoint_data["first_in_r"]
        self.going_to_sigma = checkpoint_data["going_to_sigma"]
        self.going_back_to_r = checkpoint_data["going_back_to_r"]
        self.last_r_visited = checkpoint_data["last_r_visited"]

    def _write_current_atoms(self):
        write("current_atoms.xyz", self.dyn.atoms, format="extxyz")

    def _set_initialcond_dyn(self, atoms):
        """
        Set atomic position and momenta of a dynamic
        """
        self.dyn.atoms.set_scaled_positions(atoms.get_scaled_positions())
        self.dyn.atoms.set_momenta(atoms.get_momenta())
        self.dyn.atoms.calc.results['forces'] = atoms.get_forces(apply_constraint=False)
        self.dyn.atoms.calc.results['stress'] = atoms.get_stress(apply_constraint=False)
        self.dyn.atoms.calc.results['energy'] = atoms.get_potential_energy()

    def sample_step_by_step(self, forces, energy, stress):
        """Run sampling of ini-conds calling this function steps by steps"""
        if os.path.exists(self.run_dir + "/ini_checkpoint.txt"):
            self._read_checkpoint()

        else:
            self.dyn.nsteps = 0
            n_traj_already = len([fi for fi in os.listdir(self.run_dir) if fi.endswith(".traj")])
            self.trajfile = self.run_dir + "/md_traj_{}.traj".format(n_traj_already)
        traj = self.dyn.closelater(Trajectory(filename=self.trajfile,
                                              mode="a",
                                              atoms=self.dyn.atoms,
                                              properties=['energy', 'stress', 'forces']))
        self.dyn.attach(traj.write, interval=self.cv_interval)
        barrier()
        if self.fixcm:
            self.dyn.atoms.set_constraint(FixCom())
        self.n_ini_conds_already = len(
            [ini for ini in os.listdir(self.ini_cond_dir) if ini.endswith("z")])
        if self.dyn.nsteps > 0:
            self.dyn.atoms.calc.results['forces'] = forces
            self.dyn.atoms.calc.results['stress'] = stress
            self.dyn.atoms.calc.results['energy'] = energy
            self.dyn._2nd_half_step(forces)
        else:
            self.dyn.call_observers()
            if isinstance(self.xi.cv_r, list):
                self.t_r_sigma = [[] for i in range(len(self.xi.cv_r))]
                self.t_sigma_r = [[] for i in range(len(self.xi.cv_r))]
                self.t_r_sigma_out = [[] for i in range(len(self.xi.cv_r))]
                self.t_sigma_out = [[] for i in range(len(self.xi.cv_r))]
            else:
                self.t_r_sigma = [[]]
                self.t_sigma_r = [[]]
                self.t_r_sigma_out = [[]]
                self.t_sigma_out = [[]]
        in_r = self.xi.in_r(self.dyn.atoms)
        if in_r:
            self.last_r_visited = np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
        out_of_r_zone = self.xi.is_out_of_r_zone(self.dyn.atoms)
        above_sigma = self.xi.above_sigma(self.dyn.atoms)
        if self.first_in_r:
            if self.going_to_sigma:
                self.t_r_sigma[self.last_r_visited][-1] += 1
            if self.going_back_to_r:
                self.t_sigma_r[self.last_r_visited][-1] += 1
            if above_sigma and self.going_to_sigma:
                self.going_to_sigma = False
                fname = self.ini_cond_dir + str(self.n_ini_conds_already + 1) + ".extxyz"
                write(fname, self.dyn.atoms, format='extxyz')
                self.going_back_to_r = True
            if in_r and self.going_back_to_r:
                self.going_back_to_r = False
                self.t_r_sigma[self.last_r_visited].append(0)
                self.t_sigma_r[self.last_r_visited].append(0)
                self.going_to_sigma = True
            if out_of_r_zone:
                list_atoms = read(self.trajfile, index=":")
                at = list_atoms[self.rng.choice(len(list_atoms))]
                self._set_initialcond_dyn(at)
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self.first_in_r = False
                self.going_to_sigma = True
                self.going_back_to_r = False
                t_sigma_out = self.t_sigma_r[self.last_r_visited].pop(-1)
                t_r_sigma_out = self.t_r_sigma[self.last_r_visited].pop(-1)
                self.t_sigma_out[self.last_r_visited].append(t_sigma_out)
                self.t_r_sigma_out[self.last_r_visited].append(t_r_sigma_out)
                self._write_checkpoint()
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False
            else:
                self.dyn._1st_half_step(forces)
                self.dyn.nsteps += 1
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self._write_checkpoint()
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False
        else:
            if in_r:
                self.first_in_r = True
                self.t_r_sigma[self.last_r_visited].append(0)
                self.t_sigma_r[self.last_r_visited].append(0)
            if out_of_r_zone:
                list_atoms = read(self.trajfile, index=":")
                at = list_atoms[self.rng.choice(len(list_atoms))]
                self._set_initialcond_dyn(at)
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self.first_in_r = False
                self.going_to_sigma = True
                self.going_back_to_r = False
                self._write_checkpoint()
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False
            else:
                self.dyn._1st_half_step(forces)
                self.dyn.nsteps += 1
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self._write_checkpoint()
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False


class InitialConditionsSamplerFromFile:
    """Class to sample initial conditions to run AMS later"""

    def __init__(self, xi, cv_interval=1):
        """An initial conditions sampler with a single replica

        Parameters:

        xi: CollectiveVariable object
            Object that allows to measure whether the dynamics is in reactant (R) state, in product (P) state and the
            progress on the transitions

        cv_interval: int
            The CV is evaluated every cv_interval time steps
        """
        if type(xi).__name__ != "CollectiveVariables":
            raise ValueError("""xi must be a CollectiveVariables object""")
        if isinstance(cv_interval, int) and cv_interval >= 0:
            self.cv_interval = cv_interval
        else:
            raise ValueError("""cv_interval must be an int >= 0""")

        self.xi = xi
        self.run_dir = None
        self.ini_cond_dir = None

    def set_ini_cond_dir(self, ini_cond_dir="./ini_conds", clean=False):
        """Where the initial conditions for AMS will be written, if the directory does not exist, it will create it."""
        self.ini_cond_dir = ini_cond_dir
        if world.rank == 0:
            if clean and os.path.exists(self.ini_cond_dir):
                shutil.rmtree(self.ini_cond_dir)
            if not os.path.exists(self.ini_cond_dir):
                os.mkdir(self.ini_cond_dir)

    def sample(self, file):
        """Sampling initial conditions

        Parameters:

        file: str
            path to the trajectory file
        """
        if self.ini_cond_dir is None:
            raise ValueError("""The directory to store initial conditions is not defined ! Call initialconditionssampler.set_ini_cond_dir""")

        if isinstance(self.xi.cv_r, list):
            t_r_sigma = [[] for i in range(len(self.xi.cv_r))]
            t_sigma_r = [[] for i in range(len(self.xi.cv_r))]
        else:
            t_r_sigma = [[]]
            t_sigma_r = [[]]

        n_cdt, n_stp = 0, 0
        n_ini_conds_already = len([ini for ini in os.listdir(self.ini_cond_dir) if ini.endswith(".extyxz")])
        traj = read(file, index=":")
        n_steps = len(traj)
        while not self.xi.in_r(traj[n_stp]):
            n_stp += self.cv_interval
            if n_stp >= n_steps:
                break
        while n_stp < n_steps:
            which_r = np.where(self.xi.in_which_r(traj[n_stp]) == np.max(self.xi.in_which_r(traj[n_stp])))[0][0]
            t_r_sigma[which_r].append(0)
            while not self.xi.above_sigma(traj[n_stp]):
                n_stp += self.cv_interval
                if n_stp >= n_steps:
                    t_r_sigma[which_r].pop()
                    break
                t_r_sigma[which_r][-1] += self.cv_interval
            if n_stp >= n_steps:  # If we have quit previous loop, we should also quit the main loop
                break
            fname = self.ini_cond_dir + "/" + str(n_ini_conds_already + n_cdt + 1) + ".extxyz"
            write(fname, traj[n_stp], format='extxyz')
            t_sigma_r[which_r].append(0)
            while not self.xi.in_r(traj[n_stp]):
                n_stp += self.cv_interval
                if n_stp >= n_steps:
                    t_sigma_r[which_r].pop()
                    break
                t_sigma_r[which_r][-1] += self.cv_interval
            n_cdt += 1
        return t_r_sigma, t_sigma_r


class FlemmingViotInitialConditionsSampler:
    """Class to sample initial conditions to run AMS later"""

    def __init__(self, dyn, xi, n_walkers=4, walker_index=0, fixcm=True, cv_interval=1, rng=None):
        """An initial conditions sampler with multiple replicas

        Parameters:

        dyn: MolecularDynamics object
            Should be a stochastic dynamics

        xi: CollectiveVariable object
            Object that allows to measure whether the dynamics is in reactant (R) state, in product (P) state and the
            progress on the transitions

        n_walkers: int
            number of walkers, positive

        waler_index: int
            index of this walker such that 0 <= waler_index < n_walkers

        cv_interval: int
            The CV is evaluated every cv_interval time steps
        """
        if type(xi).__name__ != "CollectiveVariables":
            raise ValueError("""xi must be a CollectiveVariables object""")
        #if type(dyn).__name__ != "Langevin":
        #    raise ValueError("""dyn must be a Langevin object""")
        if isinstance(n_walkers, int) and n_walkers >= 0:
            self.n_walkers = n_walkers
        else:
            raise ValueError("""n_walkers must be an int >= 0""")
        if isinstance(walker_index, int) and (0 <= walker_index < n_walkers):
            self.w_i = walker_index
        else:
            raise ValueError("""walker_index must be an int such that 0 <= walker_index < n_walkers""")
        if isinstance(cv_interval, int) and cv_interval >= 0:
            self.cv_interval = cv_interval
        else:
            raise ValueError("""cv_interval must be an int >= 0""")
        xi.test_the_collective_variables(dyn.atoms)
        self.xi = xi
        self.dyn = dyn
        self.calc = dyn.atoms.calc
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng
        self.fixcm = fixcm
        self.run_dir = None
        self.ini_cond_dir = None
        self.t_sigma_r = None
        self.t_r_sigma = None
        self.first_in_r = False
        self.going_to_sigma = True
        self.going_back_to_r = False
        self.last_r_visited = None
        self.t_r_sigma_out = None
        self.t_sigma_out = None
        self.n_ini_conds_already = None
        self.trajfile = None

    def _set_initialcond_dyn(self, atoms):
        """
        Set atomic position and momenta of a dynamic
        """
        self.dyn.atoms.set_scaled_positions(atoms.get_scaled_positions())
        self.dyn.atoms.set_momenta(atoms.get_momenta())
        self.dyn.atoms.calc.results['forces'] = atoms.get_forces(apply_constraint=False)
        self.dyn.atoms.calc.results['stress'] = atoms.get_stress(apply_constraint=False)
        self.dyn.atoms.calc.results['energy'] = atoms.get_potential_energy()

    def _write_checkpoint(self):
        """write checkpoint data for step by step run"""
        checkpoint_data = {"run_dir": self.run_dir,
                           "ini_cond_dir": self.ini_cond_dir,
                           "nsteps": self.dyn.nsteps,
                           "n_walkers": self.n_walkers,
                           "w_i": self.w_i,
                           "trajfile": self.trajfile,
                           "cv_interval": self.cv_interval,
                           "first_in_r": self.first_in_r,
                           "last_r_visited": self.last_r_visited,
                           "going_to_sigma": self.going_to_sigma,
                           "going_back_to_r": self.going_back_to_r,
                           "t_sigma_r": self.t_sigma_r,
                           "t_r_sigma": self.t_r_sigma,
                           "t_sigma_out": self.t_sigma_out,
                           "t_r_sigma_out": self.t_r_sigma_out}
        json_file = paropen(self.run_dir + str(self.w_i) + "/ini_fv_" + str(self.w_i) + "_checkpoint.txt", "w")
        json.dump(checkpoint_data, json_file, indent=4, cls=NumpyEncoder)
        json_file.close()

    def _read_checkpoint(self):
        """Read the necessary information to restart sampler from the checkpoint file"""
        json_file = paropen(self.run_dir + str(self.w_i) + "/ini_fv_" + str(self.w_i) + "_checkpoint.txt", "r")
        checkpoint_data = json.load(json_file)
        json_file.close()
        self.run_dir = checkpoint_data["run_dir"]
        self.ini_cond_dir = checkpoint_data["ini_cond_dir"]
        self.dyn.nsteps = checkpoint_data["nsteps"]
        self.n_walkers = checkpoint_data["n_walkers"]
        self.w_i = checkpoint_data["w_i"]
        self.trajfile = checkpoint_data["trajfile"]
        self.cv_interval = checkpoint_data["cv_interval"]
        self.t_sigma_r = checkpoint_data["t_sigma_r"]
        self.t_r_sigma = checkpoint_data["t_r_sigma"]
        self.t_sigma_out = checkpoint_data["t_sigma_out"]
        self.t_r_sigma_out = checkpoint_data["t_r_sigma_out"]
        self.first_in_r = checkpoint_data["first_in_r"]
        self.going_to_sigma = checkpoint_data["going_to_sigma"]
        self.going_back_to_r = checkpoint_data["going_back_to_r"]
        self.last_r_visited = checkpoint_data["last_r_visited"]

    def _write_current_atoms(self):
        write(self.run_dir + str(self.w_i) + "/current_atoms.xyz", self.dyn.atoms, format="extxyz")

    def set_run_dir(self, run_dir="./ini_conds_walker_", append_traj=True):
        """Where the md logs will be written, if the directory does not exist, it will create it."""
        self.run_dir = run_dir
        if not os.path.exists(self.run_dir + str(self.w_i)) and world.rank == 0:
            os.mkdir(self.run_dir + str(self.w_i))
        n_traj_already = len([fi for fi in os.listdir(self.run_dir + str(self.w_i)) if fi.endswith(".traj")])
        t = []
        for i in range(n_traj_already):
            trajfile = self.run_dir + str(self.w_i) + "/md_traj_{}.traj".format(i)
            list_atoms = read(trajfile, index=":", format='traj')
            if i == 0:
                t.append(len(list_atoms))
            else:
                t.append(len(list_atoms) + t[-1])
        if len(t) == 0:
            self.dyn.nsteps = 0
        else:
            self.dyn.nsteps = t[-1]
        if not append_traj:
            self.trajfile = self.run_dir + str(self.w_i) + "/md_traj_{}.traj".format(n_traj_already)
            traj = self.dyn.closelater(Trajectory(filename=self.trajfile,
                                                  mode="a",
                                                  atoms=self.dyn.atoms,
                                                  properties=['energy', 'stress', 'forces']))
            self.dyn.attach(traj.write, interval=self.cv_interval)


    def set_ini_cond_dir(self, ini_cond_dir="./ini_conds"):
        """Where the initial conditions for AMS will be written, if the directory does not exist, it will create it."""
        self.ini_cond_dir = ini_cond_dir
        if not os.path.exists(self.ini_cond_dir) and world.rank == 0 and self.w_i == 0:
            os.mkdir(self.ini_cond_dir)

    def _branch_fv_particle(self):
        traj_idx = None
        branch_rep_number = self.rng.choice(np.setdiff1d(np.arange(self.n_walkers), self.w_i))
        while traj_idx is None:
            n_traj_already = len([fi for fi in os.listdir(self.run_dir + str(branch_rep_number)) if fi.endswith(".traj")])
            t = []
            for i in range(n_traj_already):
                trajfile = self.run_dir + str(branch_rep_number) + "/md_traj_{}.traj".format(i)
                list_atoms = read(trajfile, index=":")
                if i == 0:
                    t.append(len(list_atoms))
                    if self.dyn.nsteps < t[-1]:
                        traj_idx = i
                else:
                    t.append(len(list_atoms) + t[-1])
                    if t[-2] <= self.dyn.nsteps < t[-1]:
                        traj_idx = i
            f = paropen(self.run_dir + str(self.w_i) + "/FV_history.txt", "a")
            f.write("Attempt branching replica " + str(self.w_i) + " from replica index: " + str(branch_rep_number) + " at time " + str(self.dyn.nsteps) + " \n")
            if traj_idx is not None:
                f.write("Succesful branching\n \n")
                f.close()
            else:
                f.write("Cannot branch as replica " + str(branch_rep_number) + "  did not reach time " + str(self.dyn.nsteps) + "\n")
                f.write("Going to sleep for 1 minute\n")
                f.close()
                time.sleep(60)
        trajfile = self.run_dir + str(branch_rep_number) + "/md_traj_{}.traj".format(traj_idx)
        list_atoms = read(trajfile, index=":", format='traj')
        atoms = list_atoms[self.dyn.nsteps - t[traj_idx - 1]]
        self._set_initialcond_dyn(atoms)
        return atoms

    def sample(self, n_conditions=100, n_steps=None):
        """Sampling initial conditions

        Parameters:

        n_conditions: int
            Number of initial conditions to sample, default is 100. If this argument is given, n_steps should be kept to
            None
        n_steps: int
            Number of integration time steps that should be performed. If this argument is given, n_conditions should be
            set to None
        """
        if self.run_dir is None:
            raise ValueError("""The running directory is not set ! Call initialconditionssampler.set_run_dir""")
        if self.ini_cond_dir is None:
            raise ValueError("""The directory to store initial conditions is not defined ! Call initialconditionssampler.set_ini_cond_dir""")
        while not os.path.exists(self.ini_cond_dir):
            time.sleep(1)
        if n_steps is None:
            n_steps = -1
            if not isinstance(n_conditions, int) or n_conditions <= 0:
                raise ValueError("""n_conditions should be an int > 0 if n_steps is left to None""")
        elif n_conditions is None:
            n_conditions = -1
            if not isinstance(n_steps, int) or n_steps <= 0:
                raise ValueError("""n_steps should be an int > 0 if n_conditions is set to None""")
        else:
            raise ValueError(
                """n_conditions should be an int > 0 if n_steps is left to None or n_steps should be an
                                int > 0 if n_conditions is set to None"""
            )
        n_cdt, n_stp = 0, 0
        if self.fixcm:
            self.dyn.atoms.set_constraint(FixCom())
        n_ini_conds_already = len([fi for fi in os.listdir(self.ini_cond_dir) if fi.startswith("walker_" + str(self.w_i) + "_")])
        if isinstance(self.xi.cv_r, list):
            if self.t_r_sigma is None:
                self.t_r_sigma = [[] for i in range(len(self.xi.cv_r))]
                self.t_sigma_r = [[] for i in range(len(self.xi.cv_r))]
                self.t_r_sigma_out = [[] for i in range(len(self.xi.cv_r))]
                self.t_sigma_out = [[] for i in range(len(self.xi.cv_r))]
        else:
            if self.t_r_sigma is None:
                self.t_r_sigma = [[]]
                self.t_sigma_r = [[]]
                self.t_r_sigma_out = [[]]
                self.t_sigma_out = [[]]
        while not self.first_in_r:
            self.dyn.run(self.cv_interval)
            n_stp += self.cv_interval
            if self.xi.is_out_of_r_zone(self.dyn.atoms):
                _ = self._branch_fv_particle()
                self.first_in_r = False
            if self.xi.in_r(self.dyn.atoms):
                self.first_in_r = True
        while n_cdt < n_conditions or n_stp < n_steps:
            self.last_r_visited = np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
            self.going_back_to_r = False
            self.going_to_sigma = True
            self.t_r_sigma[self.last_r_visited].append(0)
            self.t_sigma_r[self.last_r_visited].append(0)
            while not self.xi.above_sigma(self.dyn.atoms):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                self.t_r_sigma[self.last_r_visited][-1] += self.cv_interval
            fname = self.ini_cond_dir + "/walker_" + str(self.w_i) + '_ini_cond_' + str(n_ini_conds_already + n_cdt + 1) + ".extxyz"
            write(fname, self.dyn.atoms, format='extxyz')
            self.going_back_to_r = True
            self.going_to_sigma = False
            self._write_checkpoint()
            while not self.xi.in_r(self.dyn.atoms):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                if self.first_in_r:
                    self.t_sigma_r[self.last_r_visited][-1] += self.cv_interval
                if self.xi.is_out_of_r_zone(self.dyn.atoms):
                    _ = self._branch_fv_particle()
                    if self.first_in_r:
                        t_sigma_out = self.t_sigma_r[self.last_r_visited].pop(-1)
                        t_r_sigma_out = self.t_r_sigma[self.last_r_visited].pop(-1)
                        self.t_sigma_out[self.last_r_visited].append(t_sigma_out)
                        self.t_r_sigma_out[self.last_r_visited].append(t_r_sigma_out)
                    self.first_in_r = False
                if self.xi.in_r(self.dyn.atoms):
                    self.first_in_r = True
            n_cdt += 1
            self._write_checkpoint()

    def sample_step_by_step(self, forces, energy, stress):
        """Run sampling of ini-conds calling this function steps by steps"""
        if os.path.exists(self.run_dir + str(self.w_i) + "/ini_fv_" + str(self.w_i) + "_checkpoint.txt"):
            self._read_checkpoint()
        else:
            self.dyn.nsteps = 0
            n_traj_already = len([fi for fi in os.listdir(self.run_dir + str(self.w_i)) if fi.endswith(".traj")])
            self.trajfile = self.run_dir + str(self.w_i) + "/md_traj_{}.traj".format(n_traj_already)
        traj = self.dyn.closelater(Trajectory(filename=self.trajfile,
                                              mode="a",
                                              atoms=self.dyn.atoms,
                                              properties=['energy', 'stress', 'forces']))
        self.dyn.attach(traj.write, interval=self.cv_interval)
        barrier()
        if self.fixcm:
            self.dyn.atoms.set_constraint(FixCom())
        self.n_ini_conds_already = len(
                    [fi for fi in os.listdir(self.ini_cond_dir) if fi.startswith("walker_" + str(self.w_i) + "_")])
        if self.dyn.nsteps > 0:
            self.dyn.atoms.calc.results['forces'] = forces
            self.dyn.atoms.calc.results['stress'] = stress
            self.dyn.atoms.calc.results['energy'] = energy
            self.dyn._2nd_half_step(forces)
        else:
            self.dyn.call_observers()
            if isinstance(self.xi.cv_r, list):
                self.t_r_sigma = [[] for i in range(len(self.xi.cv_r))]
                self.t_sigma_r = [[] for i in range(len(self.xi.cv_r))]
                self.t_r_sigma_out = [[] for i in range(len(self.xi.cv_r))]
                self.t_sigma_out = [[] for i in range(len(self.xi.cv_r))]
            else:
                self.t_r_sigma = [[]]
                self.t_sigma_r = [[]]
                self.t_r_sigma_out = [[]]
                self.t_sigma_out = [[]]
        in_r = self.xi.in_r(self.dyn.atoms)
        if in_r:
            self.last_r_visited = np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
        out_of_r_zone = self.xi.is_out_of_r_zone(self.dyn.atoms)
        above_sigma = self.xi.above_sigma(self.dyn.atoms)
        if self.first_in_r:
            if self.going_to_sigma:
                self.t_r_sigma[self.last_r_visited][-1] += 1
            if self.going_back_to_r:
                self.t_sigma_r[self.last_r_visited][-1] += 1
            if above_sigma and self.going_to_sigma:
                self.going_to_sigma = False
                fname = self.ini_cond_dir + "/walker_" + str(self.w_i) + '_ini_cond_' + str(
                    self.n_ini_conds_already + 1) + ".extxyz"
                write(fname, self.dyn.atoms, format='extxyz')
                self.going_back_to_r = True
            if in_r and self.going_back_to_r:
                self.going_back_to_r = False
                self.t_r_sigma[self.last_r_visited].append(0)
                self.t_sigma_r[self.last_r_visited].append(0)
                self.going_to_sigma = True
            if out_of_r_zone:
                at = self._branch_fv_particle()
                self._set_initialcond_dyn(at)
                self.dyn.nsteps += 1
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self.first_in_r = False
                self.going_to_sigma = True
                self.going_back_to_r = False
                t_sigma_out = self.t_sigma_r[self.last_r_visited].pop(-1)
                t_r_sigma_out = self.t_r_sigma[self.last_r_visited].pop(-1)
                self.t_sigma_out[self.last_r_visited].append(t_sigma_out)
                self.t_r_sigma_out[self.last_r_visited].append(t_r_sigma_out)
                self._write_checkpoint()
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False
            else:
                self.dyn._1st_half_step(forces)
                self.dyn.nsteps += 1
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self._write_checkpoint()
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False
        else:
            if in_r:
                self.first_in_r = True
                self.t_r_sigma[self.last_r_visited].append(0)
                self.t_sigma_r[self.last_r_visited].append(0)
            if out_of_r_zone:
                at = self._branch_fv_particle()
                self._set_initialcond_dyn(at)
                self.dyn.nsteps += 1
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self.first_in_r = False
                self.going_to_sigma = True
                self.going_back_to_r = False
                self._write_checkpoint()
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False
            else:
                self.dyn._1st_half_step(forces)
                self.dyn.nsteps += 1
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self._write_checkpoint()
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False

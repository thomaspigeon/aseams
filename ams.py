import numpy as np
import os
import shutil
import json
from ase.io import Trajectory, read, write
from ase.io.formats import UnknownFileTypeError
from ase.parallel import parprint, paropen, world, barrier, broadcast


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class AMS:
    """Running AMS"""

    def __init__(self, n_rep, k_min, dyn, xi, cv_interval=1, rc_threshold=0.0, save_all=False, max_length_iter=np.infty, verbose=False):
        """
        Parameters:

        n_rep: int
            Number of replicas, must be strictly greater than 1

        k_min: int
            minimum number of replicas to kill at each iteration, must be superior or equal to 1 and strictly smaller
            than n_rep

        dyn: MolecularDynamics object
            Should be a stochastic dynamics

        xi: CollectiveVariable object
            Object that allows to measure whether the dynamics is in reactant (R) state, in product (P) state and the
            progress on the transitions

        cv_interval: int
            The CV is evaluated every cv_interval time steps

        rc_threshold: float
            The biggest difference between two rc values so that they are considered identical.

        max_length_iter : int
            Maximum length of the trajectory for one iteration

        save_all: boolean
            whether all the trajectories of the replicas should be saved. If false, only the current state of the
            replicas is written in the AMS.current_replicas_dir
        verbose: boolean
            Should AMS print information about progression
        """
        if isinstance(n_rep, int) and n_rep > 1:
            self.n_rep = n_rep
        else:
            raise ValueError("""n_rep must be an int > 1""")
        if isinstance(k_min, int) and 1 <= k_min < n_rep:
            self.k_min = k_min
        else:
            raise ValueError("""k_min must be an int such that 1 =< k_min < n_rep""")
        if type(xi).__name__ != "CollectiveVariables":
            raise ValueError("""xi must be a CollectiveVariables object""")
        # TODO Change the check here to make it consistent with otf objects.
        # if type(dyn).__name__ != "Langevin":
        #    raise ValueError("""dyn must be a Langevin object""")
        if isinstance(cv_interval, int) and cv_interval >= 1:
            self.cv_interval = cv_interval
        else:
            raise ValueError("""cv_interval must be an int >= 0""")
        if isinstance(save_all, bool):
            self.save_all = save_all
        else:
            raise ValueError("""save_all must be a boolean""")
        xi.test_the_collective_variables(dyn.atoms)
        self.success = False
        self.initialized = False
        self.finished = False
        self.xi = xi
        self.dyn = dyn
        self.dyn.nsteps = 1
        self.ini_cond_dir = None
        self.ams_dir = None
        self.z_maxs = None
        self.ams_it = 0
        self.remaining_killed = []
        self.alive = []
        self.current_rep = None
        self.current_p = None
        self.rep_weights = [[] for i in range(n_rep)]
        self.z_kill = []
        self.killed = []
        # self.calc = dyn.atoms.calc  # We should not event need to have it here I think.
        self.rc_threshold = rc_threshold
        self.max_length_iter = max_length_iter
        self.verbose = verbose

    def set_ini_cond_dir(self, ini_cond_dir="./ini_conds"):
        """Where the initial conditions for AMS will be written, raise error if the directory does not exist or is empty"""
        if not os.path.exists(ini_cond_dir):
            raise ValueError("""ini_cond_dir should exist, if not create it using initialconditionssampler.""")
        if len(os.listdir(ini_cond_dir)) == 0:
            raise ValueError("""ini_cond_dir should not be empty, use initialconditionssampler to write in it.""")
        self.ini_cond_dir = ini_cond_dir

    def set_ams_dir(self, ams_dir="./AMS", clean=False):
        """Where the alive trajectories will be stored, if the directory does not exist, it will create it.
        At the end of the run, if the estimated probability is not 0, these are reactive trajectories."""
        self.ams_dir = ams_dir
        if world.rank == 0:
            if clean and os.path.exists(self.ams_dir):
                shutil.rmtree(self.ams_dir)
            if not os.path.exists(self.ams_dir):
                os.mkdir(self.ams_dir)

    def _rc(self):
        z = self.xi.rc(self.dyn.atoms)
        if self.xi.in_r(self.dyn.atoms):
            z = -np.infty
        elif self.xi.in_p(self.dyn.atoms):
            z = np.infty
        return z

    def _set_initialcond_dyn(self, atoms):
        """
        Set atomic position and momenta of a dynamic
        """
        self.dyn.atoms.set_scaled_positions(atoms.get_scaled_positions())
        self.dyn.atoms.set_momenta(atoms.get_momenta())
        self.dyn.atoms.calc.results['forces'] = atoms.get_forces()
        self.dyn.atoms.calc.results['stress'] = atoms.get_stress()
        self.dyn.atoms.calc.results['energy'] = atoms.get_potential_energy()
        self.dyn.atoms.calc.results['free_energy'] = atoms.get_potential_energy()

    def _until_r_or_p(self, i, existing_steps=0):
        traj = self.dyn.closelater(Trajectory(filename=self.ams_dir + "/rep_" + str(i) + ".traj", mode="a", atoms=self.dyn.atoms))
        self.dyn.attach(traj.write, interval=self.cv_interval)
        self.dyn.nsteps = existing_steps  # Force writing the first step if start of the trajectory
        z = self._rc()
        f = paropen(self.ams_dir + "/rc_rep_" + str(i) + ".txt", "a")
        if existing_steps == 0:
            f.write(str(z) + "\n")
        while (z > -np.infty and z < np.infty) and self.dyn.nsteps <= self.max_length_iter:  # Cut trajectory of too long or reaching R or P
            self.dyn.run(self.cv_interval)
            z = self._rc()
            f.write(str(z) + "\n")
            if z >= self.z_maxs[i]:
                self.z_maxs[i] = z
        f.close()
        self.dyn.close()
        self.dyn.observers.pop(-1)

    def _pick_ini_cond(self, rep_index):
        if world.rank == 0:
            ini_cond = np.random.choice(
                [ini for ini in os.listdir(self.ini_cond_dir) if "_used" not in ini and ini.endswith(".extxyz")])
        else:
            ini_cond = None
        ini_cond = broadcast(ini_cond)
        atoms = read(self.ini_cond_dir + "/" + ini_cond, index=0)
        barrier()  # Wait for all mpi process to have read the file before moving it
        if world.rank == 0:
            filename, file_extension = os.path.splitext(ini_cond)
            os.rename(self.ini_cond_dir + "/" + ini_cond, self.ini_cond_dir + "/" + filename + "_used" + file_extension)
        # Initialize weight, either provided as a comment in the extxyz or set to default value
        self.rep_weights[rep_index].append(atoms.info.get("weight", 1.0) / self.n_rep)
        self._set_initialcond_dyn(atoms)
        self._write_checkpoint()  # Save weight into checkpoint

    def _initialize(self):
        """Run the N_rep replicas from the initial condition until it enters either R or P"""
        if self.ini_cond_dir is None:
            raise ValueError("""The directory of initial conditions is not defined ! Call ams.set_ini_cond_dir""")
        if self.ams_dir is None:
            raise ValueError("""The directory of alive trajectories is not defined ! Call ams.set_ams_dir""")
        self.z_maxs = (np.ones(self.n_rep) * (-np.infty)).tolist()

        existing_reps = [int(fi.split(".")[0].split("_")[-1]) for fi in os.listdir(self.ams_dir) if fi.startswith("rep_") and fi.endswith(".traj")]
        for i in existing_reps.copy():
            self.z_maxs[i] = np.max(np.loadtxt(self.ams_dir + "/rc_rep_" + str(i) + ".txt"))
            try:
                read_traj = read(filename=self.ams_dir + "/rep_" + str(i) + ".traj", format="traj", index=":")
                self._set_initialcond_dyn(read_traj[-1])
                self._until_r_or_p(i, len(read_traj))
                self._write_checkpoint()
            except UnknownFileTypeError:  # If loading fail just reinitialize the replica
                existing_reps.remove(i)

        to_initialize_rep = np.setdiff1d(np.arange(self.n_rep), existing_reps)
        for i in to_initialize_rep:
            self._pick_ini_cond(rep_index=i)
            self._until_r_or_p(i, 0)
            self._write_checkpoint()
        self.current_p = 0
        for i in range(self.n_rep):
            self.current_p += self.rep_weights[i][-1]
        self.initialized = True
        self._write_checkpoint()

    def _branch_replica(self, i, j, z_kill):
        """Branch replica i by copying replica j until z_kill and run the dynamics until it reaches either R or P"""
        if self.save_all and world.rank == 0:
            os.system("cp " + self.ams_dir + "/rep_" + str(i) + ".traj " + self.ams_dir + "/nr_rep_" + str(i) + "_killed_at_" + str(self.ams_it) + ".traj")
            os.system("cp " + self.ams_dir + "/rc_rep_" + str(i) + ".txt " + self.ams_dir + "/nr_rc_rep_" + str(i) + "_killed_at_" + str(self.ams_it) + ".txt")
            json_file = paropen(self.ams_dir + "/nr_rep_" + str(i) + "_killed_at_" + str(self.ams_it) + "_weights.txt", "w")
            weights = {"weights": self.rep_weights[i]}
            json.dump(weights, json_file, indent=4)
            json_file.close()
        self.rep_weights[i] = [self.rep_weights[j][-1]]  # self.rep_weights[j].copy()  # ou alors [self.rep_weights[j][-1]] ?
        if world.rank == 0:
            os.remove(self.ams_dir + "/rep_" + str(i) + ".traj")
            os.remove(self.ams_dir + "/rc_rep_" + str(i) + ".txt")
        barrier()
        branched_rep_z = np.loadtxt(self.ams_dir + "/rc_rep_" + str(j) + ".txt")
        branch_level = np.flatnonzero(branched_rep_z > z_kill)[0]  # First occurence of branched_rep_z above z_kill

        # Update the z_max 
        self.z_maxs[i] = np.max(branched_rep_z[: branch_level + 1])
        # Save branched traj until current point included
        f = paropen(self.ams_dir + "/rc_rep_" + str(i) + ".txt", "w")
        np.savetxt(f, branched_rep_z[: branch_level])
        f.close()

        read_traj = read(filename=self.ams_dir + "/rep_" + str(j) + ".traj", format="traj", index=":")
        write(self.ams_dir + "/rep_" + str(i) + ".traj", read_traj[: branch_level])

        self._set_initialcond_dyn(read_traj[branch_level])
        return branch_level + 1

    def _kill_reps(self):
        z_maxs_np = np.asarray(self.z_maxs)
        self.z_kill.append(np.sort(z_maxs_np[z_maxs_np > -np.infty])[self.k_min - 1]) # Ensure to take z_kill above infinity
        killed = np.flatnonzero(z_maxs_np - self.z_kill[-1] <= self.rc_threshold)
        self.killed.append(killed.tolist())
        alive = np.setdiff1d(np.arange(self.n_rep), killed)
        self._write_checkpoint()
        return killed, alive

    def _iteration(self):
        """Perform one iteration of the AMS algorithm"""
        if np.min(self.z_maxs) >= np.infty:
            self.finished = True
            self.success = True
            self._write_checkpoint()
            return False
        killed, alive = self._kill_reps()
        if len(killed) == self.n_rep:
            self.finished = True
            self.success = False
            self._write_checkpoint()
            return False
        for i in killed:
            if world.rank == 0:
                j = np.random.choice(alive)
            else:
                j = None
            j = broadcast(j)
            #len_branch = self._branch_replica(i, j, self.z_kill[-1])
            _ = self._branch_replica(i, j, self.z_kill[-1])
            self._until_r_or_p(i, 0) #always write the branching position via the first step of the dyn.run
            self._write_checkpoint()
        # update probability and weights
        for i in range(self.n_rep):
            self.rep_weights[i].append(self.rep_weights[i][-1] * ((self.n_rep - len(self.killed[-1])) / self.n_rep))
        self.current_p = self.current_p * ((self.n_rep - len(self.killed[-1])) / self.n_rep)
        return True

    def _finish_iteration(self):
        z_kill = self.z_kill[-1]
        alive = np.setdiff1d(np.arange(self.n_rep), self.killed[-1])
        for i in self.killed[-1]:
            rc_traj = np.loadtxt(self.ams_dir + "/rc_rep_" + str(i) + ".txt")
            if len(rc_traj.shape) == 0:
                rc_traj = rc_traj.reshape([1])
            if np.max(rc_traj) <= z_kill and (rc_traj[-1] <= -np.infty or rc_traj[-1] >= np.infty):
                if world.rank == 0:
                    j = np.random.choice(alive)
                else:
                    j = None
                j = broadcast(j)
                lentraj = self._branch_replica(i, j, z_kill)
            elif np.max(rc_traj) > z_kill and not (rc_traj[-1] <= -np.infty or rc_traj[-1] >= np.infty):
                read_traj = read(filename=self.ams_dir + "/rep_" + str(i) + ".traj", format="traj", index=":")
                lentraj = len(read_traj)
                self._set_initialcond_dyn(read_traj[-1])
            self._until_r_or_p(i, lentraj)
            self._write_checkpoint()
        # update probability and weights
        for i in range(self.n_rep):
            self.rep_weights[i].append(self.rep_weights[i][-1] * ((self.n_rep - len(self.killed[-1])) / self.n_rep))
        self.current_p = self.current_p * ((self.n_rep - len(self.killed[-1])) / self.n_rep)

    def p_ams(self):
        p = 0
        if not self.finished:
            print("AMS not run")
            return 0.0
        for i in range(self.n_rep):
            if self.z_maxs[i] >= np.infty:  # If in B
                p += self.rep_weights[i][-1]
        return p

    def _read_checkpoint(self):
        """Read the necessary information to restart an AMS run from the checkpoint file"""
        json_file = paropen(self.ams_dir + "/ams_checkpoint.txt", "r")
        checkpoint_data = json.load(json_file)
        json_file.close()
        self.z_maxs = checkpoint_data["z_maxs"]
        self.ams_it = checkpoint_data["iteration_number"]
        self.initialized = checkpoint_data["initialized"]
        self.finished = checkpoint_data["finished"]
        self.success = checkpoint_data["success"]
        self.rep_weights = checkpoint_data["rep_weights"]
        self.z_kill = checkpoint_data["z_kill"]
        self.killed = checkpoint_data["killed"]
        self.current_p = checkpoint_data["current_p"]
        self.current_rep = checkpoint_data["current_rep"]
        self.dyn.nsteps = checkpoint_data["current_step"]
        self.remaining_killed = checkpoint_data["remaining_killed"]
        self.alive = checkpoint_data["alive"]

    def _write_checkpoint(self):
        """Write information on the current state of AMS run to be able to restart"""
        checkpoint_data = {"z_maxs": self.z_maxs,
                           "rep_weights": self.rep_weights,
                           "killed": self.killed,
                           "alive": self.alive,
                           "z_kill": self.z_kill,
                           "remaining_killed": self.remaining_killed,
                           "iteration_number": self.ams_it,
                           "initialized": self.initialized,
                           "finished": self.finished,
                           "success": self.success,
                           "current_step": self.dyn.nsteps,
                           "current_rep": self.current_rep,
                           "current_p": self.current_p}
        json_file = paropen(self.ams_dir + "/ams_checkpoint.txt", "w")
        json.dump(checkpoint_data, json_file, indent=4, cls=NumpyEncoder)
        json_file.close()

    def _write_current_atoms(self):
        write("current_atoms.xyz", self.dyn.atoms, format="extxyz")
        write("POSCAR", self.dyn.atoms, format="vasp")

    def run(self, max_iter=-1):
        """Run AMS, should handle the restarts correctly"""
        if os.path.exists(self.ams_dir + "/ams_checkpoint.txt"):
            if self.verbose:
                parprint("Read checkpoint")
            self._read_checkpoint()
        barrier()  # Wait for all threads to read checkpoint
        if not self.initialized:
            # self.dyn.observers = []
            self._initialize()
        if self.verbose:
            parprint("Initialisation done")
        if os.path.exists(self.ams_dir + "/ams_checkpoint.txt") and self.initialized and not self.finished and self.dyn.nsteps == 1:
            # self.dyn.observers = []
            self._finish_iteration()
        continue_running = max_iter != 0
        with paropen(self.ams_dir + "/ams_progress.txt", "a") as progress_file:
            np.savetxt(progress_file, np.hstack(([self.ams_it, self.current_p, np.min(self.z_maxs), np.max(self.z_maxs)], self.z_maxs))[None, :])
            while continue_running:
                if self.verbose:
                    parprint("AMS iteration:", self.ams_it)
                # self.dyn.observers = []
                continue_running = self._iteration()
                self.ams_it += 1
                if continue_running:
                    np.savetxt(progress_file, np.hstack(([self.ams_it, self.current_p, np.min(self.z_maxs), np.max(self.z_maxs)], self.z_maxs))[None, :])
                if max_iter > 0 and self.ams_it >= max_iter:
                    self.finished = True
                    self.success = False
                    self._write_checkpoint()
                    continue_running = False

    def run_step_by_step(self, forces, energy, stress):
        """Run AMS by calling this function steps by steps"""
        if os.path.exists(self.ams_dir + "/ams_checkpoint.txt"):
            self._read_checkpoint()
        barrier()  # Wait for all threads to read checkpoint
        if not self.initialized:
            if self.current_rep is None:
                self.z_maxs = (np.ones(self.n_rep) * (-np.infty)).tolist()
                self.current_rep = 0
                self.dyn.nsteps = 0
        if not self.finished:
            traj = self.dyn.closelater(Trajectory(filename=self.ams_dir + "/rep_" + str(self.current_rep) + ".traj", mode="a", atoms=self.dyn.atoms))
            self.dyn.attach(traj.write, interval=self.cv_interval)
            z = self._rc()
            if z >= self.z_maxs[self.current_rep]:
                self.z_maxs[self.current_rep] = z
            f = paropen(self.ams_dir + "/rc_rep_" + str(self.current_rep) + ".txt", "a")
            f.write(str(z) + "\n")
            f.close()
            if self.dyn.nsteps > 0:
                self.dyn.atoms.calc.results['forces'] = forces
                self.dyn.atoms.calc.results['stress'] = stress
                self.dyn.atoms.calc.results['energy'] = energy
                self.dyn.atoms.calc.results['free_energy'] = energy
                self.dyn._2nd_half_step(forces)
            else:
                self.dyn.call_observers()
            if z > -np.infty and z < np.infty and self.dyn.nsteps <= self.max_length_iter:
                self.dyn._1st_half_step(forces)
                self.dyn.nsteps += 1
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self.dyn.atoms.calc.results['free_energy'] = np.zeros_like(energy)
                self._write_current_atoms()
                self._write_checkpoint()
                return False, False
            else:
                if not self.initialized:
                    if self.current_rep + 1 == self.n_rep:
                        self.initialized = True
                    else:
                        self.current_rep += 1
                        self.dyn.nsteps = 0
                        self._pick_ini_cond(rep_index=self.current_rep)
                if self.initialized: ## not an else here, IMPORTANT
                    if len(self.remaining_killed) == 0:
                        if np.min(self.z_maxs) >= np.infty:
                            self.finished = True
                            self.success = True
                            self._write_checkpoint()
                            return True, False
                        self.ams_it += 1
                        killed, self.alive = self._kill_reps()
                        if self.ams_it > 0:
                            for i in range(self.n_rep):
                                self.rep_weights[i].append(self.rep_weights[i][-1] * ((self.n_rep - len(self.killed[-1])) / self.n_rep))
                            self.current_p = self.current_p * ((self.n_rep - len(self.killed[-1])) / self.n_rep)
                        if len(killed) == self.n_rep:
                            self.finished = True
                            self.success = False
                            self._write_checkpoint()
                            return True, False
                        self.current_rep = killed.pop()
                        self.remaining_killed = killed
                    else:
                        self.current_rep = self.remaining_killed.pop()
                    j = np.random.choice(self.alive)
                    _ = self._branch_replica(self.current_rep, j, self.z_kill[-1])
                    self.dyn.nsteps = 0
                self._write_current_atoms()
                self._write_checkpoint()
                return False, True
        else:
            return True, False
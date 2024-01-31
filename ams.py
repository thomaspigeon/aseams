import numpy as np
import os
import shutil
import json
from ase.io import Trajectory, read
from ase.parallel import parprint, paropen, world, barrier


class AMS:
    """Running AMS"""

    def __init__(
        self,
        n_rep,
        k_min,
        dyn,
        xi,
        cv_interval=1,
        rc_threshold=1.0e-3,
        save_all=False,
    ):
        """
        Parameters:

        n_rep: int
            Number of replicas, must be strictly greater than 1

        k_min: int
            minimum number of replicas to kill at each iteration, must be superior or equal to 1 and strictly smaller
            than n_rep

        dyn: MolecularDynamics object
            Should be a stochastic dynamics
            TODO: Take a list of n_rep dynamics for parallel runs, check NEB calculations

        xi: CollectiveVariable object
            Object that allows to measure whether the dynamics is in reactant (R) state, in product (P) state and the
            progress on the transitions

        cv_interval: int
            The CV is evaluated every cv_interval time steps

        rc_threshold: float
            The biggest difference between two rc values so that they are considered identical.

        save_all: boolean
            whether all the trajectories of the replicas should be saved. If false, only the current state of the
            replicas is written in the AMS.current_replicas_dir
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
        #if type(dyn).__name__ != "Langevin":
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
        self.alive_traj_dir = None
        self.z_maxs = None
        self.ams_it = 0
        self.current_p = 1
        self.rep_weights = [[] for i in range(n_rep)]
        self.z_kill = []
        self.killed = []
        self.calc = dyn.atoms.calc
        self.rc_threshold = rc_threshold
        if save_all:
            self.non_reac_traj_dir = None

    def set_ini_cond_dir(self, ini_cond_dir="./ini_conds"):
        """Where the initial conditions for AMS will be written, raise error if the directory does not exist or is empty"""
        if not os.path.exists(ini_cond_dir):
            raise ValueError("""ini_cond_dir should exist, if not create it using initialconditionssampler.""")
        if len(os.listdir(ini_cond_dir)) == 0:
            raise ValueError("""ini_cond_dir should not be empty, use initialconditionssampler to write in it.""")
        self.ini_cond_dir = ini_cond_dir

    def set_progress_folder(self, progress_dir="./AMS_progress", clean=False):
        """Where the non reactive trajectories will be stored, if the directory does not exist, it will create it."""
        self.progress_dir = progress_dir
        if world.rank == 0:
            if clean and os.path.exists(self.progress_dir):
                shutil.rmtree(self.progress_dir)
            if not os.path.exists(self.progress_dir):
                os.mkdir(self.progress_dir)

    def set_non_reac_traj_dir(self, non_reac_traj_dir="./AMS_non_reactive_trajectories", clean=False):
        """Where the non reactive trajectories will be stored, if the directory does not exist, it will create it."""
        self.non_reac_traj_dir = non_reac_traj_dir
        if world.rank == 0:
            if clean and os.path.exists(self.non_reac_traj_dir):
                shutil.rmtree(self.non_reac_traj_dir)
            if not os.path.exists(self.non_reac_traj_dir):
                os.mkdir(self.non_reac_traj_dir)

    def set_alive_traj_dir(self, alive_traj_dir="./AMS_alive_trajectories", clean=False):
        """Where the alive trajectories will be stored, if the directory does not exist, it will create it.
        At the end of the run, if the estimated probability is not 0, these are reactive trajectories."""
        self.alive_traj_dir = alive_traj_dir
        if world.rank == 0:
            if clean and os.path.exists(self.alive_traj_dir):
                shutil.rmtree(self.alive_traj_dir)
            if not os.path.exists(self.alive_traj_dir):
                os.mkdir(self.alive_traj_dir)

    def _until_r_or_p(self, i):
        while not self.xi.in_r(self.dyn.atoms) and not self.xi.in_p(self.dyn.atoms):
            self.dyn.run(self.cv_interval)
            z = self.xi.rc(self.dyn.atoms)
            f = paropen(self.alive_traj_dir + "/rc_rep_" + str(i) + ".txt", "a")
            if self.xi.in_r(self.dyn.atoms):
                z = -1.0e8
            elif self.xi.in_p(self.dyn.atoms):
                z = 1.0e8
            f.write(str(z) + "\n")
            if z >= self.z_maxs[i]:
                self.z_maxs[i] = z
            f.close()

    def _initialize(self):
        """Run the N_rep replicas from the initial condition until it enters either R or P"""
        if self.ini_cond_dir is None:
            raise ValueError("""The directory of initial conditions is not defined ! Call ams.set_ini_cond_dir""")
        if self.alive_traj_dir is None:
            raise ValueError("""The directory of alive trajectories is not defined ! Call ams.set_alive_traj_dir""")
        if self.save_all is True and self.non_reac_traj_dir is None:
            raise ValueError("""The directory of non reactive trajectories is not defined ! Call ams.set_non_reac_traj_dir""")
        self.z_maxs = (np.ones(self.n_rep) * (-1.0e8)).tolist()
        i = 0
        if len(os.listdir(self.alive_traj_dir)) == 0:
            rep = -1
        else:
            rep = (len(os.listdir(self.alive_traj_dir)) // 2) - 1
            i = rep
            for j in range(i + 1):
                self.z_maxs[j] = np.max(np.loadtxt(self.alive_traj_dir + "/rc_rep_" + str(i) + ".txt"))
        while i < self.n_rep:
            if i > rep:
                ini_cond = np.random.choice([ini for ini in os.listdir(self.ini_cond_dir) if "_used" not in ini])
                atoms = read(self.ini_cond_dir + "/" + ini_cond, index=0)
                if world.rank == 0:
                    filename, file_extension = os.path.splitext(ini_cond)
                    os.rename(self.ini_cond_dir + "/" + ini_cond, self.ini_cond_dir + "/" + filename + "_used" + file_extension)
                self.dyn.atoms.set_scaled_positions(atoms.get_scaled_positions())
                self.dyn.atoms.set_momenta(atoms.get_momenta())
                self.dyn.atoms.set_calculator(self.calc)
                traj = self.dyn.closelater(Trajectory(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", mode="w", atoms=self.dyn.atoms))
            else:
                read_traj = read(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", format="traj", index=":")
                if world.rank == 0:
                    os.remove(self.alive_traj_dir + "/rep_" + str(i) + ".traj")
                self.dyn.atoms.set_scaled_positions(read_traj[-1].get_scaled_positions())
                self.dyn.atoms.set_momenta(read_traj[-1].get_momenta())
                self.dyn.atoms.set_calculator(self.calc)
                traj = self.dyn.closelater(Trajectory(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", mode="a", atoms=self.dyn.atoms))
                for at in read_traj[:-1]:
                    traj.write(at)
            z = self.xi.rc(self.dyn.atoms)
            f = paropen(self.alive_traj_dir + "/rc_rep_" + str(i) + ".txt", "a")
            if self.xi.in_r(self.dyn.atoms):
                z = -1.0e8
            elif self.xi.in_p(self.dyn.atoms):
                z = 1.0e8
            f.write(str(z) + "\n")
            f.close()
            self.dyn.attach(traj.write, interval=self.cv_interval)
            self.dyn.call_observers()
            self._until_r_or_p(i)
            self.dyn.close()
            self.dyn.observers.pop(-1)
            i += 1
            self._write_checkpoint()
        for i in range(self.n_rep):
            self.rep_weights[i].append(1 / self.n_rep)
        self.initialized = True
        self._write_checkpoint()

    def _branch_replica(self, i, j, z_kill):
        """Branch replica i by copying replica j until z_kill and run the dynamics until it reaches either R or P"""
        if self.save_all and world.rank == 0:
            os.system("cp " + self.alive_traj_dir + "/rep_" + str(i) + ".traj " + self.non_reac_traj_dir + "/rep_" + str(i) + "_killed_at_" + str(self.ams_it) + ".traj")
            json_file = paropen(self.non_reac_traj_dir + "/rep_" + str(i) + "_killed_at_" + str(self.ams_it) + "_weights.txt", "w")
            weights = {"weights": self.rep_weights[i]}
            json.dump(weights, json_file, indent=4)
            json_file.close()
        self.rep_weights[i] = []
        if world.rank == 0:
            os.remove(self.alive_traj_dir + "/rep_" + str(i) + ".traj")
            os.remove(self.alive_traj_dir + "/rc_rep_" + str(i) + ".txt")
        read_traj = read(filename=self.alive_traj_dir + "/rep_" + str(j) + ".traj", format="traj", index=":")
        k = 0
        traj = self.dyn.closelater(Trajectory(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", mode="w", atoms=read_traj[k]))
        f = paropen(self.alive_traj_dir + "/rc_rep_" + str(i) + ".txt", "a")
        while np.abs(self.xi.rc(read_traj[k]) - z_kill) <= self.rc_threshold:
            traj.write(read_traj[k])
            if self.xi.rc(read_traj[k]) >= self.z_maxs[i]:
                self.z_maxs[i] = self.xi.rc(read_traj[k])
            f.write(str(self.xi.rc(read_traj[k])) + "\n")
            k += 1
        f.write(str(self.xi.rc(read_traj[k])) + "\n")
        f.close()
        self.dyn.close()
        self.dyn.atoms.set_scaled_positions(read_traj[k].get_scaled_positions())
        self.dyn.atoms.set_momenta(read_traj[k].get_momenta())
        self.dyn.atoms.set_calculator(self.calc)
        traj = self.dyn.closelater(Trajectory(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", mode="a", atoms=self.dyn.atoms))
        self.dyn.attach(traj.write, interval=self.cv_interval)
        self.dyn.call_observers()

    def _iteration(self):
        """Perform one iteration of the AMS algorithm"""
        if np.min(self.z_maxs) >= 1.0e8:
            self.finished = True
            self.success = True
            self._write_checkpoint()
            return False
        z_kill = np.sort(self.z_maxs)[self.k_min - 1]
        self.z_kill.append(z_kill)
        killed = np.where(np.abs(self.z_maxs - z_kill) <= self.rc_threshold)[0].tolist()
        self.killed.append(killed.copy())
        alive = np.setdiff1d(np.arange(self.n_rep), killed)
        self.current_p = self.current_p * ((self.n_rep - len(self.killed[-1])) / self.n_rep)
        self._write_checkpoint()
        if len(killed) == self.n_rep:
            self.finished = True
            self.success = False
            return False
        while len(killed) > 0:
            i = killed.pop()
            j = np.random.choice(alive)
            self._branch_replica(i, j, z_kill)
            self._until_r_or_p(i)
            self.dyn.close()
            self.dyn.observers.pop(-1)
            self._write_checkpoint()
        for i in range(self.n_rep):
            self.rep_weights[i].append((self.n_rep - len(self.killed[-1])) / self.n_rep)
        return True

    def _finish_iteration(self):
        z_kill = self.z_kill[-1]
        killed = self.killed[-1].copy()
        alive = np.setdiff1d(np.arange(self.n_rep), killed)
        while len(killed) > 0:
            i = killed.pop()
            rc_traj = np.loadtxt(self.alive_traj_dir + "/rc_rep_" + str(i) + ".txt")
            if len(rc_traj.shape) == 0:
                rc_traj.reshape([1])
            if np.max(rc_traj) <= z_kill and (rc_traj[-1] <= -1.0e8 or rc_traj[-1] >= 1.0e8):
                j = np.random.choice(alive)
                self._branch_replica(i, j, z_kill)
                self._until_r_or_p(i)
            elif np.max(rc_traj) > z_kill and not (rc_traj[-1] <= -1.0e8 or rc_traj[-1] >= 1.0e8):
                read_traj = read(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", format="traj", index=":")
                self.dyn.atoms.set_scaled_positions(read_traj[-1].get_scaled_positions())
                self.dyn.atoms.set_momenta(read_traj[-1].get_momenta())
                traj = self.dyn.closelater(Trajectory(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", mode="a", atoms=self.dyn.atoms))
                self.dyn.attach(traj.write, interval=self.cv_interval)
                self._until_r_or_p(i)
                self.dyn.close()
                self.dyn.observers.pop(-1)
            self._write_checkpoint()
        for i in range(self.n_rep):
            self.rep_weights[i].append((self.n_rep - len(self.killed[-1])) / self.n_rep)

    def _read_checkpoint(self):
        """Read the necessary information to restart an AMS run from the checkpoint file"""
        json_file = paropen(self.progress_dir + "/ams_checkpoint.txt", "r")
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

    def _write_checkpoint(self):
        """Write information on the current state of AMS run to be able to restart"""
        checkpoint_data = {"z_maxs": self.z_maxs, "iteration_number": self.ams_it, "initialized": self.initialized, "finished": self.finished, "success": self.success, "rep_weights": self.rep_weights, "z_kill": self.z_kill, "killed": self.killed, "current_p": self.current_p}
        json_file = paropen(self.progress_dir + "/ams_checkpoint.txt", "w")
        json.dump(checkpoint_data, json_file, indent=4)
        json_file.close()

    def run(self):
        """Run AMS, should handle the restarts correctly"""
        if os.path.exists(self.progress_dir + "/ams_checkpoint.txt"):
            parprint("Read checkpoint")
            self._read_checkpoint()
        barrier()  # Wait for all threads to read checkpoint
        if not self.initialized:
            self.dyn.observers = []
            self._initialize()
        if os.path.exists(self.progress_dir + "/ams_checkpoint.txt") and self.initialized and not self.finished and self.dyn.nsteps == 1:
            self.dyn.observers = []
            self._finish_iteration()
        continue_running = True
        with paropen(self.progress_dir + "/ams_progress.txt", "a") as progress_file:
            while continue_running:
                self.dyn.observers = []
                continue_running = self._iteration()
                np.savetxt(progress_file, np.hstack(([self.ams_it, self.current_p], self.z_maxs))[None, :])
                self.ams_it += 1

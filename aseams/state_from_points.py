#!python3


import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial.distance import cdist


def get_circumsphere(S):
    """
    Computes the circumsphere of a set of points

    Parameters
    ----------
    S : (M, N) ndarray, where 1 <= M <= N + 1
            The input points

    Returns
    -------
    C, r2 : ((2) ndarray, float)
            The center and the squared radius of the circumsphere
    """

    U = S[1:] - S[0]
    B = np.sqrt(np.square(U).sum(axis=1))
    U /= B[:, None]
    B /= 2
    C = np.dot(np.linalg.solve(np.inner(U, U), B), U)
    r2 = np.square(C).sum()
    C = S[0] + C
    return C, r2


def get_bounding_ball(S, epsilon=1e-7, rng=np.random.default_rng()):
    """
    Computes the smallest bounding ball of a set of points

    Parameters
    ----------
    S : (M, N) ndarray, where 1 <= M <= N + 1
            The input points

    epsilon : float
            Tolerance used when testing if a set of point belongs to the same
            sphere. Default is 1e-7

    rng : np.random.Generator
        Pseudo-random number generator used internally. Default is the default
        one provided by np.

    Returns
    -------
    C, r2 : ((2) ndarray, float)
            The center and the squared radius of the circumsphere
    """

    # Iterative implementation of Welzl's algorithm, see
    # "Smallest enclosing disks (balls and ellipsoids)" Emo Welzl 1991

    def circle_contains(D, p):
        c, r2 = D
        return np.square(p - c).sum() <= r2

    def get_boundary(R):
        if len(R) == 0:
            return np.zeros(S.shape[1]), 0.0

        if len(R) <= S.shape[1] + 1:
            return get_circumsphere(S[R])

        c, r2 = get_circumsphere(S[R[: S.shape[1] + 1]])
        if np.all(np.fabs(np.square(S[R] - c).sum(axis=1) - r2) < epsilon):
            return c, r2

    class Node(object):
        def __init__(self, P, R):
            self.P = P
            self.R = R
            self.D = None
            self.pivot = None
            self.left = None
            self.right = None

    def traverse(node):
        stack = [node]
        while len(stack) > 0:
            node = stack.pop()

            if len(node.P) == 0 or len(node.R) >= S.shape[1] + 1:
                node.D = get_boundary(node.R)
            elif node.left is None:
                pivot_index = rng.integers(len(node.P))
                node.pivot = node.P[pivot_index]
                node.left = Node(node.P[:pivot_index] + node.P[pivot_index + 1 :], node.R)
                stack.extend((node, node.left))
            elif node.right is None:
                if circle_contains(node.left.D, S[node.pivot]):
                    node.D = node.left.D
                else:
                    node.right = Node(node.left.P, node.R + [node.pivot])
                    stack.extend((node, node.right))
            else:
                node.D = node.right.D
                node.left, node.right = None, None

    # S = S.astype(float, copy=False)
    root = Node(list(range(S.shape[0])), [])
    traverse(root)
    return root.D


class DensityState:
    """
    A class that define a state whatever the point is close enough from a set of points
    """

    def __init__(self, X, state_level=1.0, N_max=None, sigma_factor=1.1, k_max=4, metric="minkowski"):
        """
        Initialize the state
        X is the set of points within the state
        state_level is the fraction of points to be considered core set
        N_max is the max number of points to be considered
        sigma_factor is the extra radius taken for sigma boundary
        kmax is the number of neighbours taken for construction of the spanning graph
        """

        n_points, dim = X.shape
        if N_max is None:
            N_max = n_points
        else:
            N_max = min(N_max, n_points)

        # Select a random subset of the points
        X = X[np.random.choice(n_points, N_max, replace=False), :]

        distances, _ = NearestNeighbors(n_neighbors=k_max, algorithm="auto", metric=metric).fit(X).kneighbors(X)
        self.state_inds = np.argsort(distances[:, -1])[: int(state_level * N_max)]  # We take the ones with smallest volume
        self.X = X[self.state_inds, :]
        self.state_nbrs = NearestNeighbors(n_neighbors=k_max, algorithm="ball_tree", metric=metric).fit(self.X)
        self.kmax = k_max

        # TODO : Check connected components
        connectivity_graph = self.state_nbrs.kneighbors_graph(mode="distance")
        n_comps, labels = connected_components(connectivity_graph)
        if n_comps > 1:
            print("WARNING there is {} connected components, increase kmax or split the set of points".format(n_comps))

        spanning_tree = minimum_spanning_tree(connectivity_graph)
        self.state_radius = (spanning_tree + spanning_tree.T).max(axis=1).toarray()[:, 0]  # Find radius in order to get connected graph
        self.sigma_factor = max(np.abs(sigma_factor), 1.0)

    def __call__(self, x):
        """
        Return True or false if points x is within state or not
        connectivity_level ask the number of points to be close enough to be considered as within the state
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        dist, inds = self.state_nbrs.kneighbors(x)
        return (dist[:, : self.kmax] <= self.state_radius[inds[:, : self.kmax]]).any(axis=1)

    def sigma(self, x):
        """
        Return True or false if points x is within state or not
        connectivity_level ask the number of points to be close enough to be considered as within the state
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        dist, inds = self.state_nbrs.kneighbors(x)
        return (dist[:, : self.kmax] <= self.sigma_factor * self.state_radius[inds[:, : self.kmax]]).any(axis=1)

    def dist_to_state(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        dist, inds = self.state_nbrs.kneighbors(x)
        d = dist[:, : self.kmax] - self.state_radius[inds[:, : self.kmax]]  # Remove distance to point

        return np.where(d < 0, 0.0, d).min(axis=1).reshape(-1, 1)


class EnclosingCircleState:
    """
    Define state as the smallest enclosing circle of all points
    """

    def __init__(self, X, state_level=1.0, N_max=None, sigma_factor=1.1, k_max=4, metric="euclidean"):
        """
        Initialize the state
        X is the set of points within the state
        state_level is the fraction of points to be considered core set
        N_max is the max number of points to be considered
        sigma_factor is the extra radius taken for sigma boundary
        kmax is the number of neighbours taken for construction of the spanning graph
        """

        n_points, dim = X.shape
        if N_max is None:
            N_max = n_points
        else:
            N_max = min(N_max, n_points)

        # Select a random subset of the points
        X = X[np.random.choice(n_points, N_max, replace=False), :]

        self.metric = metric

        distances, _ = NearestNeighbors(n_neighbors=k_max, algorithm="auto", metric=metric).fit(X).kneighbors(X)
        self.state_inds = np.argsort(distances[:, -1])[: int(state_level * N_max)]  # We take the ones with smallest volume
        self.X = X[self.state_inds, :]
        C, r2 = get_bounding_ball(X[self.state_inds, :])
        self.C = C.reshape(1, -1)
        self.r = np.sqrt(r2)
        self.sigma_factor = max(np.abs(sigma_factor), 1.0)

    def __call__(self, x):
        """
        Return True or false if points x is within state or not
        connectivity_level ask the number of points to be close enough to be considered as within the state
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        dist = cdist(x, self.C, metric=self.metric)
        return (dist <= self.r)[:, 0]

    def sigma(self, x):
        """
        Return True or false if points x is within state or not
        connectivity_level ask the number of points to be close enough to be considered as within the state
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        dist = cdist(x, self.C, metric=self.metric)
        return (dist <= self.sigma_factor * self.r)[:, 0]

    def dist_to_state(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        dist = cdist(x, self.C, metric=self.metric) - self.r
        return np.where(dist < 0, 0.0, dist)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle  # $matplotlib/patches.py

    def circle(xy, radius, color="lightsteelblue", facecolor="none", alpha=1, ax=None):
        """add a circle to ax= or current axes"""
        # from .../pylab_examples/ellipse_demo.py
        e = Circle(xy=xy, radius=radius)
        if ax is None:
            ax = plt.gca()  # ax = subplot( 1,1,1 )
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_edgecolor(color)
        e.set_facecolor(facecolor)  # "none" not None
        e.set_alpha(alpha)

    X = np.random.randn(15, 2)
    test = np.random.randn(6, 2)
    # X = np.vstack((np.random.randn(10, 2), np.array([25.0, 25.0])[None, :] + np.random.randn(10, 2)))

    x = np.linspace(-4, 4, 250)
    y = np.linspace(-4, 4, 250)
    xx, yy = np.meshgrid(x, y)
    X_eval = np.vstack([xx.ravel(), yy.ravel()]).T

    # DensityState
    fig, ax = plt.subplots(num="Density State")
    state = DensityState(X, sigma_factor=1.1)

    res = state.dist_to_state(X_eval)
    h = ax.contourf(x, y, res.reshape(x.shape[0], y.shape[0]))
    ax.axis("scaled")

    print("Test points inside Density State", state(test))

    ax.scatter(X[:, 0], X[:, 1])
    ax.scatter(test[:, 0], test[:, 1], marker="x")
    for n in range(X.shape[0]):
        circle(state.X[n, :], state.state_radius[n], ax=ax)

    # # Circle state
    fig, ax = plt.subplots(num="Circle State")
    cState = EnclosingCircleState(X, sigma_factor=1.1)
    res = cState.sigma(X_eval)

    print("Test points inside Circle State", cState(test))

    h = ax.contourf(x, y, res.reshape(x.shape[0], y.shape[0]))
    ax.scatter(X[:, 0], X[:, 1])
    ax.scatter(cState.C[:, 0], cState.C[:, 1])
    ax.scatter(test[:, 0], test[:, 1], marker="x")
    circle(cState.C[0, :], cState.r, color="red")

    plt.show()

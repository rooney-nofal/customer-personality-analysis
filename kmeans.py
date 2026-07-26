"""
K-means clustering implemented from scratch with NumPy.

Customer Personality Analysis - La Plateforme
Author: Rooney Nofal
"""

import numpy as np


class KMeans:
    """
    K-means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (K) to find.
    max_iter : int
        Maximum number of iterations of the assign/update loop.
    tol : float
        Convergence tolerance: if centroids move less than this, stop.
    random_state : int
        Seed for the random initialization, for reproducible results.

    Attributes
    ----------
    centroids : ndarray of shape (n_clusters, n_features)
        Final centroid coordinates.
    labels : ndarray of shape (n_samples,)
        Cluster index assigned to each point.
    inertia_ : float
        Sum of squared distances of points to their assigned centroid.
    n_iter_ : int
        Number of iterations actually run.
    """

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.n_iter_ = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _euclidean_distance(self, points, centroid):
        """Straight-line distance from every point to one centroid."""
        return np.sqrt(np.sum((points - centroid) ** 2, axis=1))

    def _initialize_centroids(self, X):
        """Pick K distinct real data points as starting centroids (Forgy method)."""
        np.random.seed(self.random_state)
        random_indices = np.random.choice(
            X.shape[0], self.n_clusters, replace=False
        )
        return X[random_indices]

    def _compute_distances(self, X):
        """Distance matrix: one row per point, one column per centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = self._euclidean_distance(X, self.centroids[k])
        return distances

    def _assign_clusters(self, X):
        """Assign each point to its nearest centroid."""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """Move each centroid to the mean position of the points assigned to it."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            points_in_cluster = X[labels == k]
            if len(points_in_cluster) == 0:
                # Empty cluster: keep the centroid where it is.
                new_centroids[k] = self.centroids[k]
            else:
                new_centroids[k] = points_in_cluster.mean(axis=0)
        return new_centroids

    def _compute_inertia(self, X, labels):
        """Sum of squared distances between points and their own centroid."""
        total = 0.0
        for k in range(self.n_clusters):
            points_in_cluster = X[labels == k]
            if len(points_in_cluster) > 0:
                total += np.sum((points_in_cluster - self.centroids[k]) ** 2)
        return total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        Run the K-means algorithm on X.

        Repeats the assign / update loop until the centroids stop moving
        (shift <= tol) or max_iter is reached.
        """
        X = np.asarray(X, dtype=float)
        self.centroids = self._initialize_centroids(X)

        for i in range(self.max_iter):
            # Step 1: assign every point to its nearest centroid.
            self.labels = self._assign_clusters(X)

            # Step 2: move the centroids to the mean of their points.
            new_centroids = self._update_centroids(X, self.labels)

            # Step 3: measure how far the centroids moved this round.
            shift = np.sqrt(np.sum((new_centroids - self.centroids) ** 2))
            self.centroids = new_centroids
            self.n_iter_ = i + 1

            # Step 4: stop early if they barely moved (convergence).
            if shift <= self.tol:
                break

        # Final assignment with the final centroids, then measure quality.
        self.labels = self._assign_clusters(X)
        self.inertia_ = self._compute_inertia(X, self.labels)
        return self

    def predict(self, X):
        """Assign new, unseen points to the clusters already learned."""
        X = np.asarray(X, dtype=float)
        return self._assign_clusters(X)

    def fit_predict(self, X):
        """Convenience method: fit the model and return the labels."""
        self.fit(X)
        return self.labels
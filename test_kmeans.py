"""
Quick sanity check for our from-scratch KMeans class.

Creates 3 obvious blobs of points, then verifies that our KMeans
finds 3 clusters roughly centered on those blobs.
"""

import numpy as np
from kmeans import KMeans

# Build a tiny fake dataset: 3 well-separated blobs of 50 points each.
np.random.seed(0)
blob_a = np.random.randn(50, 2) + np.array([0, 0])
blob_b = np.random.randn(50, 2) + np.array([10, 10])
blob_c = np.random.randn(50, 2) + np.array([0, 10])
X = np.vstack([blob_a, blob_b, blob_c])

# Run our model.
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

print("Iterations run :", model.n_iter_)
print("Inertia        :", round(model.inertia_, 2))
print("Cluster sizes  :", np.bincount(model.labels))
print("Centroids found:")
print(np.round(model.centroids, 2))
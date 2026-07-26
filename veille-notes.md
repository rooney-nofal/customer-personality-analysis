# Veille - Clustering Algorithms

 
## 1. K-means

K is the number of groups (clusters) we want to find, chosen in advance.

Steps:
1. Place K random centroids among the data.
2. Each data point joins its nearest centroid (by straight-line distance).
3. Each centroid moves to the average position of the points assigned to it.
4. Repeat steps 2-3 until centroids stop moving (convergence).

**Strengths:** simple, fast, works well on large datasets.
**Weaknesses:** need to choose K in advance, sensitive to random starting centroids, struggles with non-round-shaped groups.


## 2. CAH (Classification Ascendante Hierarchique / Agglomerative Hierarchical Clustering)

Steps:
1. Start with every data point as its own cluster.
2. Find the two closest clusters and merge them into one.
3. Repeat step 2 until only one cluster remains (everyone merged).
4. The merging history is drawn as a tree diagram called a dendrogram.

Key difference from K-means: no need to choose K in advance. Build the full tree first, then cut it at the height that gives the desired number of clusters.

**Strengths:** no need to guess K upfront, produces a visual dendrogram, works well on small-to-medium datasets.
**Weaknesses:** slow on large datasets, merges cannot be undone once made.


## 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

Finds clusters based on density: wherever points are densely packed, that's a cluster. Isolated points are noise.

Two hyperparameters (chosen by the data scientist):
- **eps**: the radius to look around each point.
- **min_samples**: minimum neighbors within eps for a point to be a core point.

How it works: a point with at least min_samples neighbors within eps is a core point. Connected core points chain together to form a cluster. Points near a cluster but not dense themselves are border points. Points belonging to nothing are noise.

**Strengths:** finds clusters of any shape (not just round), automatically detects outliers/noise.
**Weaknesses:** choosing eps and min_samples is tricky (bad values = everything is noise or everything is one cluster), struggles when clusters have very different densities.


## 4. Choosing the optimal number of clusters

### Elbow Method
For each candidate K, run K-means and measure the **inertia** (total squared distance between each point and its centroid). Small inertia = compact clusters.

Inertia always decreases as K grows (if K = number of points, inertia = 0, which is useless). So we don't pick the smallest inertia. Instead we plot inertia vs K and look for the **elbow**: the point where the curve stops dropping steeply and flattens out. Beyond the elbow, adding clusters barely helps.

### Silhouette Score
For each point, compares (a) how close it is to its own cluster's members vs (b) how far it is from the nearest other cluster. Gives a value between -1 and +1, averaged over all points.

- close to +1: points well placed, clusters clean and separated
- around 0: clusters overlap, boundaries unclear
- negative: points likely in the wrong cluster, bad clustering

Pick the K with the highest average silhouette score.

### Using both together
The elbow gives a quick visual candidate; the silhouette gives a precise number to confirm it or break ties. The silhouette also works for CAH and DBSCAN, which makes it useful for comparing different algorithms against each other.


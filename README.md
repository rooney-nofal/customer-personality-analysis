# Customer Personality Analysis

Unsupervised clustering of a grocery shop's customer base, to answer one
question: **who buys what?**

Project carried out at **La Plateforme** (Marseille) as part of the AI programme.
The K-means algorithm is implemented from scratch in `kmeans.py` and validated
against Scikit-Learn.

---

## Table of contents

1. [Context and problem](#1-context-and-problem)
2. [Repository structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Veille — clustering algorithms](#4-veille--clustering-algorithms)
5. [Veille — choosing and evaluating clusters](#5-veille--choosing-and-evaluating-clusters)
6. [The data](#6-the-data)
7. [Method](#7-method)
8. [Results](#8-results)
9. [Customer segments](#9-customer-segments)
10. [Conclusion](#10-conclusion)
11. [Limitations](#11-limitations)

---

## 1. Context and problem

A business does not sell to "customers" in general. It sells to distinct groups
of people who differ in income, family situation, habits and taste. Marketing a
new product to the entire customer base is expensive and inefficient: it is far
better to identify which segment is most likely to buy, and target that segment.

The difficulty is that nobody has labelled the customers. There is no column
saying "this person is a bargain hunter". The groups have to be discovered from
the data itself. That is **unsupervised learning**, and more precisely
**clustering**.

This project:

- researches three clustering algorithms (K-means, CAH, DBSCAN) and the methods
  for choosing and evaluating the number of clusters;
- implements K-means from scratch and validates it on the Iris dataset against
  Scikit-Learn;
- applies three clustering algorithms to a real grocery customer dataset;
- compares the algorithms and profiles the resulting segments.

---

## 2. Repository structure

```
customer-personality-analysis/
├── data/
│   ├── marketing_campaign.csv        raw dataset (2240 customers, 29 columns)
│   ├── customers_clean.csv           after cleaning and feature engineering
│   └── customers_with_clusters.csv   final data with the cluster label
├── notebooks/
│   ├── 01_iris_kmeans.ipynb          our K-means vs Scikit-Learn on Iris
│   ├── 02_customer_exploration.ipynb exploration, cleaning, feature engineering
│   └── 03_customer_modeling.ipynb    PCA, 3 algorithms, comparison, profiling
├── kmeans.py                         K-means implemented from scratch
├── test_kmeans.py                    sanity check on synthetic blobs
├── veille-notes.md                   research notes
├── requirements.txt                  exact library versions
└── README.md
```

---

## 3. Installation

```bash
git clone https://github.com/rooney-nofal/customer-personality-analysis.git
cd customer-personality-analysis

python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

Then open the notebooks in order (01, 02, 03) and select the `venv` kernel.

Built with Python 3.14.6, pandas, NumPy, Matplotlib, seaborn, Scikit-Learn and
Jupyter.

---

## 4. Veille — clustering algorithms

### 4.1 K-means

K is the number of clusters, chosen in advance.

1. Place K centroids at random positions among the data points.
2. Assign every point to its nearest centroid (Euclidean distance).
3. Move each centroid to the mean position of the points assigned to it.
4. Repeat steps 2 and 3 until the centroids stop moving (convergence).

**Strengths.** Simple, fast, scales well to large datasets, easy to interpret.

**Weaknesses.** K must be chosen in advance; the result depends on the random
initialisation; it assumes roughly spherical clusters of similar size; centroids
are means, so it is sensitive to outliers.

Source: *K-means : Focus sur cet algorithme de Clustering & Machine Learning*
(knowledge base provided with the brief).

### 4.2 CAH — Classification Ascendante Hiérarchique

Agglomerative hierarchical clustering.

1. Start with every point as its own cluster.
2. Merge the two closest clusters.
3. Repeat until a single cluster remains.
4. The merge history is drawn as a **dendrogram**.

The key difference from K-means: the number of clusters is **not** chosen in
advance. The full tree is built first, then cut at the height that gives the
desired number of groups.

Several linkage criteria exist (single, complete, average, Ward). This project
uses **Ward linkage**, which merges the pair of clusters that minimises the
increase in within-cluster variance — the criterion closest in spirit to
K-means, which makes the comparison between the two methods meaningful.

**Strengths.** No need to fix k upfront; the dendrogram is a strong visual
explanation; works well on small and medium datasets.

**Weaknesses.** Computationally expensive as the number of points grows; a merge
can never be undone.

Source: *Machine Learning & Clustering : Focus sur l'algorithme CAH*.

### 4.3 DBSCAN — Density-Based Spatial Clustering of Applications with Noise

Clusters are defined by **density**: wherever points are packed closely
together, that is a cluster; isolated points belong to nothing.

Two hyperparameters, both chosen by the analyst:

- `eps` — the radius of the neighbourhood searched around each point;
- `min_samples` — the minimum number of neighbours within `eps` for a point to
  count as a core point.

A point with at least `min_samples` neighbours within `eps` is a **core point**.
Connected core points chain together to form a cluster. Points close to a
cluster but not dense themselves are **border points**. Everything else is
**noise**.

**Strengths.** Finds clusters of arbitrary shape, not just spherical ones; the
number of clusters is discovered rather than imposed; outliers are detected
automatically instead of being forced into a group.

**Weaknesses.** `eps` and `min_samples` are difficult to tune — poor values give
either one giant cluster or almost pure noise; it struggles when different
clusters have very different densities.

Choosing the hyperparameters in practice: `min_samples` is often set to roughly
twice the number of dimensions, and `eps` is read off the **k-distance plot**
(distance to the k-th nearest neighbour, sorted; the elbow of that curve is a
reasonable value for `eps`).

Source: *Machine Learning & Clustering : Focus sur l'Algorithme DBSCAN*.

---

## 5. Veille — choosing and evaluating clusters

### 5.1 Elbow method

For each candidate k, run K-means and record the **inertia**: the sum of squared
distances between each point and its assigned centroid. Low inertia means
compact clusters.

Inertia always decreases as k increases — with k equal to the number of points,
inertia is zero and the model is useless. So the minimum is not the answer.
Instead, inertia is plotted against k and the **elbow** is located: the point
where the curve stops falling steeply and flattens out. Beyond it, adding
clusters buys very little.

### 5.2 Silhouette score

For each point, the silhouette compares how close it is to the other members of
its own cluster against how far it is from the nearest neighbouring cluster. The
result lies between -1 and +1, and is averaged over all points:

- close to **+1** — well-placed points, clean and well-separated clusters;
- around **0** — overlapping clusters, unclear boundaries;
- **negative** — points are probably assigned to the wrong cluster.

The k with the highest average silhouette is the preferred one.

### 5.3 Davies-Bouldin index

The average, over all clusters, of the similarity between each cluster and its
most similar one, where similarity increases with cluster spread and decreases
with the distance between cluster centres. **Lower is better**, 0 being perfect.

### 5.4 Why more than one metric is needed

These are all *internal* metrics: they measure geometry, not usefulness. As the
results below show, an algorithm can obtain the best Davies-Bouldin index while
producing a partition that is worthless for marketing, simply because it created
one dominant cluster and a few specks. Metrics guide the decision; they do not
replace it.

---

## 6. The data

`marketing_campaign.csv` — 2240 customers, 29 tab-separated columns, covering:

- **identity** — year of birth, education, marital status, income, children at home;
- **spending over two years** — wine, fruit, meat, fish, sweets, gold products;
- **purchase channels** — web, catalogue, store, discounted deals, web visits per month;
- **marketing campaigns** — responses to five past campaigns, complaints.

Source: Customer Personality Analysis dataset (Kaggle, CC0 public domain), as
linked in the project brief.

### Cleaning

| Step | Detail |
|---|---|
| Column names | the file is space-padded, so names and text values were stripped and `Income` converted back to a numeric type |
| Missing values | `Income` missing for 24 customers (1.07%); those rows were dropped rather than imputed, to avoid inventing income figures that would shift the centroids |
| Outliers | 3 customers with impossible ages (birth years in the 1890s) and 1 with an income above 600,000 were removed as data entry errors — K-means centroids are means and are dragged by extreme values |
| Constant columns | `Z_CostContact` and `Z_Revenue` hold a single value for every row and carry no information |
| Final size | **2212 customers** |

### Feature engineering

New, more interpretable features were derived: `Age`, `Customer_Days` (time since
enrolment), `Total_Spending` (sum of the six product categories),
`Total_Purchases` (sum of the four channels), `Children` (kids + teenagers),
`Family_Size`, `Is_Parent`, and simplified `Education_Level` and `Living_With`
categories.

### What the exploration showed

- Income and total spending are strongly related (**r = 0.79**).
- The number of children is negatively related to spending (**r = -0.50**).
- Income is negatively related to website visits (**r = -0.65**): wealthier
  customers browse the site less and buy through the store and catalogue.
- **Wine dominates every other category** (675,296 spent in total), followed by
  meat (369,470); fruit and sweets are marginal (around 58,000 each).

---

## 7. Method

1. **Feature selection** — twelve interpretable features were kept, dropping
   redundant columns (`Total_Spending` already sums the `Mnt...` columns) and
   the campaign-response columns, which describe past marketing outcomes rather
   than customer behaviour.
2. **Standardisation** — every feature scaled to mean 0 and standard deviation 1,
   so that income (in tens of thousands) does not dominate the distance
   calculation against the number of children (0 to 3).
3. **PCA** — projection onto three principal components, which makes distances
   more meaningful and allows the clusters to be visualised.
4. **Clustering** — our own K-means, CAH (Ward) and DBSCAN, all applied to the
   same reduced space so that the comparison is fair.
5. **Evaluation** — silhouette score and Davies-Bouldin index.
6. **Profiling** — cluster labels attached back to the original unscaled data so
   the segments can be described in euros, years and counts.

---

## 8. Results

### 8.1 Validation of the from-scratch K-means (Iris)

| | Our implementation | Scikit-Learn |
|---|---|---|
| Iterations | 7 | 4 |
| Inertia | 140.033 | 139.821 |
| Cluster sizes | 56 / 50 / 44 | 53 / 50 / 47 |

**Adjusted Rand Index between the two labelings: 0.941** — the two models agree
almost perfectly. The small difference in inertia comes from Scikit-Learn
restarting ten times (`n_init=10`) and keeping its best run, whereas ours runs
once from a single seed.

Running our model across several seeds (0, 1, 7, 42, 99, 123, 2024) gave
inertias of 140.90, 140.03 and, for seed 2024, 197.47 with sizes [13, 37, 100] —
a clear demonstration of K-means' sensitivity to initialisation, and of why
Scikit-Learn restarts several times by default.

### 8.2 Comparison of the three algorithms on the customer data

| Algorithm | Clusters | Noise points | Silhouette (higher better) | Davies-Bouldin (lower better) |
|---|---|---|---|---|
| **Our K-means** | 4 | 0 | **0.316** | 1.111 |
| CAH (Ward) | 4 | 0 | 0.266 | 1.269 |
| DBSCAN | 4 (nominally) | 86 | 0.137 | 0.645 |

**K-means** gave the best silhouette and four balanced segments (23%, 28%, 27%,
22% of the base).

**CAH** produced a very similar partition with a slightly lower silhouette. Two
independent methods converging on the same structure is good evidence that the
segments are real rather than an artefact of one algorithm.

**DBSCAN** nominally returned four clusters, but their sizes were **2096, 15, 9
and 6**, plus 86 points labelled as noise: one large mass and three tiny density
pockets. Its Davies-Bouldin index is the best of the three, which is precisely
the trap described in section 5.4 — a single dominant cluster is trivially
compact, so the metric looks excellent while the partition is useless.

**Why DBSCAN behaves this way here.** Density-based clustering needs empty space
between groups. Customer data of this kind forms a continuous cloud: income, age
and spending vary smoothly, with no gaps. The segments are regions of a
continuum, not physically separated islands. This is not a failure of the
algorithm — it is informative about the shape of the data, and it explains why a
partitioning method such as K-means is the appropriate tool here.

### 8.3 Choice of k

The silhouette score peaked at **k = 2**, while the inertia curve bent around
k = 3-4. We chose **k = 4**: it sits at the elbow, keeps a reasonable silhouette,
and above all produces segments that can be acted on. A two-way split says
little more than "spends a lot" versus "spends little", which is not a
segmentation a marketing team can use. This is a deliberate trade-off between a
geometric metric and the business question, and it is stated as such.

---

## 9. Customer segments

| | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---|---|---|---|---|
| Customers | 511 (23%) | 613 (28%) | 601 (27%) | 487 (22%) |
| Average income | 58,502 | 39,541 | **75,537** | 31,627 |
| Average spending | 813 | 124 | **1,301** | 145 |
| Average age | 49 | 46 | 46 | **39** |
| Children | 1.3 | **1.5** | 0.2 | 0.8 |
| Deal purchases | **4.5** | 2.0 | 1.1 | 1.9 |
| Web visits / month | 6.1 | 5.9 | 2.8 | **6.9** |

**Cluster 2 — the premium segment (27%).** The highest income and by far the
highest spending, almost no children, barely uses discounts and rarely visits the
website. They buy without needing to be persuaded.

**Cluster 0 — deal-driven high spenders (23%).** Good income and solid spending,
families with young children, and the heaviest users of discounted offers of all
four groups. They spend well, but they respond to promotions.

**Cluster 1 — budget families (28%).** The largest group. Low income, the most
children, and the lowest spending of all: a tenth of what Cluster 2 spends.

**Cluster 3 — young browsers (22%).** The youngest and lowest-income segment,
with the highest web activity but very low spending. They visit often and
convert rarely.

Wine is the leading category in every segment, but the amounts differ by an
order of magnitude between Cluster 2 and Cluster 1.

---

## 10. Conclusion

The original question was *who buys what*. The answer is that this customer base
splits along two axes — **purchasing power** and **family situation** — into four
usable segments.

Rather than promoting a new product to all 2212 customers, the shop can target
the segment whose profile matches the offer:

- premium wine and meat campaigns to **Cluster 2**, which already spends heavily
  and does not need discounts;
- discount bundles and promotional offers to **Cluster 0**, which demonstrably
  responds to them;
- family-sized value products to **Cluster 1**, the largest but most
  price-constrained group;
- web retargeting and first-purchase incentives to **Cluster 3**, which already
  visits the site frequently but does not convert.

This is exactly the reasoning set out in the introduction of the project brief:
identify the segment most likely to buy, and market to that segment only.

On the technical side, the from-scratch K-means reproduces Scikit-Learn's
results (ARI 0.94 on Iris) and outperformed both CAH and DBSCAN on this dataset
in terms of silhouette and interpretability.

---

## 11. Limitations

- **k is a human choice.** The silhouette favoured k = 2; k = 4 was selected for
  interpretability. Defensible, but a judgement call rather than an objective
  truth.
- **Internal metrics only.** Silhouette and Davies-Bouldin measure geometry.
  Confirming these segments would require external validation — for instance,
  running a campaign and measuring the response rate per segment.
- **Simple encoding.** `Education_Level` and `Living_With` were label-encoded,
  which imposes an arbitrary order on categories.
- **A single point in time.** The dataset is a snapshot. It says nothing about
  seasonality, nor about customers moving from one segment to another over time.
- **Small sample.** 2212 customers is enough to find structure, but a larger base
  would give more stable segments.

---

## Author

**Rooney Nofal** — La Plateforme, Marseille.

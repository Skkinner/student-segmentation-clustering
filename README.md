# Student Segmentation: Comparing Clustering Algorithms — Campus Placement Dataset

Discovering meaningful student groups using **unsupervised machine learning**, with K-Means, Hierarchical Clustering, and DBSCAN.

This repository contains a mini project that applies and compares three clustering algorithms on the Campus Placement dataset. The goal is to identify student academic profiles **without using placement outcomes**, evaluate cluster quality, and determine which algorithm produces the most meaningful groups.

---

## 📌 Project Overview

| | |
|---|---|
| **Task** | Unsupervised learning: Student segmentation |
| **Dataset** | [Factors Affecting Campus Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement) (Kaggle) |
| **Algorithms** | K-Means, Hierarchical (Agglomerative), DBSCAN |
| **Evaluation** | Silhouette Score, cluster sizes, noise detection, cluster profiling |
| **Best Algorithm** | K-Means (k=2), Silhouette Score = **0.2015** |
| **Key idea** | A higher Silhouette Score does not always indicate better clustering. Cluster balance, noise coverage, and interpretability also matter. |

---

## 📊 Dataset

- **215** MBA students, **15** original columns.
- **12** student characteristics retained after removing:
  - `sl_no`: unique identifier with no meaningful clustering information.
  - `status`: placement outcome (Placed / Not Placed), excluded to maintain unsupervised learning.
  - `salary`: placement-related outcome with **31.16% missing values**.
- No missing values in the retained features.
- No duplicate full rows detected.
- **5 numerical** academic features and **7 categorical** background features.
- **14 features** after one-hot encoding.

**Numerical features:** `ssc_p`, `hsc_p`, `degree_p`, `etest_p`, `mba_p`

**Categorical features:** `gender`, `ssc_b`, `hsc_b`, `hsc_s`, `degree_t`, `workex`, `specialisation`

The objective is not to predict whether a student gets placed. Instead, the algorithms group students according to similarities in academic performance and educational background.

---

## ⚙️ Methodology

```text
Dataset
   ↓
Data Quality Checks
   ↓
Remove Identifier and Outcome Columns
   ↓
StandardScaler + One-Hot Encoding
   ↓
Feature Matrix (215 × 14)
   ↓
K-Means | Hierarchical | DBSCAN
   ↓
Parameter Tuning + Visualization
   ↓
Silhouette Score + Cluster Profiling
   ↓
Algorithm Comparison + Final Insights
```

### 1. Data Preparation

- Checked missing values and duplicate rows.
- Removed `sl_no`, `status`, and `salary`.
- Standardized the five numerical features using `StandardScaler`.
- One-hot encoded categorical features using `pd.get_dummies(drop_first=True)`.
- Used the same preprocessed feature matrix for all three algorithms.

### 2. K-Means Clustering

K-Means partitions students into groups by minimizing distances between observations and their cluster centroids.

**Configuration:**

| Parameter | Value |
|---|---|
| Algorithm | `KMeans` |
| n_clusters | 2 |
| random_state | 42 |
| K values evaluated | 2–10 |
| Evaluation | Silhouette Score |

**Silhouette analysis:**

| K | Silhouette Score |
|---|---|
| **2** | **0.2015** |
| 3 | 0.1462 |
| 4 | 0.1215 |
| 5 | 0.1203 |
| 6 | 0.1177 |
| 7 | 0.1062 |
| 8 | 0.1114 |
| 9 | 0.0915 |
| 10 | 0.1039 |

**Result:** `k=2` achieved the highest Silhouette Score among the tested values.

The algorithm identified two relatively balanced academic groups:

- **Cluster 0:** 98 students — generally higher academic performance.
- **Cluster 1:** 117 students — generally lower academic performance.

### 3. Hierarchical Clustering

Agglomerative Hierarchical Clustering starts with each student as an individual cluster and progressively merges similar groups.

Four linkage methods were evaluated with `n_clusters=2`.

| Linkage | Silhouette Score | Cluster Sizes |
|---|---|---|
| Ward | 0.1646 | 134 / 81 |
| Complete | 0.1695 | 171 / 44 |
| Average | **0.2800** | 214 / 1 |
| Single | **0.2800** | 214 / 1 |

**Selected method: Ward linkage**

Although Average and Single linkage achieved higher Silhouette Scores, both isolated only one student from the remaining 214.

Ward linkage was selected because it produced a more balanced and interpretable segmentation.

This demonstrates why **Silhouette Score should not be evaluated in isolation**.

### 4. DBSCAN

DBSCAN identifies clusters based on data density and labels observations outside sufficiently dense regions as noise (`-1`).

**Parameters evaluated:**

| Parameter | Values |
|---|---|
| `eps` | 0.5, 1, 1.5, 2, 2.5, 3, 4 |
| `min_samples` | 3, 5, 10 |
| Total combinations | 21 |

A configuration was accepted only if it:

1. Produced at least two clusters.
2. Classified fewer than 50% of students as noise.
3. Had a valid Silhouette Score.

**Exploratory results:**

| eps | min_samples | Clusters | Noise Points | Silhouette |
|---|---|---|---|---|
| 1.5 | 3 | 12 | 140 | 0.2165 |
| 1.5 | 5 | 3 | 182 | **0.2429** |

The highest-scoring exploratory configuration assigned **182 of 215 students (84.7%)** to noise.

Because neither configuration satisfied the noise-coverage requirement, **no final DBSCAN model was selected**.

The exploratory DBSCAN Silhouette Scores were calculated using non-noise observations only and are not directly comparable to the full-dataset scores.

---

## 📈 Results & Algorithm Comparison

| Metric | K-Means | Hierarchical (Ward) | DBSCAN |
|---|---|---|---|
| Number of clusters | **2** | 2 | No final model |
| Cluster sizes | 98 / 117 | 134 / 81 | Exploratory: 3 clusters, 33 points |
| Silhouette Score | **0.2015** | 0.1646 | 0.2429* |
| Noise points | 0 | 0 | Exploratory: 182 |
| Cluster separation | Weak | Weak | Limited coverage |
| Interpretability | High | High | Low for this dataset |
| Final selection | **Best overall** | Alternative | Not selected |

*DBSCAN's exploratory Silhouette Score is based on only 33 non-noise students.*

### Cluster Profiling

The average standardized academic scores reveal the characteristics of each cluster.

**K-Means**

| Feature | Cluster 0 (98) | Cluster 1 (117) |
|---|---|---|
| `ssc_p` | 0.73 | -0.61 |
| `hsc_p` | 0.57 | -0.48 |
| `degree_p` | 0.65 | -0.54 |
| `etest_p` | 0.48 | -0.40 |
| `mba_p` | 0.66 | -0.56 |

**Interpretation:**

- Cluster 0 consists of students with generally above-average academic scores.
- Cluster 1 consists of students with generally below-average academic scores.
- Academic performance is an important distinguishing characteristic of the discovered groups.

**Hierarchical (Ward)** produced a similar distinction, but with less balanced group sizes.

### Best-Performing Algorithm

**K-Means with k=2** was the strongest option among the evaluated methods because:

- It achieved the highest full-dataset Silhouette Score (**0.2015**) of the selected models.
- It produced relatively balanced clusters (**98 / 117**).
- Its cluster profiles were easy to interpret.
- It assigned every student to a group.

However, the Silhouette Score is relatively low, indicating **weak cluster separation and considerable overlap**.

K-Means is the best algorithm tested in this project, not necessarily the optimal clustering method for all possible configurations or feature representations.

---

## 🔍 Key Insights

1. **Two academic profiles emerged.** K-Means and Ward mainly separated students into higher-performing and lower-performing groups.

2. **K-Means performed best overall.** It achieved the strongest full-dataset Silhouette Score among the selected models while maintaining balanced groups.

3. **Cluster separation was weak.** Scores around 0.16–0.20 suggest overlapping rather than clearly distinct student populations.

4. **DBSCAN struggled with the tested parameters.** Small neighborhood radii produced substantial noise, while larger radii generally merged students into one cluster.

5. **Higher Silhouette Scores can be misleading.** Average and Single linkage isolated one student, while DBSCAN's exploratory scores excluded most observations.

6. **Preprocessing matters.** Standardization and categorical encoding enabled distance-based clustering across different student characteristics.

7. **Clustering is not classification.** The discovered clusters reflect similarity between students, not predictions of their placement status.

---

## ⚠️ Limitations

- The dataset contains only **215 students**, so findings may not generalize to other institutions.
- Numerical and one-hot-encoded categorical features use different scaling conventions, which may influence distances.
- Cluster separation is weak despite K-Means being the best evaluated option.
- The 2D visualizations display only two academic features, while clustering uses all 14 processed features.
- DBSCAN conclusions are limited to the tested parameter combinations.
- The identified groups are descriptive patterns, not evidence of causal relationships or placement outcomes.

**Practical application:** Student segmentation could support exploratory analysis of academic profiles and help universities investigate different educational support needs. These clusters should not be used alone for consequential student decisions.

---

## 📁 Repository Structure

```text
student-segmentation-clustering/
├── Clustering_Algorithms_Implementation_and_Comparison.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repository**

```bash
git clone https://github.com/Skkinner/student-segmentation-clustering.git
cd student-segmentation-clustering
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Launch Jupyter Notebook**

```bash
jupyter notebook Clustering_Algorithms_Implementation_and_Comparison.ipynb
```

**`requirements.txt`:**

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
kagglehub
tqdm
jupyter
```

The notebook automatically downloads the Campus Placement dataset using `kagglehub`.

Depending on the environment, Kaggle authentication may be required.

Alternatively, the notebook can be opened and executed in Google Colab.

---

## 👥 Team

| Name | GitHub |
|---|---|
| Fahad Al-Nahi | [@Skkinner](https://github.com/Skkinner) |
| Basma Almuntaser | [@Basmaalmuntaser](https://github.com/username) |
| Raghad Alharbi | [@Raghad5MH](https://github.com/Raghad5MH) |
| Reham | [@username](https://github.com/username) |

---
## 🙏 Acknowledgments

- **Dataset:** Ben Roshan D, [Factors Affecting Campus Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement), Kaggle.
- **Machine Learning Library:** [Scikit-learn Documentation](https://scikit-learn.org/stable/).
- **K-Means:** [Scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
- **Hierarchical Clustering:** [Scikit-learn AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html).
- **DBSCAN:** [Scikit-learn DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

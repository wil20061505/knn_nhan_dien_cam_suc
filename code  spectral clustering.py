# ============================================================
# SPECTRAL CLUSTERING - CASE STUDY
# Phân cụm khách hàng E-commerce
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.linalg import eigh


# ============================================================
# 1. TẠO DỮ LIỆU KHÁCH HÀNG
# ============================================================

np.random.seed(42)

n_customers = 5000

# 4 nhóm khách hàng
cluster_sizes = [1420, 1180, 1240, 1160]

data = []

# -----------------------------
# Cụm 1: Khách hàng VIP
# Chi tiêu cao - mua thường xuyên
# -----------------------------
for i in range(cluster_sizes[0]):
    row = [
        np.random.normal(25, 5),       # Số đơn hàng
        np.random.normal(15000000, 2500000),  # Tổng chi tiêu
        np.random.normal(600000, 80000),     # Giá trị trung bình/đơn
        np.random.normal(12, 2),        # Tần suất mua/tháng
        np.random.normal(15, 3),        # Số danh mục sản phẩm
        np.random.normal(85, 5),        # Tỷ lệ mua khi khuyến mãi
        np.random.normal(10, 2),        # Số lần tương tác website
        np.random.normal(8, 2),         # Số lần thêm giỏ hàng
        np.random.normal(2, 1),         # Số lần hủy đơn
        np.random.normal(5, 2)          # Ngày từ lần mua cuối
    ]
    data.append(row)


# -----------------------------
# Cụm 2: Khách hàng trung bình
# -----------------------------
for i in range(cluster_sizes[1]):
    row = [
        np.random.normal(15, 3),
        np.random.normal(7000000, 1500000),
        np.random.normal(450000, 60000),
        np.random.normal(7, 1.5),
        np.random.normal(10, 2),
        np.random.normal(60, 8),
        np.random.normal(7, 2),
        np.random.normal(5, 1.5),
        np.random.normal(3, 1),
        np.random.normal(15, 4)
    ]
    data.append(row)


# -----------------------------
# Cụm 3: Khách hàng săn khuyến mãi
# Chi tiêu thấp - tần suất cao
# -----------------------------
for i in range(cluster_sizes[2]):
    row = [
        np.random.normal(18, 4),
        np.random.normal(3500000, 900000),
        np.random.normal(200000, 40000),
        np.random.normal(10, 2),
        np.random.normal(12, 3),
        np.random.normal(90, 5),
        np.random.normal(12, 3),
        np.random.normal(9, 2),
        np.random.normal(5, 2),
        np.random.normal(10, 3)
    ]
    data.append(row)


# -----------------------------
# Cụm 4: Khách hàng ít mua
# Chi tiêu cao nhưng tần suất thấp
# -----------------------------
for i in range(cluster_sizes[3]):
    row = [
        np.random.normal(6, 2),
        np.random.normal(10000000, 2000000),
        np.random.normal(1000000, 150000),
        np.random.normal(2, 1),
        np.random.normal(5, 2),
        np.random.normal(30, 8),
        np.random.normal(3, 1),
        np.random.normal(2, 1),
        np.random.normal(1, 1),
        np.random.normal(45, 8)
    ]
    data.append(row)


# ============================================================
# 2. TẠO DATAFRAME
# ============================================================

columns = [
    "So_don_hang",
    "Tong_chi_tieu",
    "Gia_tri_trung_binh_don",
    "Tan_suat_mua_thang",
    "So_danh_muc_da_mua",
    "Ty_le_mua_khi_khuyen_mai",
    "Tuong_tac_website",
    "Them_vao_gio_hang",
    "Huy_don",
    "Ngay_tu_lan_mua_cuoi"
]

df = pd.DataFrame(data, columns=columns)

# Xử lý dữ liệu âm do random
for col in df.columns:
    df[col] = df[col].clip(lower=0)

print("Kích thước dữ liệu:", df.shape)
print("\n5 dòng đầu:")
print(df.head())


# ============================================================
# 3. CHUẨN HÓA DỮ LIỆU
# ============================================================

# Chuẩn hóa Z-score bằng NumPy (không dùng scikit-learn)
X = df.to_numpy(dtype=float)
X = (X - X.mean(axis=0)) / X.std(axis=0)

print("\nĐã chuẩn hóa dữ liệu.")


# ============================================================
# 4. XÂY DỰNG SIMILARITY MATRIX
#    Gaussian Kernel + KNN
# ============================================================

print("\nĐang xây dựng Similarity Matrix...")

n_neighbors = 10
sigma = 1.0

# Tìm hàng xóm gần nhất bằng SciPy cKDTree
tree = cKDTree(X)
distances, indices = tree.query(X, k=n_neighbors + 1)

# Ma trận W
W = np.zeros((n_customers, n_customers))

for i in range(n_customers):

    for j in range(1, n_neighbors + 1):

        neighbor_index = indices[i, j]
        distance = distances[i, j]

        similarity = np.exp(
            -(distance ** 2) / (2 * sigma ** 2)
        )

        W[i, neighbor_index] = similarity

# Làm đối xứng
W = np.maximum(W, W.T)

print("Similarity Matrix:", W.shape)


# ============================================================
# 5. SPECTRAL CLUSTERING
# ============================================================

print("\nĐang chạy Spectral Clustering...")

k = 4

# ------------------------------------------------------------
# Spectral Clustering tự cài đặt bằng NumPy + SciPy
# ------------------------------------------------------------
def kmeans_numpy(X, n_clusters, random_state=42, n_init=10, max_iter=300):
    rng_master = np.random.default_rng(random_state)
    best_labels = None
    best_inertia = np.inf

    for _ in range(n_init):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        centers = X[rng.choice(len(X), n_clusters, replace=False)].copy()

        for _ in range(max_iter):
            distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()
            for c in range(n_clusters):
                members = X[labels == c]
                if len(members) > 0:
                    new_centers[c] = members.mean(axis=0)

            if np.allclose(centers, new_centers, atol=1e-7):
                centers = new_centers
                break
            centers = new_centers

        inertia = ((X - centers[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def spectral_clustering_numpy(W, n_clusters, random_state=42):
    # Degree matrix và normalized graph Laplacian
    degrees = W.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degrees)
    nonzero = degrees > 0
    inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

    L_sym = np.eye(len(W)) - (
        inv_sqrt_degree[:, None] * W * inv_sqrt_degree[None, :]
    )

    # Lấy k eigenvector ứng với k eigenvalue nhỏ nhất
    eigenvalues, eigenvectors = eigh(
        L_sym,
        subset_by_index=[0, n_clusters - 1]
    )

    # Chuẩn hóa từng hàng của embedding
    embedding = eigenvectors
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(row_norms, 1e-12)

    labels = kmeans_numpy(
        embedding,
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    return labels, embedding, eigenvalues


spectral_labels, spectral_embedding, spectral_eigenvalues = (
    spectral_clustering_numpy(W, k, random_state=42)
)

df["Spectral_Cluster"] = spectral_labels + 1


# ============================================================
# Các chỉ số đánh giá tự cài đặt (không dùng scikit-learn)
# ============================================================

def pairwise_euclidean(X):
    sq = np.sum(X * X, axis=1, keepdims=True)
    distances_sq = sq + sq.T - 2 * X @ X.T
    return np.sqrt(np.maximum(distances_sq, 0.0))


def silhouette_score_numpy(X, labels):
    D = pairwise_euclidean(X)
    scores = []

    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False

        if same.sum() == 0:
            scores.append(0.0)
            continue

        a = D[i, same].mean()
        b = np.inf

        for c in np.unique(labels):
            if c == labels[i]:
                continue
            other = labels == c
            if other.sum():
                b = min(b, D[i, other].mean())

        scores.append((b - a) / max(a, b, 1e-12))

    return float(np.mean(scores))


def calinski_harabasz_numpy(X, labels):
    overall = X.mean(axis=0)
    unique_labels = np.unique(labels)
    n = len(X)
    k = len(unique_labels)

    between = 0.0
    within = 0.0

    for c in unique_labels:
        cluster = X[labels == c]
        center = cluster.mean(axis=0)
        between += len(cluster) * np.sum((center - overall) ** 2)
        within += np.sum((cluster - center) ** 2)

    return float((between / (k - 1)) / max(within / (n - k), 1e-12))


def davies_bouldin_numpy(X, labels):
    unique_labels = np.unique(labels)
    centers = []
    scatter = []

    for c in unique_labels:
        cluster = X[labels == c]
        center = cluster.mean(axis=0)
        centers.append(center)
        scatter.append(np.mean(np.linalg.norm(cluster - center, axis=1)))

    centers = np.asarray(centers)
    scatter = np.asarray(scatter)

    center_dist = pairwise_euclidean(centers)
    ratios = np.zeros((len(unique_labels), len(unique_labels)))

    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            if i != j:
                ratios[i, j] = (scatter[i] + scatter[j]) / max(
                    center_dist[i, j], 1e-12
                )

    return float(np.mean(np.max(ratios, axis=1)))


# ============================================================
# 6. ĐÁNH GIÁ SPECTRAL CLUSTERING
# ============================================================

silhouette_spectral = silhouette_score_numpy(X, spectral_labels)

calinski_spectral = calinski_harabasz_numpy(X, spectral_labels)

davies_spectral = davies_bouldin_numpy(X, spectral_labels)

print("\n====================================")
print(" KẾT QUẢ SPECTRAL CLUSTERING")
print("====================================")

print("Silhouette Score:",
      round(silhouette_spectral, 3))

print("Calinski-Harabasz Index:",
      round(calinski_spectral, 3))

print("Davies-Bouldin Index:",
      round(davies_spectral, 3))


# ============================================================
# 7. KÍCH THƯỚC CÁC CỤM
# ============================================================

cluster_counts = df["Spectral_Cluster"].value_counts().sort_index()

print("\nKích thước các cụm:")

for cluster, count in cluster_counts.items():

    percentage = count / len(df) * 100

    print(
        f"Cụm {cluster}: "
        f"{count} khách hàng "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 8. SPECTRAL EMBEDDING
#    Giảm dữ liệu xuống 2 chiều để trực quan hóa
# ============================================================

print("\nĐang tạo Spectral Embedding...")

# Dùng 2 eigenvector đầu tiên của Spectral Clustering để trực quan hóa
X_embedding = spectral_embedding[:, :2]


# ============================================================
# 9. VẼ BIỂU ĐỒ PHÂN CỤM
# ============================================================

plt.figure(figsize=(10, 7))

for cluster in range(1, k + 1):

    mask = df["Spectral_Cluster"] == cluster

    plt.scatter(
        X_embedding[mask, 0],
        X_embedding[mask, 1],
        s=10,
        label=f"Cụm {cluster}"
    )

plt.title(
    "Spectral Clustering - Phân cụm khách hàng",
    fontsize=16
)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.legend()
plt.grid(alpha=0.2)

plt.show()


# ============================================================
# 10. BIỂU ĐỒ TRÒN
# ============================================================

plt.figure(figsize=(7, 7))

plt.pie(
    cluster_counts,
    labels=[
        f"Cụm {i}"
        for i in cluster_counts.index
    ],
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Phân bố kích thước các cụm khách hàng"
)

plt.show()


# ============================================================
# 11. K-MEANS ĐỂ SO SÁNH
# ============================================================

print("\nĐang chạy K-Means...")

kmeans_labels = kmeans_numpy(
    X,
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Đánh giá
silhouette_kmeans = silhouette_score_numpy(X, kmeans_labels)

calinski_kmeans = calinski_harabasz_numpy(X, kmeans_labels)

davies_kmeans = davies_bouldin_numpy(X, kmeans_labels)


# ============================================================
# 12. SO SÁNH HAI THUẬT TOÁN
# ============================================================

comparison = pd.DataFrame({

    "Phương pháp": [
        "Spectral Clustering",
        "K-Means"
    ],

    "Silhouette Score": [
        silhouette_spectral,
        silhouette_kmeans
    ],

    "Calinski-Harabasz": [
        calinski_spectral,
        calinski_kmeans
    ],

    "Davies-Bouldin": [
        davies_spectral,
        davies_kmeans
    ]

})

print("\n====================================")
print(" SO SÁNH SPECTRAL CLUSTERING VS K-MEANS")
print("====================================")

print(
    comparison.round(3).to_string(index=False)
)


# ============================================================
# 13. PROFILE CÁC CỤM
# ============================================================

profile = df.groupby(
    "Spectral_Cluster"
)[columns].mean()

print("\n====================================")
print(" PROFILE CÁC CỤM")
print("====================================")

print(profile.round(2))


# ============================================================
# 14. LƯU KẾT QUẢ
# ============================================================

df.to_csv(
    "customer_spectral_clustering.csv",
    index=False,
    encoding="utf-8-sig"
)

comparison.to_csv(
    "clustering_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

profile.to_csv(
    "cluster_profile.csv",
    encoding="utf-8-sig"
)

print("\nĐã lưu:")
print("- customer_spectral_clustering.csv")
print("- clustering_comparison.csv")
print("- cluster_profile.csv")


# ============================================================
# 15. HIỂN THỊ THÔNG TIN TỔNG KẾT
# ============================================================

print("\n")
print("==============================================")
print("        HOÀN THÀNH CASE STUDY")
print("==============================================")

print("Số khách hàng :", n_customers)
print("Số features   :", len(columns))
print("K              :", k)
print("KNN            :", n_neighbors)
print("Sigma          :", sigma)

print("\nSpectral Clustering:")
print("Silhouette     :", round(silhouette_spectral, 3))
print("Calinski       :", round(calinski_spectral, 3))
print("Davies-Bouldin :", round(davies_spectral, 3))

print("\nK-Means:")
print("Silhouette     :", round(silhouette_kmeans, 3))
print("Calinski       :", round(calinski_kmeans, 3))
print("Davies-Bouldin :", round(davies_kmeans, 3))

print("\n==============================================")

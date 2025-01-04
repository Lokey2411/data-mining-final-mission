import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import OPTICS
from scipy.spatial.distance import cdist

# Đường dẫn file CSV
FILE_PATH = "country_wise_latest.csv"

# Đọc dữ liệu từ file CSV
data = pd.read_csv(FILE_PATH)

# Lựa chọn các cột cần thiết
try:
    X = data[["Confirmed", "Deaths"]].values
except KeyError:
    print("Kiểm tra lại tên các cột trong file CSV!")
    exit()

# Áp dụng OPTICS
clust = OPTICS(min_samples=10, xi=0.01, min_cluster_size=0.05)
clust.fit(X)

# Lấy các điểm cụm và điểm nhiễu
labels = clust.labels_
unique_clusters = np.unique(labels[labels != -1])  # Các cụm hợp lệ (bỏ qua nhiễu)
noise_points = X[labels == -1]  # Các điểm nhiễu

# Tính tâm của từng cụm
cluster_totals = {}
for cluster_id in unique_clusters:
    cluster_points = X[labels == cluster_id]
    total_confirmed = cluster_points[:, 0].sum()  # Tổng số ca nhiễm
    cluster_totals[cluster_id] = total_confirmed

# Sắp xếp cụm theo mức độ nguy hiểm
sorted_clusters = sorted(cluster_totals.keys(), key=lambda c: cluster_totals[c])

# Gán màu sắc theo mức độ nguy hiểm
colors_map = {
    sorted_clusters[0]: "green",  # Cụm ít nguy hiểm nhất
    sorted_clusters[len(sorted_clusters) // 2]: "yellow",  # Cụm trung bình
    sorted_clusters[-1]: "red",  # Cụm nguy hiểm nhất
}
# Gán màu mặc định cho cụm còn lại
for cluster_id in sorted_clusters:
    if cluster_id not in colors_map:
        colors_map[cluster_id] = "orange"

# Gán các điểm nhiễu vào cụm gần nhất
if len(noise_points) > 0:
    distances = cdist(noise_points, [X[labels == cid].mean(axis=0) for cid in unique_clusters], metric="euclidean")
    nearest_cluster_ids = distances.argmin(axis=1)
    for i, point in enumerate(noise_points):
        closest_cluster = nearest_cluster_ids[i]
        labels[np.where((X == point).all(axis=1))[0]] = unique_clusters[closest_cluster]

# Vẽ biểu đồ clustering
plt.figure(figsize=(8, 6))

# Vẽ các cụm
LABELS = "Mức độ nguy hiểm số"
for cluster_id in unique_clusters:
    cluster_points = X[labels == cluster_id]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=f"{LABELS} {cluster_id}",
        color=colors_map[cluster_id],
        alpha=0.6,
    )

# Vẽ các điểm nhiễu
plt.scatter(
    X[labels == -1, 0],
    X[labels == -1, 1],
    c="black",
    marker="+",
    label="Nhiễu",
)

plt.xlabel("Confirmed")
plt.ylabel("Deaths")
plt.title("OPTICS Clustering (Color-coded by Risk)")
plt.legend()
plt.tight_layout()
plt.show()

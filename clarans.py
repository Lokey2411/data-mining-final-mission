import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Đọc dữ liệu từ file CSV
FILE_PATH = "country_wise_latest.csv"
data = pd.read_csv(FILE_PATH)

# Lựa chọn các cột cần thiết
try:
    X = data[["Confirmed", "Deaths"]].values  # Lấy 2 cột "Confirmed" và "Deaths"
except KeyError:
    print("Kiểm tra lại tên các cột trong file CSV!")
    exit()

# Tính ma trận khoảng cách Euclidean giữa các điểm
distance_matrix = cdist(X, X, metric="euclidean")

# Hàm tính tổng chi phí khoảng cách từ các điểm đến medoids gần nhất
def calculate_cost(distance_matrix, medoids):
    n = distance_matrix.shape[0]
    total_cost = 0
    for i in range(n):
        min_distance = min([distance_matrix[i][medoid] for medoid in medoids])
        total_cost += min_distance
    return total_cost

# Hàm gán các điểm vào cụm tương ứng với medoid gần nhất
def assign_clusters(distance_matrix, medoids):
    clusters = {medoid: [] for medoid in medoids}  # Khởi tạo cụm
    for i in range(distance_matrix.shape[0]):
        # Tìm medoid gần nhất cho điểm i
        closest_medoid = min(medoids, key=lambda medoid: distance_matrix[i][medoid])
        clusters[closest_medoid].append(i)
    return clusters

# Thuật toán CLARANS
def clarans(distance_matrix, k, num_local=5, max_neighbor=5):
    n = distance_matrix.shape[0]  # Số lượng điểm
    best_medoids = None
    best_cost = float('inf')

    for _ in range(num_local):
        # Chọn ngẫu nhiên k medoids ban đầu
        current_medoids = random.sample(range(n), k)
        current_cost = calculate_cost(distance_matrix, current_medoids)
        neighbor_counter = 0

        while neighbor_counter < max_neighbor:
            # Chọn một medoid ngẫu nhiên và thử thay thế bằng một điểm khác
            medoid_to_replace = random.choice(current_medoids)
            candidate = random.choice([i for i in range(n) if i not in current_medoids])

            # Tạo một bộ medoid mới
            new_medoids = current_medoids.copy()
            new_medoids.remove(medoid_to_replace)
            new_medoids.append(candidate)

            # Tính chi phí cho bộ medoid mới
            new_cost = calculate_cost(distance_matrix, new_medoids)

            if new_cost < current_cost:
                current_medoids = new_medoids
                current_cost = new_cost
                neighbor_counter = 0  # Reset bộ đếm hàng xóm
            else:
                neighbor_counter += 1

        # Cập nhật lời giải tốt nhất
        if current_cost < best_cost:
            best_medoids = current_medoids
            best_cost = current_cost

    # Gán các điểm vào cụm dựa trên medoids tốt nhất
    clusters = assign_clusters(distance_matrix, best_medoids)
    return best_medoids, best_cost, clusters

# Thực thi thuật toán và vẽ biểu đồ
if __name__ == "__main__":
    k = 3  # Số lượng medoids cần chọn
    best_medoids, best_cost, clusters = clarans(distance_matrix, k)

    print("Medoids được chọn (vị trí trong dữ liệu):", best_medoids)
    print("Tổng chi phí khoảng cách nhỏ nhất:", best_cost)
    print("Phân cụm:")
    for medoid, points in clusters.items():
        print(f"  Medoid {medoid}: {points}")

    # Vẽ biểu đồ phân cụm
    plt.figure(figsize=(12, 8))
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']  # Danh sách màu cho từng cụm
    markers = ['o', 's', 'D', '^', 'P', '*']      # Danh sách ký hiệu điểm

    # Vẽ từng cụm
    for idx, (medoid, points) in enumerate(clusters.items()):
        cluster_points = X[points]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[idx], marker=markers[idx], label=f"Cụm {idx + 1}")
        plt.scatter(X[medoid, 0], X[medoid, 1], c='k', marker='*', s=250, label=f"Medoid {idx + 1} (trung tâm)")

    plt.title("Phân cụm CLARANS", fontsize=16)
    plt.xlabel("Confirmed", fontsize=14)
    plt.ylabel("Deaths", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

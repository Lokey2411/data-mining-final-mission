import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import OPTICS

# Giảm số lượng điểm được tạo ra
np.random.seed(0)
n_points_per_cluster = 100  # Giảm từ 250 xuống 100

C1 = [-5, -2] + 0.8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + 0.1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + 0.2 * np.random.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + 0.3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
X = np.vstack((C1, C2, C3, C4, C5, C6))

clust = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.05)

# Run the fit
clust.fit(X)

# In danh sách tọa độ các điểm trong từng cụm (OPTICS)
print("\nOPTICS Clusters:")
for cluster_id in np.unique(clust.labels_):
    if cluster_id != -1:  # Bỏ qua các điểm nhiễu
        points_in_cluster = X[clust.labels_ == cluster_id]
        print(f"Cluster {cluster_id}:")
        print(points_in_cluster)
    # else:
    #     noise_points = X[clust.labels_ == -1]
    #     print("Noise:")
    #     print(noise_points)

# OPTICS with smaller epsilon
clust_small_eps = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.05, max_eps=0.5)
clust_small_eps.fit(X)
labels_small_eps = clust_small_eps.labels_

# print("\nOPTICS Clusters (Epsilon=0.5):")
# for cluster_id in np.unique(labels_small_eps):
#     if cluster_id != -1:  # Bỏ qua các điểm nhiễu
#         points_in_cluster = X[labels_small_eps == cluster_id]
#         print(f"Cluster {cluster_id}:")
#         print(points_in_cluster)
#     else:
#         noise_points = X[labels_small_eps == -1]
#         print("Noise:")
#         print(noise_points)

# OPTICS with larger epsilon
clust_large_eps = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.05, max_eps=2.0)
clust_large_eps.fit(X)
labels_large_eps = clust_large_eps.labels_

# print("\nOPTICS Clusters (Epsilon=2.0):")
# for cluster_id in np.unique(labels_large_eps):
#     if cluster_id != -1:  # Bỏ qua các điểm nhiễu
#         points_in_cluster = X[labels_large_eps == cluster_id]
#         print(f"Cluster {cluster_id}:")
#         print(points_in_cluster)
#     else:
#         noise_points = X[labels_large_eps == -1]
#         print("Noise:")
#         print(noise_points)

# Plotting
space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(15, 10))
G = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(G[0, 0])
ax2 = plt.subplot(G[0, 1])
ax3 = plt.subplot(G[1, 0])
ax4 = plt.subplot(G[1, 1])

# Reachability plot
colors = ["g.", "r.", "b.", "y.", "c."]
for klass, color in enumerate(colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
ax1.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
ax1.set_ylabel("Reachability (epsilon distance)")
ax1.set_title("Reachability Plot (Default)")

# OPTICS overall clustering
colors = ["g.", "r.", "b.", "y.", "c."]
for klass, color in enumerate(colors):
    Xk = X[clust.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], "k+", alpha=0.1)
ax2.set_title("OPTICS Clustering (Overall)")

# OPTICS with smaller epsilon
colors = ["g.", "r.", "b.", "y.", "c."]
for klass, color in enumerate(colors):
    Xk = X[labels_small_eps == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax3.plot(X[labels_small_eps == -1, 0], X[labels_small_eps == -1, 1], "k+", alpha=0.1)
ax3.set_title("OPTICS Clustering\nEpsilon=0.5")

# OPTICS with larger epsilon
colors = ["g.", "m.", "y.", "c."]
for klass, color in enumerate(colors):
    Xk = X[labels_large_eps == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax4.plot(X[labels_large_eps == -1, 0], X[labels_large_eps == -1, 1], "k+", alpha=0.1)
ax4.set_title("OPTICS Clustering\nEpsilon=2.0")

plt.tight_layout()
plt.show()
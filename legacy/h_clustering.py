import pickle
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# 1. Load — CHANGED to empty room and video game
with open("doppler_output/emp/empty_room_stream_0.txt", "rb") as f:
    d1 = pickle.load(f)
with open("doppler_output/vid/video_game_stream_0.txt", "rb") as f:
    d2 = pickle.load(f)

# 2. Threshold — remove noise floor
thresh1 = np.percentile(d1, 80)
thresh2 = np.percentile(d2, 80)
d1_clean = np.where(d1 > thresh1, d1, 0)
d2_clean = np.where(d2 > thresh2, d2, 0)

# 3. Chunk time axis to capture temporal patterns
# CHANGED chunk size — empty room is 72 pps, video game is 153 pps
# Use the lower one as reference so chunks are comparable in real time
# 72 frames ≈ 1 second for empty room
chunk_size = 72
d1_chunks = d1_clean[:len(d1_clean)//chunk_size * chunk_size].reshape(-1, chunk_size * d1_clean.shape[1])

# 153 frames ≈ 1 second for video game
chunk_size_vid = 153
d2_chunks = d2_clean[:len(d2_clean)//chunk_size_vid * chunk_size_vid].reshape(-1, chunk_size_vid * d2_clean.shape[1])

# 4. Stack with labels
# NOTE: chunk vectors are different lengths (72*100 vs 153*100)
# So we need to use mean Doppler profile per chunk instead of raw flatten
d1_feat = np.mean(d1_clean[:len(d1_clean)//chunk_size * chunk_size].reshape(-1, chunk_size, d1_clean.shape[1]), axis=1)
d2_feat = np.mean(d2_clean[:len(d2_clean)//chunk_size_vid * chunk_size_vid].reshape(-1, chunk_size_vid, d2_clean.shape[1]), axis=1)

X = np.vstack([d1_feat, d2_feat])
labels_true = np.array([0]*d1_feat.shape[0] + [1]*d2_feat.shape[0])

# 5. Run hierarchical clustering
Z = linkage(X, method='ward')

# 6. Plot dendrogram
plt.figure(figsize=(12, 5))
dendrogram(Z, no_labels=True, color_threshold=0)
plt.title("Hierarchical Clustering — Empty Room vs Video Game")
plt.xlabel("Chunks (1-second windows)")
plt.ylabel("Distance")
plt.show()

# 7. Cut into 2 clusters and check separation
cluster_labels = fcluster(Z, t=2, criterion='maxclust')
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(labels_true, cluster_labels)
print(f"Adjusted Rand Index: {ari:.3f}")

# 8. Confusion breakdown
w1_in_c1 = np.sum(cluster_labels[:d1_feat.shape[0]] == 1)
w1_in_c2 = np.sum(cluster_labels[:d1_feat.shape[0]] == 2)
w2_in_c1 = np.sum(cluster_labels[d1_feat.shape[0]:] == 1)
w2_in_c2 = np.sum(cluster_labels[d1_feat.shape[0]:] == 2)
print(f"Empty Room  → Cluster 1: {w1_in_c1}, Cluster 2: {w1_in_c2}")
print(f"Video Game  → Cluster 1: {w2_in_c1}, Cluster 2: {w2_in_c2}")
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
from collections import Counter

# Load data
with open("hand_gesture_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

X = np.array(dataset["data"])
y = np.array(dataset["labels"])

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# KNN logic
class ImprovedKNNClassifier:
    def __init__(self, k=5, weighted=True):
        self.k = k
        self.weighted = weighted

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _predict_single(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = distances.argsort()[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        if self.weighted:
            weights = 1 / (distances[k_indices] + 1e-5)
            vote_counts = {}
            for label, weight in zip(k_nearest_labels, weights):
                vote_counts[label] = vote_counts.get(label, 0) + weight
        else:
            vote_counts = Counter(k_nearest_labels)

        return max(vote_counts.items(), key=lambda item: item[1])[0]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

# Train KNN in 2D space
knn = ImprovedKNNClassifier(k=5, weighted=True)
knn.fit(X_pca, y_encoded)

# Create meshgrid
h = 0.01
x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict on mesh
Z = knn.predict(grid_points)
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 6))
cmap_light = ListedColormap(plt.cm.tab10.colors[:len(np.unique(y_encoded))])
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)

scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap=ListedColormap(plt.cm.tab10.colors), edgecolor='k', s=50)
handles, _ = scatter.legend_elements()
plt.legend(handles=list(handles), labels=list(le.classes_), loc='upper right', title='Gestures')
plt.title("KNN Decision Boundary (2D PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

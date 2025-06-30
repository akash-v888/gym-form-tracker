import numpy as np
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class ImprovedKNNClassifier:
    def __init__(self, k=5, weighted=True):
        self.k = k
        self.weighted = weighted

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _predict_single(self, x, return_top_k=False):
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

        if return_top_k:
            total = sum(vote_counts.values())
            top_k = Counter(vote_counts).most_common(3)
            return [(label, count / total) for label, count in top_k]
        else:
            return max(vote_counts.items(), key=lambda item: item[1])[0]

    def predict(self, X, return_top_k=False):
        X = np.array(X)
        return [self._predict_single(x, return_top_k=return_top_k) for x in X]

# Load dataset
with open("hand_gesture_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

X = dataset["data"]
y = dataset["labels"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train KNN
knn = ImprovedKNNClassifier(k=5, weighted=True)
knn.fit(X_train, y_train)

# Evaluate
train_acc = np.mean(np.array(knn.predict(X_train)) == y_train)
test_acc = np.mean(np.array(knn.predict(X_test)) == y_test)
print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

# Save model and label encoder
with open("gesture_knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("gesture_knn_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

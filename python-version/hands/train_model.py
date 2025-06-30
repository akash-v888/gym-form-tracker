import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from neural_net_algo import SimpleNeuralNet

# Load dataset
with open("hand_gesture_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

X = np.array(dataset["data"])
y = np.array(dataset["labels"])

# Encode labels (e.g., 'fist_L' → 0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize X (scale between 0 and 1)
X = X / np.max(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Model setup
input_size = X.shape[1]
hidden_size = 64  # increased from 32
output_size = len(np.unique(y_encoded))

model = SimpleNeuralNet(input_size, hidden_size, output_size, lr=0.05)

# Train model
model.train(X_train, y_train, epochs=2000)

# Evaluate
train_acc = model.evaluate(X_train, y_train)
test_acc = model.evaluate(X_test, y_test)
print(f"\nFinal Train Accuracy: {train_acc:.2f}")
print(f"Final Test Accuracy: {test_acc:.2f}")

# Save model and encoder
model.save_model("gesture_model.pkl")
with open("gesture_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model and label encoder saved.")

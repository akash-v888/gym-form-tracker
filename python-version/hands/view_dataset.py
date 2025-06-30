import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import Counter

# MediaPipe-style connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Palm edge
]

# Load dataset
with open("hand_gesture_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = dataset["data"]
labels = dataset["labels"]

# Extract gesture type and handedness
gesture_types = []
hand_sides = []
for label in labels:
    if '_' in label:
        gesture, side = label.rsplit('_', 1)
        gesture_types.append(gesture)
        hand_sides.append(side)
    else:
        gesture_types.append(label)
        hand_sides.append("Unknown")

# Count occurrences of full labels
label_counts = Counter(labels)

# Show label distribution
plt.figure(figsize=(10, 6))
plt.bar(label_counts.keys(), label_counts.values(), color="teal")
plt.title("Gesture Sample Counts (with Handedness)")
plt.xlabel("Gesture_Label")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# Display one example per unique label
label_to_sample = {}
for idx, label in enumerate(labels):
    if label not in label_to_sample:
        label_to_sample[label] = data[idx]

for label, sample in label_to_sample.items():
    xs = sample[0::2]
    ys = sample[1::2]
    ys = [-y for y in ys]  # Flip Y for upright

    plt.figure()
    plt.title(f"Example: {label}")

    # Plot landmarks
    plt.scatter(xs, ys, color="blue")

    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.text(x, y, str(i), fontsize=8, color="gray")

    # Draw connections
    for a, b in HAND_CONNECTIONS:
        x1, y1 = xs[a], ys[a]
        x2, y2 = xs[b], ys[b]
        plt.plot([x1, x2], [y1, y2], color="orange", linewidth=2)

    plt.axis("equal")
    plt.grid(True)

plt.show()

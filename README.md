# Hand Gesture Recognition System

This project implements a real-time hand gesture recognition system using MediaPipe and custom-trained models in Python. It supports both neural network and k-nearest neighbors (KNN) classification and is designed for extensibility.

## Background

The original goal of this project was to build an exercise form tracker. I began by developing a prototype in Python using computer vision, and later reimplemented it in C++ for performance and deeper control. However, for the purpose of creating a more approachable demo, I pivoted to building a hand gesture recognizer in Python — something simpler, but still rich in application potential.

This version focuses on recognizing static hand gestures like thumbs up/down, peace sign, fist, "OK", and open hand. It supports both left and right hand variants and allows easy expansion with new gesture types.

## Features

- Real-time hand tracking using MediaPipe
- Live gesture classification with either:
  - A lightweight feedforward neural network (from scratch)
  - An improved KNN classifier with distance weighting
- Custom data collection and annotation tool
- Dataset visualizer with label distribution and hand landmark plots
- Top-3 prediction support with confidence scores
- Undo functionality during data collection
- Appends new data to existing dataset for incremental learning
- Handedness detection (left vs. right hand)

## File Structure

```
project-root/
├── cpp-version/
│   ├── models/                        # YOLOv8 pose ONNX model
│   ├── src/
│   │   └── main.cpp                   # Main C++ file for full-body + hand keypoint visualization
│   └── CMakeLists.txt                 # Build instructions for the C++ version
│
├── python-version/
│   └── hands/
│       ├── collect_data.py                 # Interactive tool for labeling and saving hand gestures
│       ├── view_dataset.py                 # Visualizes dataset samples and class distribution
│       ├── train_model.py                  # Trains neural network from scratch
│       ├── train_model_knn.py              # Trains improved KNN classifier
│       ├── gesture_recognizer.py           # Live inference using neural network
│       ├── gesture_recognizer_knn.py       # Live inference using KNN
│       ├── neural_net_algo.py              # Custom neural network class
│       ├── hand_tracker.py                 # Utility for MediaPipe hand tracking
│       ├── hand_kp_stream.py               # Streams hand keypoints via socket to C++ listener
│       ├── hand_gesture_dataset.pkl        # Serialized landmark/label data
│       ├── gesture_model.pkl               # Trained neural network model
│       ├── gesture_knn_model.pkl           # Trained KNN model
│       ├── gesture_label_encoder.pkl       # Label encoder for neural net
│       ├── gesture_knn_label_encoder.pkl   # Label encoder for KNN
│       └── __pycache__/                    # Cached bytecode
│
├── .gitignore                          # Ignores virtualenvs, build files, models, etc.
└── README.md                           # Project overview and documentation
```

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- matplotlib (for dataset visualization)

You can install dependencies using:

```bash
pip install -r requirements.txt
```

## Key Files

### `main.cpp`
Located in `cpp-version/src`, this is the C++ entry point for the pose tracking demo. It loads a YOLOv8 pose estimation ONNX model for full-body detection, receives hand keypoints over a socket from the Python script `hand_kp_stream.py`, and overlays both sets of landmarks on the webcam feed. It uses OpenCV’s DNN module for inference and runs a lightweight keypoint visualizer.

### `collect_data.py`
This script allows users to capture and label hand gesture keypoints in real-time using MediaPipe. It saves the collected (x, y) landmark coordinates along with user-defined labels into a `.pkl` dataset file (`hand_gesture_dataset.pkl`) for model training. A toggle allows appending to or overwriting the dataset.

### `train_model.py`
Trains a simple neural network from scratch on the collected dataset using NumPy. The model architecture includes a single hidden layer and a softmax output layer. Training progress is printed per epoch, and both the model and label encoder are saved to disk (`gesture_model.pkl`, `gesture_label_encoder.pkl`) for later inference.

### `neural_net_algo.py`
Defines the `SimpleNeuralNet` class used in `train_model.py` and `gesture_recognizer.py`. This minimal implementation includes forward propagation, backpropagation, and gradient descent logic using raw NumPy. It also includes utilities to save and load the model's weights for reuse.

### `train_model_knn.py`
Implements an improved K-Nearest Neighbors classifier with optional distance weighting. It loads and encodes the gesture dataset, splits it into train/test sets, evaluates classification accuracy, and saves the trained model and label encoder. This version supports smoother prediction by giving closer neighbors more weight.

## Getting Started

### 1. Collect Data

Run:

```bash
python collect_data.py
```

Use the keyboard to label each gesture as it's detected (it may be easier to hold down a key to collect more smaples). Press **Backspace** to undo a sample. At the end, a `.pkl` dataset is saved.

### 2. Visualize Dataset (Optional)

```bash
python view_dataset.py
```

Shows a breakdown of gesture types and plots landmark points per class.

### 3. Train a Model

You can choose between:

- Neural Network:

  ```bash
  python train_model.py
  ```

- KNN Classifier:

  ```bash
  python train_model_knn.py
  ```

Both will save their respective models and encoders for future use.

### 4. Recognize Gestures in Real Time

```bash
# Neural Net version
python gesture_recognizer.py

# KNN version
python gesture_recognizer_knn.py
```

## Notes

- The system can detect handedness, which helps disambiguate similar gestures from left vs. right hand.
- If a prediction is too uncertain (low confidence), the system will hold the last valid result.
- Data quality matters: make sure to collect varied samples for each gesture (different angles, hand orientation, lighting).

## Future Plans

- Improve accuracy for edge-case gestures (e.g., tilted or partially occluded hands)
- Support for dynamic gestures or motion-based classification
- Expand symbol set to full ASL alphabet
- Integrate with the original form tracker project for workout analysis

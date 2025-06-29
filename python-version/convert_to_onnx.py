from ultralytics import YOLO

# Load PyTorch model
model = YOLO("yolov8n-pose.pt")

# Export to ONNX
model.export(format="onnx", opset=12)

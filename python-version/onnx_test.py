import onnx

model = onnx.load("yolov8n-pose.onnx")
print("== Output Info ==")
for output in model.graph.output:
    print(f"Name: {output.name}")
    print(f"Type: {output.type}")
    shape = output.type.tensor_type.shape
    dims = [dim.dim_value for dim in shape.dim]
    print(f"Shape: {dims}")
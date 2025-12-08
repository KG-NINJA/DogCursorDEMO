from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model

# Export the model
# format='onnx': exports to ONNX
# opset=12: widely supported opset version for web
# int8=True: (Optional) for quantization, but let's stick to FP32 first for stability unless specified
success = model.export(format='onnx', opset=12)

print(f"Export completed: {success}")

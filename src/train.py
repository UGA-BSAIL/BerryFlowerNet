from ultralytics import YOLO
import ultralytics
print(ultralytics.__file__)

# build from YAML and transfer weights
model = YOLO("CFNet.yaml").load("yolo11s.pt")

# Train the model
results = model.train(data="data.yaml", epochs=200,
                      imgsz=1600, batch=4, patience=200, device=0, seed=0)

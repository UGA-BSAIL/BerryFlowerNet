from ultralytics import YOLO

# Load a model
model = YOLO("./best.pt")

# Validate the model
metrics = model.val(data='./data.yaml', imgsz=1600, batch=1,
                    conf=0.25, iou=0.6, device=0, split='test', max_det=600)

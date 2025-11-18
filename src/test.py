from ultralytics import YOLO
import torch

# Load a model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("./best_weights.pt", map_location=device)
model = YOLO("./CFNet.yaml")
model.model.load_state_dict(ckpt["model"])
model.model.eval()

# Validate the model
metrics = model.val(data='./data.yaml', imgsz=1600, batch=1,
                    conf=0.25, iou=0.6, device=0, split='test', max_det=600)

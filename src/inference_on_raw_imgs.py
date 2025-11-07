import os
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
import cv2
from sahi import AutoDetectionModel
import numpy as np
import csv
dates = ['20240205', '20240222', '20240307']
rows = ['cam2-row2', 'cam2-row4', 'cam2-row16', 'cam2-row17']
img_base_dir = './Raw_image_for_analysis/'

save_base_dir = './ultralytics/predictions'
# os.makedirs(save_dir, exist_ok=True)

model_path = './best.pt'

colors = {
    'bud': (255, 0, 0),
    'extendedbud': (0, 255, 0),
    'fullbloom': (0, 0, 255),
    'greenfruit': (255, 255, 0),
    'petalfall': (255, 0, 255),
}

confs = [0.30]
slice_size = 1600
for confidence in confs:
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=model_path,
        confidence_threshold=confidence,
        device="cuda:0",
    )

    # Prepare to collect all results
    all_counts = []

    for date in dates:
        for row_name in rows:
            imgpath = f"{img_base_dir}/{date}/{row_name}/"
            save_dir = f"{save_base_dir}_{slice_size}_{confidence}/{date}/{row_name}/"
            os.makedirs(save_dir, exist_ok=True)
            imgs = os.listdir(imgpath)
            imgs = sorted(imgs)

            for img in imgs:
                img_dir = os.path.join(imgpath, img)
                image = cv2.imread(img_dir)
                img_height, img_width, _ = image.shape
                result = get_sliced_prediction(
                    image,
                    detection_model,
                    slice_height=slice_size,
                    slice_width=slice_size,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    postprocess_class_agnostic=True
                )
                object_prediction_list = result.object_prediction_list

                # Count classes
                class_counts = {}
                for prediction in object_prediction_list:
                    name = prediction.category.name
                    class_counts[name] = class_counts.get(name, 0) + 1

                # Prepare row for CSV
                row_dict = {
                    'date': date,
                    'row': row_name,
                    'image': img
                }
                row_dict.update(class_counts)
                all_counts.append(row_dict)

                # Draw boxes
                for prediction in object_prediction_list:
                    bbox = prediction.bbox.to_xyxy()
                    name = prediction.category.name
                    color = colors[name]
                    x1, y1, x2, y2 = int(bbox[0]), int(
                        bbox[1]), int(bbox[2]), int(bbox[3])
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 10)

                save_path = os.path.join(save_dir, img)
                cv2.imwrite(save_path, image)

    # Save all counts to a single CSV
    csv_path = os.path.join(save_base_dir + '_' + str(slice_size) + '_' + str(confidence),
                            "class_counts_per_image" + "_" + str(slice_size) + '_' + str(confidence) + ".csv")
    # Get all class names for header
    all_class_names = set()
    for row in all_counts:
        all_class_names.update(row.keys())
    all_class_names.discard('date')
    all_class_names.discard('row')
    all_class_names.discard('image')
    header = ['date', 'row', 'image'] + sorted(all_class_names)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for row in all_counts:
            for cls in all_class_names:
                if cls not in row:
                    row[cls] = 0
            writer.writerow(row)

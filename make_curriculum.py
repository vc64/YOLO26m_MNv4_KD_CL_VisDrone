import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from torchvision.ops import box_iou
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

def load_teacher():
    model_path = hf_hub_download(
        repo_id="kailunw/visdrone-yolo26m",
        filename="best.pt"
    )

    model = YOLO(model_path)
    return model

def xywhn2xyxy(x, w, h):
    # convert normalized yolo xywh to absolute xyxy
    y = np.copy(x)
    # top left x,y
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2)

    # bottom right x,y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2)
    return y

def calculate_dataset_difficulty(data_yaml, iou_threshold=0.5):
    # simple curriculum based on teacher's avg confidence on true labels in training set
    model = load_teacher()

    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    dataset_path = "datasets" / Path(data_cfg.get('path', ''))
    images_dir = dataset_path / data_cfg['train']
    labels_dir = Path(str(images_dir).replace('images', 'labels'))

    image_difficulties = []

    for img_path in images_dir.glob("*.jpg"):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]

        with open(label_path, 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]
            if not lines:
                continue

            gt_cls = torch.tensor([float(x[0]) for x in lines])
            gt_xywhn = np.array([[float(x) for x in line[1:]] for line in lines])
            gt_boxes = torch.tensor(xywhn2xyxy(gt_xywhn, w, h))

        results = model(img, verbose=False)[0]
        pred_boxes = results.boxes.xyxy.cpu()
        pred_cls = results.boxes.cls.cpu()
        pred_conf = results.boxes.conf.cpu()

        image_confidences = []

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            # calculate IoU between all true boxes and all pred boxes
            ious = box_iou(gt_boxes, pred_boxes)

            for i, gt_class in enumerate(gt_cls):
                # find pred with the highest overlap for this true box
                best_iou, best_pred_idx = ious[i].max(0)

                if best_iou > iou_threshold and pred_cls[best_pred_idx] == gt_class:
                    image_confidences.append(pred_conf[best_pred_idx].item())
                else:
                    image_confidences.append(0.0)
        else:
            # if the model predicted nothing, confidence is 0 on all true boxes
            image_confidences = [0.0] * len(gt_boxes)

        # difficulty is just avg confidence
        avg_confidence = np.mean(image_confidences)
        image_difficulties.append((str(img_path), avg_confidence))

    return image_difficulties, dataset_path, data_cfg

def generate_splits(difficulties, dataset_path, base_cfg):
    difficulties.sort(key=lambda x: x[1], reverse=True)
    total_imgs = len(difficulties)

    print(f"\nTotal Analyzed Images: {total_imgs}")

    # split into thirds, so easiest 33% and easiest 66% are the thresholds
    stage1_cutoff = int(total_imgs * 0.33)
    stage2_cutoff = int(total_imgs * 0.66)

    splits = {
        "stage1_easy": [x[0] for x in difficulties[:stage1_cutoff]],
        "stage2_med":  [x[0] for x in difficulties[:stage2_cutoff]],
        "stage3_full": [x[0] for x in difficulties]
    }

    # generate the yaml files
    for stage_name, image_list in splits.items():
        txt_filename = f"{stage_name}.txt"
        with open(txt_filename, 'w') as f:
            f.write('\n'.join(image_list))

        new_yaml_cfg = {
            'path': str(dataset_path.resolve()),
            'train': str(Path(txt_filename).resolve()),
            'val': base_cfg['val'],
            'names': base_cfg['names']
        }

        yaml_filename = f"visdrone_{stage_name}.yaml"
        with open(yaml_filename, 'w') as f:
            yaml.dump(new_yaml_cfg, f, sort_keys=False)

        if stage_name != "stage3_full":
            worst_conf_in_split = difficulties[len(image_list)-1][1]
            print(f"Generated {yaml_filename} -> {len(image_list)} images (Min Conf: {worst_conf_in_split:.3f})")
        else:
            print(f"Generated {yaml_filename} -> {len(image_list)} images (Full Dataset)")

if __name__ == "__main__":
    difficulties, dataset_path, data_cfg = calculate_dataset_difficulty(data_yaml="ultralytics/ultralytics/cfg/datasets/VisDrone.yaml")
    generate_splits(difficulties, dataset_path, data_cfg)
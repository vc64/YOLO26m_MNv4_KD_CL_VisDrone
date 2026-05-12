# Knowledge Distillation and Curriculum Learning for Object Detection on Drone-based Images

Final project for COMPSCI 682: Neural Networks at UMass Amherst for Spring 2026.

Explored use of curriculum learning and knowledge distillation for object detection.

## Overview

### Dataset
- Used the VisDrone dataset: https://docs.ultralytics.com/datasets/detect/visdrone
- Features images captured from drones in rural, suburban, and urban areas of China, with bounding boxes and labels for objects
- 10 possible labels: pedestrian, people, bicycle, car, van, truck, tricycle, awning tricycle, bus, and motor.

### Knowledge Distillation
- Used pre-trained YOLO26m from https://huggingface.co/kailunw/visdrone-yolo26m as teacher model.
- Used YOLO26m with backbone swapped out for MobileNetV4 as student model.
- Performed knowledge distillation using both the outputs of the teacher model (i.e. logits for classification) and the features from the teacher's backbone.
- Feature distillation done using Masked Generative Distillation: https://arxiv.org/abs/2205.01529.

### Curriculum Learning
- Categorized training data by easy, medium, and hard based on the average confidence of the teacher model on the correct label across all objects in an image.
- Aimed for roughly equal-sized partitions of the data per difficulty level.
- Trained the model on progressively more data, starting with just the easy data for 20 epochs, then the easy + medium data for 50 epochs, and lastly the entire training data for 130 epochs (with early stopping).

### Training
- Trained four models:
  - **Student baseline**: student model architecture trained directly on the labeled dataset
  - **Knowledge distillation**: student model with just knowledge distillation (at output and feature level) using teacher model described above
  - **Curriculum learning**: student model architecture trained on the labeled dataset but in staged progression as described above
  - **Knowledge distillation + curriculum learning**: combined both knowledge distillation and curriculum learning
- All training was done with early stopping (patience = 30)

### Evaluation
- We evaluate all four models on the predefined validation and test sets from VisDrone
- We measure mean Average Precision (mAP50 and mAP50-95) across all validation and test cases

## Resources
- ultralytics fork (with UIB implementation added): https://github.com/vc64/ultralytics
- Model files: https://huggingface.co/vc64/YOLO26m_MNv4_KD_CL_VisDrone

# NYCU Computer Vision 2026 HW2
- Student ID: 111550148
- Name: 陳冠達

## Introduction
This task aims to perform digit detection, 
where the model is required to localize and classify digits in images. 
My core idea is to adopt a Deformable DETR framework with a pretrained ResNet-50 backbone 
to leverage both strong feature extraction and efficient transformer-based object detection.

## Environment Setup
Required libraries can be installed using:
```base
!pip install -r requirements.txt
```

## Usage
### Training
How to train your model.
```bash
python hw2.py
```

### Inference
Run inference using a trained model checkpoint:
```bash
python hw2.py --predict_only path/to/model.pt
```

## Performance Snapshot
<img width="1441" height="61" alt="image" src="https://github.com/user-attachments/assets/b4f221ab-6ee3-4790-8eb2-a0ad03969dc5" />





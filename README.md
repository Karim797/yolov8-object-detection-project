# YOLOv8 Object Detection Project

## Project Overview
This project was developed as part of the Epsilon AI Computer Vision track.

It implements an end-to-end object detection system using YOLOv8 to detect objects in real-world images. Multiple training strategies were explored, including training from scratch, transfer learning, and fine-tuning, to determine the best-performing model.

The final model was deployed using Streamlit for interactive object detection.

---

## Objectives
- Build a complete end-to-end computer vision project
- Apply object detection using YOLOv8
- Compare multiple training strategies
- Deploy the best-performing model
- Deliver a production-style AI project

---

## Project Context
This project was completed as part of the **Epsilon AI training program**, following real-world AI project delivery requirements including:
- Model development
- Training experimentation
- Performance comparison
- Deployment
- Documentation
- Presentation

---

## Dataset
This project uses a hybrid dataset consisting of:

- COCO 2017 dataset
- Manually annotated images using Roboflow

Dataset Link:
https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset

Dataset characteristics:
- 80 object classes
- Real-world object detection scenarios
- Mixed annotation sources
- Class imbalance
- Small object detection challenges

---

## Exploratory Data Analysis
Key findings:
- Dataset contains significant class imbalance
- Object sizes vary considerably
- Small objects increase detection difficulty
- Mixed annotation sources may introduce label noise

These characteristics make transfer learning and fine-tuning highly suitable.

---

## Models Implemented

### 1. YOLOv8 Training from Scratch
A baseline model trained entirely from scratch.

### 2. YOLOv8 Transfer Learning
Pretrained YOLOv8 weights were used to accelerate learning.

### 3. YOLOv8 Fine-Tuning
The transfer learning model was further fine-tuned to improve performance.

---

## Final Model
Best trained model:

```bash
best_model.pt
```

---

## Tech Stack
- Python
- YOLOv8
- Ultralytics
- OpenCV
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Roboflow
- Google Colab

---

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Repository Structure
- app.py
- best_model.pt
- requirements.txt
- Object_Detection_Project.ipynb
- presentation.pptx
- screenshots/
- README.md

---

## Results
The fine-tuned YOLOv8 model delivered the best overall object detection performance.

Prediction examples are included in the screenshots folder.

---

## Demo Video
roject Demo Video
---

## Author
Karim Deheya

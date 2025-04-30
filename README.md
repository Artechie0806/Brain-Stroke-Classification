# 🧠 Brain Stroke Detection with Vision Transformer (ViT-B16)

This project demonstrates a deep learning application for **brain stroke classification** from CT scan images using a **Vision Transformer (ViT-B16)** architecture. A lightweight **Flask web app** allows users to interactively upload CT scans and receive predictions powered by the trained model.

Model Download link: https://drive.google.com/drive/folders/1dnhGtSpCbt76Pn22_BHjoU9L2P3rHTff?usp=drive_link

---

## 🔍 Overview

| Feature          | Description                                          |
|------------------|------------------------------------------------------|
| **Model**        | Vision Transformer (ViT-B16), fine-tuned             |
| **Task**         | Binary classification (Stroke vs. No Stroke)         |
| **Interface**    | Flask web app                                        |
| **Input**        | Brain CT scan images                                 |
| **Output**       | Stroke risk prediction (probability + label)         |

---

## 🖼️ How It Works
-Upload a brain CT image (.jpg / .png)

-The image is resized and normalized for ViT input

-The ViT-B16 model outputs a probability score

-The result is displayed as:

---Stroke Detected

---No Stroke Detected

---

## 📚 Dataset: Brain Stroke CT Dataset
-This project utilizes the Brain Stroke CT Dataset by Ozgur Aslank, hosted on Kaggle. The dataset comprises 2,393 labeled CT scan images categorized into two classes:

-Stroke: CT images indicating the presence of a stroke.

-No Stroke: CT images without signs of a stroke.

## 📁 Dataset Details:
-Total Images: 6850

-Image Format: PNG

-Resolution: 640x640 pixels


## 🧠 Model Architecture

- **Base Model**: Vision Transformer (ViT-B16) from `vit_keras`
- **Input Size**: 224x224 pixels
- **Custom Head**:
  ```python
  Flatten()
  Dense(256, activation='swish')
  Dense(1, activation='sigmoid')
- **Loss Function**: Binary Crossentropy
- **Pretrained**: On ImageNet, then fine-tuned on medical images

---

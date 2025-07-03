# Deep_Learning_on_Oxford-IIIT_Pet_Dataset

A study on how modern deep learning models perform on the Oxford-IIIT Pet Dataset for **image classification** and **semantic segmentation** tasks.

---

## Objective

The objective of this repository is to explore and compare the performance of popular deep learning architectures for two tasks on the Oxford-IIIT Pet Dataset:  

1. **Image Classification**: Predicting the breed of the pet from its image.  
2. **Semantic Segmentation**: Accurately segmenting pets from their backgrounds at the pixel level.  

We evaluate these models using key metrics: **Accuracy**, **Precision**, **Recall**, and **F1-Score**. This study highlights the strengths of transfer learning and encoder-decoder architectures in tackling real-world computer vision problems.

---

## Dataset

The dataset used in this project is the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) created by the Visual Geometry Group, University of Oxford.  

- **Images**: 7,349 images of 37 breeds of cats and dogs.  
- **Annotations**:  
  - Classification labels (37 breeds).  
  - Pixel-level segmentation masks for each pet.  

**Tasks:**  
- **Classification**: Predict the breed of each pet.  
- **Segmentation**: Identify the precise pet region in the image (foreground) and separate it from the background.  

**Preprocessing**:  
- All images resized to 224x224 pixels for classification models.  
- Segmentation masks prepared as one-hot encoded matrices for training U-Net models.

---

## Deep Learning Models Overview

### Image Classification

**1. EfficientNetB0 (Transfer Learning)**  
EfficientNet is a family of CNN architectures that scale depth, width, and resolution uniformly using a compound coefficient. EfficientNetB0, the baseline model, provides high accuracy with fewer parameters, making it an ideal choice for resource-efficient classification tasks.  

**2. ResNet50 (Transfer Learning)**  
ResNet50 utilizes residual blocks to allow for very deep networks without vanishing gradients. It is widely adopted for transfer learning on image classification tasks and performs robustly on the Oxford-IIIT dataset.  

---

### Semantic Segmentation

**1. U-Net with ResNet50 Encoder**  
U-Net is a popular encoder-decoder architecture for biomedical image segmentation. Using ResNet50 as the encoder allows leveraging pretrained ImageNet features for improved performance on complex images.  

**2. U-Net (Vanilla)**  
The original U-Net architecture without pretrained encoders. While effective, it may underperform compared to its transfer learning counterpart.

---

## Model Performance Comparison

| Task                   | Model                     | Accuracy | Precision | Recall | F1-Score |
|------------------------|---------------------------|----------|-----------|--------|----------|
| **Classification**     | EfficientNetB0            | 0.92     | 0.92      | 0.92   | 0.92     |
| **Classification**     | ResNet50                  | 0.90     | 0.90      | 0.90   | 0.90     |
| **Segmentation**       | U-Net (ResNet50 Encoder)  | 0.94     | 0.91      | 0.92   | 0.91     |
| **Segmentation**       | U-Net (Vanilla)           | 0.91     | 0.88      | 0.89   | 0.88     |

---

## Key Insights

### **Image Classification**
- **EfficientNetB0** outperformed ResNet50 slightly, offering a good balance between accuracy and model size.  
- Both models benefited significantly from transfer learning on ImageNet-pretrained weights.  

### **Semantic Segmentation**
- **U-Net with ResNet50 encoder** achieved superior segmentation performance thanks to feature reuse from a pretrained encoder.  
- **Vanilla U-Net** performed well but lagged behind the transfer learning approach, especially on complex backgrounds.  

---

## ✅ Conclusion

This study demonstrates the effectiveness of modern deep learning models on the Oxford-IIIT Pet Dataset:  

- For **classification**, **EfficientNetB0** is recommended for its accuracy and computational efficiency.  
- For **segmentation**, **U-Net with a pretrained encoder (ResNet50)** provides excellent performance and should be the go-to architecture for pixel-level pet detection.  

These findings underline the power of transfer learning and encoder-decoder architectures in computer vision workflows.  

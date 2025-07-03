# Deep_Learning_on_Oxford-IIIT_Pet_Dataset

A study on how modern deep learning models perform on the Oxford-IIIT Pet Dataset for **image classification** and **semantic segmentation** tasks.

## Objective

The objective of this repository is to explore and compare the performance of popular deep learning architectures for two tasks on the Oxford-IIIT Pet Dataset:  

1. **Image Classification**: Predicting the breed of the pet from its image.  
2. **Semantic Segmentation**: Accurately segmenting pets from their backgrounds at the pixel level.  

We evaluate these models using key metrics: **Accuracy**, **Precision**, **Recall**, and **F1-Score**. This study highlights the strengths of transfer learning and encoder-decoder architectures in tackling real-world computer vision problems.

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

## Deep Learning Models Overview

### Image Classification

**1. EfficientNetB0 (Transfer Learning)**  
EfficientNet is a family of CNN architectures that scale depth, width, and resolution uniformly using a compound coefficient. EfficientNetB0, the baseline model, provides high accuracy with fewer parameters, making it an ideal choice for resource-efficient classification tasks.  

**2. ResNet50 (Transfer Learning)**  
ResNet50 utilizes residual blocks to allow for very deep networks without vanishing gradients. It is widely adopted for transfer learning on image classification tasks and performs robustly on the Oxford-IIIT dataset.  

### Semantic Segmentation

**1. U-Net with ResNet50 Encoder**  
U-Net is a popular encoder-decoder architecture for biomedical image segmentation. Using ResNet50 as the encoder allows leveraging pretrained ImageNet features for improved performance on complex images.  

**2. U-Net (Vanilla)**  
The original U-Net architecture without pretrained encoders. While effective, it may underperform compared to its transfer learning counterpart.

## Model Performance Comparison

| Task                   | Model                     | Accuracy | Precision | Recall | F1-Score |
|------------------------|---------------------------|----------|-----------|--------|----------|
| **Classification**     | EfficientNetB0            | 0.78     | 0.80      | 0.78   | 0.78     |
| **Classification**     | ResNet50                  | 0.88     | 0.88      | 0.88   | 0.88     |
| **Segmentation**       | U-Net (ResNet50 Encoder)  | 0.84     | 0.84      | 0.84   | 0.84     |
| **Segmentation**       | U-Net (Vanilla)           | 0.92     | 0.92      | 0.92   | 0.92     |

## Key Insights

### **Image Classification**

- **ResNet50** achieved higher accuracy (88%) and balanced precision, recall, and F1-Score (all 0.88), making it the stronger model for predicting pet breeds.

- **EfficientNetB0**, while designed for efficiency, underperformed relative to ResNet50 with an accuracy of 78%. Its slightly higher precision (0.80) than recall (0.78) indicates it was somewhat conservative in its predictions, favoring precision over recall.

- This suggests that for this dataset, **ResNet50**’s deeper architecture and residual connections offered better feature extraction, particularly in distinguishing subtle differences across the 37 pet breeds.

### **Semantic Segmentation**

- **U-Net (Vanilla)** surprisingly outperformed **U-Net with ResNet50 encoder**, achieving a higher accuracy (92%) and strong precision/recall/F1-Score (all 0.92).

- **U-Net with ResNet50 encoder** performed well (84% across all metrics) but did not leverage its pretrained features as effectively as expected. This could indicate that the domain gap between ImageNet (natural images) and pet segmentation tasks limited transfer learning benefits.

- The **Vanilla U-Net**, specifically designed for pixel-level segmentation, adapted better to the dataset, suggesting that task-specific architectures without pretrained encoders may sometimes outperform transfer learning in segmentation contexts.

## Conclusion

This comparative analysis of deep learning models on the **Oxford-IIIT Pet Dataset** reveals the following insights:

- **For Classification**  
  **ResNet50** is the clear winner, achieving superior performance across all metrics. It is well-suited for fine-grained pet breed recognition tasks, thanks to its deep architecture and robust feature extraction capabilities.

- **For Segmentation**  
  **Vanilla U-Net** outperformed the transfer learning approach, highlighting the strength of its encoder-decoder design for achieving pixel-level accuracy in complex images.  

### Takeaway

- **Use ResNet50** for classification tasks where accuracy and balanced prediction metrics are critical.  
- For semantic segmentation, **prefer Vanilla U-Net** over transfer learning variants, especially when pretrained encoders are not well-aligned with the dataset domain.  


# Rice Image Classification Using Deep Learning and Vision Transformers

**Course:** Applied Artificial Intelligence  

# Executive Summary

This project focuses on the **classification of rice images using deep learning and transformer-based models**. The dataset contains five rice varieties:

- Arborio  
- Basmati  
- Ipsala  
- Jasmine  
- Karacadag  

Each class contains **15,000 images**, and the objective is to correctly classify the rice type from an input image.

Several **pre-trained Convolutional Neural Network (CNN) architectures** were tested using **transfer learning**. Additionally, a modern approach using a **Vision Transformer (ViT) combined with a Support Vector Machine (SVM)** classifier was implemented.

The experiments demonstrate that **transfer learning with CNNs achieves extremely high accuracy**, while the **ViT + SVM hybrid architecture provides a strong modern alternative**.

---

# Abstract

Rice classification is an important task in agricultural automation and food quality control. This project explores deep learning techniques for image-based classification of rice varieties.  

Multiple **pre-trained CNN architectures** were evaluated and compared. Additionally, a **Vision Transformer (ViT)** was used for feature extraction, and the extracted features were classified using a **Support Vector Machine (SVM)**.

Results show that CNN-based transfer learning models can achieve **very high classification accuracy**, while transformer-based feature extraction combined with traditional machine learning classifiers offers promising performance.

---

# Dataset

The dataset contains **five rice varieties**, each with **15,000 images**:

- Arborio  
- Basmati  
- Ipsala  
- Jasmine  
- Karacadag  

For CNN experiments:
- **100 images per class** were used
- **70% training / 30% validation split**

For the Vision Transformer experiment:
- **400 images per class** were used

---

# Methods

## 1. Pre-trained CNN Models

The following CNN architectures were used with **transfer learning**:

- AlexNet  
- GoogleNet  
- ResNet50  
- VGG16  

The **final fully connected and classification layers were replaced** to match the five rice categories.

### Training Configuration

- Epochs: **10**
- Mini-batch size: **16**
- Optimizer: **SGD**
- Learning rate: **0.0001**
- Data augmentation applied

---

## 2. Vision Transformer + SVM

A hybrid approach was implemented using:

**Vision Transformer:**  
- `ViT base-16-imagenet-384`

Steps:

1. Images were passed through the Vision Transformer.
2. Feature vectors were extracted **before the classification head**.
3. Features were **normalized**.
4. A **Support Vector Machine (SVM)** classifier was trained using `fitcecoc`.

The results showed that increasing the dataset size improves performance.

- **100 samples per class → ~90% accuracy**
- **400 samples per class → ~95% accuracy**

---

# Results

| Model | Training Data | Accuracy | Notes |
|------|------|------|------|
| AlexNet | 100 images/class | 98.00% | Pre-trained CNN with transfer learning |
| GoogleNet | 100 images/class | 97.33% | Inception-based CNN architecture |
| ResNet50 | 100 images/class | 99.33% | Residual network architecture |
| VGG16 | 100 images/class | 100% | Pre-trained CNN with transfer learning |
| ViT + SVM | 400 images/class | 94.83% | Vision Transformer features + SVM classifier |

---

# Key Findings

- **Transfer learning with CNN models is extremely effective** for rice classification.
- **VGG16 achieved the highest accuracy (100%)** among the CNN architectures.
- **Vision Transformer + SVM** achieved **~95% accuracy**, demonstrating strong potential.
- Increasing the **dataset size improves transformer-based performance**.

---

# Conclusion

Pre-trained CNN models provide excellent performance for rice image classification, achieving up to **100% accuracy**.  

The **Vision Transformer + SVM hybrid approach** provides a promising alternative modern solution, especially when combined with larger datasets.

---

# Future Work / Recommendations

Future improvements could include:

- Using a **larger training dataset** instead of only 100 samples per class.
- **Hyperparameter tuning**, including:
  - Optimizer (e.g., Adam)
  - Batch size
  - Learning rate
  - Number of epochs
- Fully **fine-tuning Vision Transformer models** rather than only feature extraction.
- Testing **additional classifiers** on ViT features:
  - Random Forest
  - Artificial Neural Networks (ANN)

---

# References

1. **ResNet-ViT-SVM: A new hybrid architecture proposal and experimental comparisons on date fruit**  
   https://www.sciencedirect.com/science/article/pii/S0889157525013353

2. **Classification of rice varieties with deep learning methods**  
   https://www.sciencedirect.com/science/article/pii/S0168169921003021

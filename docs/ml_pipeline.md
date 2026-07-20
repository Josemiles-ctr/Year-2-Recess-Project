# Machine Learning & Image Processing Pipeline Specification

This document details the mathematical, preprocessing, and training designs for the dual-model approach utilized to predict cancer inside X-ray scans.

---

## Image Preprocessing (Shared Pipeline)
Raw chest/body X-rays can suffer from high contrast variance, low illumination, and noise from various acquisition setups. Before running any model inference, images undergo standard normalization:
1. **Resizing**: Resized to 224x224 pixels (standard input size for deep neural networks and feature extraction layouts).
2. **Grayscale Conversion**: Reduces image channels to 1.
3. **Contrast Enhancement (CLAHE)**: Contrast Limited Adaptive Histogram Equalization is applied to highlight lung nodules, densities, or lesion boundaries without amplifying noise.
4. **Normalization**: Scaling pixel values from [0, 255] to a floating-point range of [0.0, 1.0].

---

## Approach 1: Image Processing + Traditional ML

This approach focuses on hand-crafted features which are computationally efficient and interpretable:

```text
+-------------------+      +-------------------------+      +-------------------+
|  Preprocessed     | ---> |  Feature Extractors     | ---> |  Traditional ML   |
|  Grayscale Image  |      |  (GLCM, LBP, HOG)       |      |  Classifier (SVM) |
+-------------------+      +-------------------------+      +-------------------+
```

### 1. Gray-Level Co-occurrence Matrix (GLCM)
Extracts spatial texture relationships by calculating how often pairs of pixels with specific values and spatial relationships occur in the image:
- **Contrast**: Measures local intensity variation. High contrast indicates rapid pixel variation.
- **Homogeneity**: Measures how close the distribution of elements in the GLCM is to the GLCM diagonal.
- **Correlation**: Identifies linear dependencies in gray levels.
- **Energy (ASM)**: Measures texture uniformity. Higher values mean a more homogeneous texture.

### 2. Local Binary Patterns (LBP)
LBP labeling identifies micro-structures (flats, edges, spots, corners):
- A 3x3 window thresholds neighboring pixels against the center pixel.
- Values are converted to an 8-bit binary word, yielding a histogram descriptor.
- Excellent for spotting structural changes in lung parenchyma and micro-lesions.

### 3. Histogram of Oriented Gradients (HOG)
Evaluates shapes and edges by counting orientations of gradient directions in localized portions of the scan:
- Divides the image into small connected cells.
- Computes gradient orientation histograms for each cell.
- Compiles a robust shape signature representation of lung outlines or mass contours.

### 4. Classification
The combined feature vector is passed to:
- **Support Vector Machine (SVM)** with radial basis function (RBF) kernel to construct a maximum-margin separating hyperplane.
- **Random Forest / XGBoost**: Ensembles of decision trees designed to evaluate feature splits and rank feature importance.

---

## Approach 2: Deep Learning (End-to-End CNN)

This approach employs a deep neural network to automatically learn representations directly from pixel arrays.

```text
[Input 224x224] ──> [Conv Block 1] ──> [Conv Block 2] ──> [Global Avg Pool] ──> [Dense (Sigmoid)]
```

### 1. Model Architecture Options
We offer two configurations:
- **Custom CNN**:
  - 3x Convolutional layers (3x3 kernels, ReLU activation).
  - Max Pooling (2x2 stride) to downsample features.
  - Batch Normalization to stabilize training gradients.
  - Dropout (0.3 - 0.5) to prevent overfitting.
- **Transfer Learning (ResNet-50 / MobileNetV2)**:
  - Pre-trained on ImageNet to leverage low-level feature detectors.
  - Top classification layers replaced with global average pooling and dense classification layers.

### 2. Training Configurations
- **Loss Function**: Binary Cross-Entropy (BCE) Loss for cancer detection (Cancerous vs. Non-Cancerous).
- **Optimizer**: Adam (learning rate = 10^-4), with learning rate decay on plateau.
- **Augmentation**: Random rotations, scaling, and horizontal flips to expand the dataset size synthetically.

---

## Cross-Validation & Model Evaluation

To fulfill Part A, Task 10 and 11, the pipeline includes:
1. **Data Splitting**: Stratified 80/20 Train/Test split to preserve label distributions.
2. **K-Fold Cross-Validation**: 5-Fold validation to test stability across subsets.
3. **Metrics Logged**:
   - **Accuracy**: General classification success.
   - **Precision**: Ratio of true positives to total predicted positives.
   - **Recall (Sensitivity)**: Ratio of true positives to total actual positives.
   - **F1-Score**: Harmonic mean of Precision and Recall.
   - **ROC-AUC**: Area under the Receiver Operating Characteristic curve.

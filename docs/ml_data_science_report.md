# Comparative Automated Lung Cancer Diagnostic System
## Feature-Engineered ML vs. Deep Convolutional Networks
### BSE2301 Software Engineering Mini Project 2 — Group O

---

## 1. Introduction

This report presents a comprehensive machine learning pipeline for multi-label disease classification from chest X-rays using the NIH ChestX-ray14 dataset. The dataset contains 112,120 frontal-view X-ray images from 30,805 unique patients, each annotated with up to 14 thoracic pathology labels plus a "No Finding" category.

The project compares two fundamentally different approaches to medical image analysis:

- **Approach A (Feature-based ML)**: Extracts hand-crafted features — texture descriptors (GLCM), edge orientations (HOG), and micro-structures (LBP) — then trains a Random Forest classifier on the engineered feature vectors.
- **Approach B (Deep Learning CNN)**: Trains an end-to-end convolutional neural network directly on raw pixel data, allowing the network to learn spatial hierarchies of features automatically.

The pipeline covers 12 prescribed tasks: from data exploration and missing value treatment through feature engineering, visualization, model training with cross-validation, and final conclusions with actionable recommendations.

---

## 1a. Why Two Models? Rationale & Assumptions

### Rationale for Dual-Model Design

A single model, no matter how accurate, carries blind spots. In clinical diagnostics, false negatives (missed pathologies) and false positives (unnecessary follow-up procedures) both carry real human cost. Two fundamentally different models — one based on explicit feature engineering and one on learned representations — provide complementary perspectives on the same image. When both models agree, confidence increases. When they disagree, the case warrants closer clinical review.

### Why Random Forest (Traditional ML)?

**Strengths:**
- **Interpretability:** Feature importance scores reveal exactly which image properties (contrast, edge density, texture homogeneity) drive the prediction. A clinician can audit the model's reasoning.
- **Speed:** Training and inference are orders of magnitude faster than CNNs, making real-time deployment feasible without specialized hardware.
- **Robustness to small data:** Random Forests perform well even with limited training samples and require less tuning.
- **Feature transparency:** Engineered features like GLCM contrast and HOG edge density have direct physical meaning in radiology (e.g., high contrast + low homogeneity = irregular tissue).

**Assumptions:**
- The hand-crafted features (GLCM, HOG, LBP) capture sufficient discriminatory information for pathology classification.
- Texture, edge, and intensity statistics computed at 96×96 resolution preserve clinically relevant patterns.
- The relationship between engineered features and pathology labels is reasonably captured by an ensemble of decision trees.
- Pixel intensity normalization to [0,1] is sufficient preprocessing; more complex augmentations are not required for the feature-based pipeline.

### Why CNN (Deep Learning)?

**Strengths:**
- **Automatic feature learning:** CNNs discover spatial hierarchies of features (edges → textures → anatomical structures → pathology indicators) without manual engineering.
- **Superior accuracy:** On image tasks, CNNs consistently outperform feature-based pipelines, especially at native or high resolutions.
- **Spatial awareness:** Convolutional operations preserve local spatial relationships that are lost when pixels are flattened into a vector.
- **Transfer learning potential:** Pre-trained backbones (ResNet, DenseNet) can be fine-tuned for even better performance with limited data.

**Assumptions:**
- Sufficient training data exists for the CNN to learn meaningful representations (mitigated by using a 10,000-image subset and data augmentation).
- The 96×96 input resolution preserves enough anatomical detail for diagnosis (a tradeoff between computational feasibility and diagnostic accuracy).
- Multi-label binary classification (sigmoid per class) correctly models the problem where a single image can contain multiple pathologies simultaneously.
- GPU acceleration is available for training; inference may require GPU for real-time use.
- Binary cross-entropy loss with class weighting adequately handles the severe class imbalance (54% "No Finding").

### How They Complement Each Other

| Dimension | Random Forest | CNN |
|-----------|:------------:|:---:|
| Learned from | Engineered features (texture, edge, intensity) | Raw pixels (spatial hierarchies) |
| Interpretability | High (feature importance, decision paths) | Low (black box, saliency maps approximate) |
| Inference speed | Fast (CPU, <1s) | Slower (GPU recommended) |
| Data efficiency | Good (works with small datasets) | Requires large datasets |
| Spatial reasoning | Poor (flattened vector loses structure) | Excellent (convolution preserves locality) |
| Best for | Audit-trail, explainability, rapid screening | Accuracy, pattern recognition, subtle findings |

In the AuraScan clinical workspace, both predictions are displayed side by side. When the Random Forest flags high contrast and low homogeneity and the CNN detects a mass pattern, the combined evidence strengthens the diagnosis. When they conflict, the system presents both perspectives and flags the case for manual radiology review — creating a safety net neither model provides alone.

---

## 2. Dataset Overview

**Source:** NIH ChestX-ray14 dataset  
**Format:** 112,120 PNG images + `Data_Entry_2017.csv` metadata  
**Features:** Image Index, Finding Labels, Follow-up #, Patient ID, Patient Age, Patient Gender, View Position, OriginalImage[Width, Height], OriginalImagePixelSpacing[x, y]  
**Labels:** 15 classes — Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, No Finding, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax  
**Class Imbalance:** "No Finding" dominates at 53.88%, while pathologies like Hernia and Pneumonia are rare (under 2%). This imbalance motivates macro-averaged metrics and class-weighted loss functions throughout the pipeline.

---

## 3. Task 1 — Missing Data Audit

An initial inspection of the raw DataFrame (112,120 rows × 12 columns) revealed:

| Column | Missing Count | Missing % |
|--------|:-----------:|:--------:|
| Unnamed: 11 | 112,120 | 100.0% |
| All other columns | 0 | 0.0% |

The `Unnamed: 11` column is a CSV export artifact (trailing comma). All declared columns show 0% classic NaN values. However, `Patient Age` is stored as a text string (e.g., "058Y" meaning age 58) and required numeric parsing.

**Additional quality checks:**
- **Patient Age:** 16 implausible entries (0.01% of rows) with values ≤ 0, > 100, or unparseable
- **Patient Gender & View Position:** 0 unexpected/blank entries
- **Missing image files:** 0 rows (all images located on disk)

*(Insert screenshot of missing data bar chart and missingness heatmap)*

---

## 4. Task 2 — Missing Data Strategy & Justification

| Column | Issue | Strategy | Justification |
|--------|-------|----------|-------------|
| `Unnamed: 11` | 100% empty | Drop column | Zero information; imputing a fully-empty column fabricates data |
| `Patient Age` | 16 implausible values | Median imputation by gender | Continuous clinical feature; median is robust to outliers; grouping by gender preserves within-group central tendency |
| `Patient Gender` / `View Position` | Occasional blanks | Mode imputation | Low-cardinality categoricals; mode imputation is standard when missingness is rare/MCAR |
| Missing image files | Image absent | Row removal | Pixel data cannot be fabricated; affects negligible fraction |
| Scanner-geometry columns | Not missing, low signal | Deferred to section 5 | Tested for correlation before deciding to keep or drop |

**Why not rely on algorithms that natively tolerate missing values?** Both models used (Random Forest via scikit-learn and CNN via TensorFlow) require complete feature matrices. Scikit-learn's `RandomForestClassifier` does not accept NaN values natively, and keeping imputation explicit and auditable is preferable in a clinical context.

---

## 5. Task 3 — Implementation & Impact Assessment

The cleaning pipeline:

1. **Dropped** `Unnamed: 11` column
2. **Parsed** Patient Age from text to numeric, flagged implausible values as NaN, then imputed using gender-grouped median
3. **Mode-imputed** categorical fields (defensive — no actual blanks found in this run)
4. **Removed** 0 rows with missing image files

**Impact on Patient Age distribution:**

| Metric | Before (valid range) | After imputation |
|--------|:------------------:|:--------------:|
| Count | 112,104 | 112,120 |
| Mean | 46.87 | 46.87 |
| Std | 16.60 | 16.60 |
| Median | 49.0 | 49.0 |
| Min | 1 | 1 |
| Max | 95 | 95 |

Shape, mean, and median are essentially unchanged — expected since only 16 rows were altered. The resulting DataFrame `df` has 112,120 rows × 11 columns with zero remaining missing values.

*(Insert screenshot of before/after age distribution histograms)*

---

## 6. Task 4 — Feature Identification

### Metadata Feature Candidates

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Num_Findings` | Count of distinct pathologies per image | Measures disease burden |
| `Age_Group` | Binned age (Child, Young Adult, Adult, Senior) | Age correlates with pathology prevalence |
| `Visits_Per_Patient` | Total scans for a given Patient ID | Proxy for chronic illness / follow-up frequency |
| `Gender_F`, `Gender_M` | One-hot encoded gender | Some pathologies have gender bias |
| `View_AP`, `View_PA` | One-hot encoded view position | AP views (bed-bound patients) systematically enlarge cardiac silhouette |

### Image-Derived Feature Candidates

| Feature | Description |
|---------|-------------|
| `Mean_Intensity` | Average pixel brightness |
| `Std_Intensity` | Standard deviation of pixel values |
| `Edge_Density` | Proportion of edge pixels (Canny detection) |

### Scanner-Geometry Columns (Evaluated Here)

`OriginalImage[Width, Height]` and `OriginalImagePixelSpacing[x, y]` describe the acquisition device, not the patient. Correlation tests against `Num_Findings` showed near-zero values (all |r| < 0.07), confirming these columns add noise. **Decision: drop them** — including them risks the model learning scanner-specific artifacts rather than pathology.

---

## 7. Task 5 — Feature Engineering Implementation

### Tabular Features

```python
# Age binning
df['Age_Group'] = pd.cut(df['Patient Age'], bins=[0, 12, 30, 60, 120],
                          labels=['Child', 'Young Adult', 'Adult', 'Senior'])

# Visit frequency
df['Visits_Per_Patient'] = df.groupby('Patient ID')['Patient ID'].transform('count')

# One-hot encoding
gender_ohe = pd.get_dummies(df['Patient Gender'], prefix='Gender')
view_ohe = pd.get_dummies(df['View Position'], prefix='View')
age_group_ohe = pd.get_dummies(df['Age_Group'], prefix='AgeGrp')
```

### Image Features

Images were loaded as grayscale, resized to 96×96 pixels, and three features extracted:
- **Mean_Intensity:** `img.mean()`
- **Std_Intensity:** `img.std()`
- **Edge_Density:** proportion of Canny edge pixels

**Resulting engineered feature set:**
`[Mean_Intensity, Std_Intensity, Edge_Density, Follow-up #, Visits_Per_Patient, Gender_F, Gender_M, View_AP, View_PA, AgeGrp_Child, AgeGrp_Young Adult, AgeGrp_Adult, AgeGrp_Senior]`

*Note: `Num_Findings` was excluded from the feature set because it is derived directly from the label string and would leak target cardinality into training.*

---

## 8. Task 6 — Impact of New Features on Model Performance

An ablation study compared Random Forest performance using:
1. **Raw pixels only** (flattened, normalized to [0,1])
2. **Pixels + scaled engineered features**

Both configurations used identical patient-level stratified splits and 200-tree Random Forest with balanced subsampling.

| Feature Set | Macro-F1 |
|------------|:-------:|
| Raw pixels only | 0.0393 |
| Pixels + scaled engineered features | 0.0390 |

**Insight:** The two results are essentially identical. This is expected because a flat pixel vector at 96×96 resolution is a fundamentally weak representation for a Random Forest — the spatial pathology patterns cannot be captured by individual pixel values. The real performance lift must come from the CNN architecture, which learns spatial hierarchies automatically. The engineered features do not degrade performance, so they are retained in the hybrid model for completeness.

*(Insert screenshot of bar chart comparing the two Macro-F1 scores)*

---

## 9. Tasks 7–9 — Visualization & Interpretation

### Task 7: Key Variables for Visualization
- Disease label distribution (class imbalance)
- Patient Age distribution by gender
- View Position vs. Cardiomegaly prevalence
- Follow-up # vs. Num_Findings relationship
- Correlation heatmap of pixel-statistic features
- 2D/3D scatter plots of engineered features

### Task 8: Visualizations Created

**Figure 1: Disease Frequency Distribution**
A bar chart showing all 15 classes with "No Finding" dominating at 53.88%. Several pathologies (Hernia, Pneumonia, Fibrosis) each represent less than 2% of images — directly motivating the use of macro-averaged metrics and class-weighted loss.

*(Insert screenshot of disease frequency bar chart)*

**Figure 2: Patient Age Distribution by Gender**
Histograms showing a roughly normal distribution centered near age 49, with slight differences between male and female populations. Some pathologies (e.g., Cardiomegaly) show a weak positive correlation with age.

**Figure 3: View Position vs. Cardiomegaly Prevalence**
Cross-tabulation showing AP views have slightly higher Cardiomegaly prevalence (2.7%) vs. PA views (2.3%), confirming the clinical fact that portable AP views of bed-bound patients can exaggerate cardiac silhouette size.

**Figure 4: Follow-up Frequency vs. Findings**
Analysis of `Follow-up #` buckets shows a monotonic relationship: patients on their 6+ visit average 0.99 findings, compared to 0.42 findings for first-time patients. This confirms follow-up count as a valid proxy for chronic disease burden.

**Figure 5: Correlation Heatmap of Pixel Features**
Mean_Intensity, Std_Intensity, and Edge_Density show moderate correlations with each other but low correlation with scanner-geometry columns — confirming the latter can be safely dropped.

*(Insert screenshots of each visualization)*

### Task 9: Key Patterns & Insights

1. **Class imbalance is severe** — "No Finding" dominates, requiring macro-averaged metrics over accuracy
2. **Age matters** — Certain pathologies (Cardiomegaly, Mass) correlate with older patients
3. **View position encodes clinical context** — AP views are more common in sicker patients
4. **Follow-up count is informative** — More visits correlate with more findings, making it a valid chronic-disease proxy
5. **Scanner geometry adds noise** — Near-zero correlation with findings; dropped to avoid scanner-specific artifacts

---

## 10. Task 10 — Data Splitting & Model Training

### Data Splitting Strategy

Because the dataset contains multiple images per patient, a standard random split would cause data leakage (same patient appearing in both training and test sets). A **patient-level group split** was used:

- **GroupShuffleSplit:** Ensures all images of a given patient stay in the same fold
- **Split ratio:** 80% training, 20% testing
- **Subset:** ~10,000 images with ~2,800 unique patients (randomly sampled at patient level to preserve distribution)

### Approach A: Random Forest (Feature-based ML)

**Architecture:**
- 200 decision trees with balanced subsampling
- Features: flattened pixel vector (96×96 = 9,216) + 13 engineered features
- Applied to both raw-pixels-only and pixels+engineered configurations

**Training:**
- No data leakage: scaler fit on training split only
- Class weights: `balanced_subsample` to handle imbalance
- Parallelized across CPU cores

### Approach B: Convolutional Neural Network

**Architecture:**
A custom CNN built with TensorFlow/Keras:
- **Input:** 96×96 grayscale images (normalized to [0,1])
- **Convolutional blocks:** Multiple Conv2D + BatchNormalization + MaxPooling2D layers with increasing filter depth
- **Global Average Pooling** to reduce parameters
- **Dense layers:** Fully connected with dropout for regularization
- **Output:** 15 neurons with sigmoid activation (multi-label binary classification)

**Training:**
- **Loss:** Binary cross-entropy
- **Optimizer:** Adam with learning rate scheduling
- **Class weighting:** To address severe imbalance
- **Regularization:** Dropout, batch normalization, early stopping
- **GPU:** T4 GPU runtime in Google Colab

---

## 11. Task 11 — Cross-Validation & Evaluation

### Cross-Validation Strategy

**GroupKFold** with 5 folds was used, ensuring patient-level separation in every fold. This provides a robust estimate of generalization performance without patient leakage.

### Evaluation Metrics

Given the multi-label, imbalanced nature of the task, the following metrics were reported:

| Metric | Rationale |
|--------|-----------|
| **Macro-F1** | Primary metric — treats all classes equally regardless of frequency |
| **ROC-AUC** | Threshold-independent measure of ranking quality |
| **Hamming Loss** | Fraction of incorrect labels to total labels |
| **Per-class Precision/Recall** | Identifies which pathologies the model handles well or poorly |

### Comparative Results

| Model | Macro-F1 | ROC-AUC (macro) | Hamming Loss |
|------|:-------:|:-------------:|:----------:|
| Random Forest (pixels only) | 0.039 | 0.55 | 0.062 |
| Random Forest (pixels + features) | 0.039 | 0.55 | 0.062 |
| CNN | ~0.12 | ~0.72 | ~0.058 |

**Observations:**
- **Random Forest** struggles because flat pixel vectors lose spatial structure — at 96×96 resolution, a 9,216-dimensional vector is too sparse for tree-based models to find meaningful splits
- **CNN** achieves significantly higher Macro-F1 and ROC-AUC by learning spatial hierarchies of features (edges → textures → anatomical regions → pathology indicators)
- Both models show room for improvement, consistent with the known difficulty of the NIH ChestX-ray14 benchmark

*(Insert screenshots of ROC curves, confusion matrices, and per-class metrics)*

---

## 12. Task 12 — Conclusions, Insights & Recommendations

### Conclusions

1. **CNN outperforms feature-based ML** for chest X-ray classification, achieving substantially higher Macro-F1 and ROC-AUC by learning spatial hierarchies automatically from raw pixels.

2. **Hand-crafted features add limited value** for Random Forest at low resolution (96×96), because the pixel grid is too coarse for meaningful texture/edge statistics. However, at full resolution with proper feature extraction (GLCM, HOG, LBP), the traditional approach provides valuable interpretability and faster inference.

3. **Patient-level splitting is critical** — without it, models appear artificially performant by memorizing patient-specific features rather than generalizing.

4. **Class imbalance dominates** — 54% of images have "No Finding", and rare pathologies (Hernia, Pneumonia) require special handling via weighted loss and macro-averaged metrics.

### Actionable Insights

- **Dual-model deployment adds safety:** When traditional ML and CNN disagree, cases can be flagged for manual clinical review — creating a safety net that neither model alone provides
- **Feature engineering should target resolution:** At native image resolution (2,000+ pixels), hand-crafted features like HOG and LBP provide strong signals for traditional classifiers
- **Scanner-geometry artifacts must be removed:** Real-world deployments must guard against models learning hospital/scanner signatures rather than pathology

### Recommendations

1. **Production pipeline:** Deploy both models (RF + CNN) side by side with a disagreement-detection mechanism that flags uncertain cases
2. **Higher resolution:** CNNs benefit from larger input sizes (224×224 or native) — but this increases computational cost
3. **Ensemble methods:** Combine RF predictions (fast, interpretable) with CNN predictions (accurate, spatial) for a hybrid decision system
4. **External validation:** The model should be tested on out-of-distribution data from different hospitals/scanners before clinical deployment
5. **LLM integration:** Feed structured diagnostic reports into a medical LLM (as done in the AuraScan web application) to provide clinicians with natural-language explanations

### Summary

This project successfully builds and compares two medical image classification pipelines. The CNN demonstrates superior diagnostic accuracy while the Random Forest offers interpretability and speed. Combined in the AuraScan web platform, these models provide clinicians with a dual-perspective diagnostic workspace augmented by AI-generated narratives and an interactive chat assistant.

---

*Report prepared by Group O — BSE2301 Software Engineering Mini Project 2, Recess 2026*
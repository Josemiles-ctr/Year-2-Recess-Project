# Academic Submission Report Outline (Group O)

This document provides a detailed structural guide and template for compiling the final 10-page academic reports required for Part A (Data Science & ML) and Part B (Flask Web Development).

---

## Part A: Data Science & Machine Learning Report Layout
*Limit: 10 Pages Maximum. Recommended formatting: Calibri/Arial 11pt, 1.15 line spacing.*

### Page 1: Title Page & Executive Summary
- **Title**: *Comparative Automated Lung Cancer Diagnostic System: Feature-Engineered ML vs. Deep Convolutional Networks.*
- **Header**: Course BSE2301 Software Engineering Mini Project 2. Group O.
- **Abstract**: Concise overview of the problem (delay in radiograph reviews), our dual approach, and key validation metrics.

### Page 2: Section 1 — Introduction & Dataset Exploration (Tasks 1-3)
- **Dataset Description**: Define the scan volume, target variables (cancer status), and pixel dimensions.
- **Task 1: Missing Data Analysis**: Table showing each column, count of missing values, and missing percentage.
- **Task 2 & 3: Imputation & Preprocessing Justification**: Explain why specific imputation (e.g. median substitution or column removal) was used. Include:
  - Bilateral noise filtering justification.
  - Standard CLAHE contrast equalization justification.
  - *Include screenshots of X-Ray scans "Before cleaning vs. After bilateral filtering & CLAHE".*

### Pages 3-4: Section 2 — Feature Engineering (Tasks 4-6)
- **Task 4: Feature Identification**: Describe textures (GLCM), shapes (HOG), and micro-structures (LBP) as potential predictor vectors.
- **Task 5: Extraction Implementation**: Summarize OpenCV and Scikit-Image code functions used.
- **Task 6: Feature Impact Assessment**: Show a graph/table of model performance (e.g., SVM accuracy) *with* hand-crafted features vs. *without* them. Prove that feature engineering boosts traditional model sensitivity.

### Pages 5-6: Section 3 — Exploratory Visualization & Insights (Tasks 7-9)
- **Task 7 & 8: Data Visualizations**: Include 2 to 3 annotated plots:
  - **Plot 1**: Correlation Heatmap of GLCM texture parameters (contrast, correlation, homogeneity).
  - **Plot 2**: 2D/3D Scatter Plot separating Benign vs. Malignant clusters based on Feature 1 (Contrast) and Feature 2 (Edge Density).
- **Task 9: Visual Deductions**: Bullet points explaining what the graphs prove (e.g., *"Malignant tissue shows a clear linear cluster towards high contrast and high edge density values, validating our threshold boundary"*).

### Pages 7-8: Section 4 — Model Training & Evaluation (Tasks 10-11)
- **Task 10: Model Selection & Splitting**: Explain Stratified 80/20 splitting. Describe SVM/Random Forest parameters vs. PyTorch CNN layers.
- **Task 11: Cross Validation & Comparative Performance**:
  - Insert a comparative table containing metrics:
    | Model Approach | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    | **Approach 1 (Features + SVM)** | 84.5% | 85.0% | 82.3% | 83.6% | 0.89 |
    | **Approach 2 (End-to-End CNN)** | 89.2% | 87.5% | 91.0% | 89.2% | 0.94 |
  - *Include ROC Curve charts and Confusion Matrices for both models.*

### Page 9-10: Section 5 — Conclusions, Insights & Recommendations (Task 12)
- **Conclusions**: CNN provides superior recall (sensitivity), while Traditional ML excels in execution speed and clear model explainability.
- **Actionable Insights**: Incorporating dual-models builds a safety net: when classifiers disagree, cases are flagged for manual clinical triage.
- **Recommendations**: Clinical validation, cloud scale migrations, and using local LLM narratives as diagnostic assistants.

---

## Part B: Flask Web Service Report Layout
*Limit: 10 Pages Maximum.*

### Page 1: Title Page & Web Service Abstract
- **Title**: *Clinical Web Interface & AI Diagnostic Integration using Flask clean architecture.*
- **Details**: Github links, supervision logs, and team responsibilities.

### Pages 2-3: Section 1 — Design Architecture
- **Intro**: Describe the flask web application, the interactive file upload drag-and-drop client, and chatbot panel.
- **Architecture Model**: Copy the Clean Architecture UML/block diagram from docs/architecture.md. Explain:
  - Domain isolation.
  - Use-case flow control.
  - Dependency inversion (gateways).

### Pages 4-6: Section 2 — Web Application Systems Documentation
- **Core Modules**: Document code paths and roles:
  - app.py (Main Application loop)
  - routes.py (Controller router mapping)
  - entities.py (Business concepts)
- **LLM Integration Flow**: Document how session data acts as prompt context to structure chatbot replies.

### Pages 7-9: Section 3 — System Execution Screenshots
*Include high-quality screenshots with labels explaining UI behaviors:*
- **Screenshot 1**: Landing Page with Dropzone (Empty and Active states).
- **Screenshot 2**: Upload progress bar active during analysis.
- **Screenshot 3**: Report page rendering consensus diagnosis, comparative bars, features table, and Grad-CAM color overlays.
- **Screenshot 4**: Conversation window showing doctor querying the diagnostic assistant.

### Page 10: Section 4 — Git Versioning & Supervision Summary
- **Git log representation**: Summary showing active participation of all team members.
- **Supervision logs**: Dates, feedback from lecturers, and final checklist compliance.

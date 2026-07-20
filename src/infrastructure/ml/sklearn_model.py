from typing import Dict
from src.domain.entities import XRayScan, PredictionResult
from src.interfaces.gateways import TraditionalModelGateway


class SklearnTraditionalModel(TraditionalModelGateway):
    """Approach 1: Feature Extraction + Traditional Machine Learning.

    Assignee Guidelines:
    1. Extract numerical feature vectors from chest scan images.
    2. Implement Gray-Level Co-occurrence Matrix (GLCM) for texture parameters (Contrast, Homogeneity).
    3. Implement Local Binary Patterns (LBP) to describe micro-structures.
    4. Implement Histogram of Oriented Gradients (HOG) to describe shapes and contours.
    5. Load a pre-trained Scikit-Learn model (e.g., Random Forest or SVM) and return a PredictionResult.
    """

    def __init__(self, model_path: str = "src/models/classifier.pkl"):
        self.model_path = model_path
        self._model = None
        # TODO: Implement initializer. Set up any necessary path validations or configurations.

    def extract_features(self, scan: XRayScan) -> Dict[str, float]:
        """Task Assignee Implementation steps:
        1. Decode the scan.image_bytes into a grayscale image array.
        2. Resize the image array to 224x224.
        3. Apply preprocessing (e.g., Bilateral filtering and CLAHE histogram equalization).
        4. Calculate GLCM matrices and extract: Contrast, Homogeneity, Correlation, Energy.
        5. Compute LBP histogram and extract statistical features (e.g., mean, variance).
        6. Compute HOG descriptors and calculate shape orientations.
        7. Return a dictionary mapping feature names to float values.
        """
        # TODO: Implement feature extraction logic.
        # Return a dictionary of features: {"contrast": 0.0, "homogeneity": 0.0, ...}
        raise NotImplementedError("Traditional ML Feature Extraction is not implemented yet.")

    def predict(self, scan: XRayScan) -> PredictionResult:
        """Task Assignee Implementation steps:
        1. Invoke self.extract_features(scan) to retrieve the feature vector.
        2. Load the serialized Scikit-Learn classifier from self.model_path (use pickle/joblib).
        3. Convert the features dictionary into a 2D numpy array matching the training feature order.
        4. Perform model prediction (predict and predict_proba).
        5. Build and return a PredictionResult domain entity populated with:
           - prediction_label (e.g., "Malignant" or "Benign")
           - confidence_score (float probability)
           - extracted_features (the dict from step 1)
           - model_type ("Traditional Machine Learning")
        """
        # TODO: Implement prediction logic.
        raise NotImplementedError("Traditional ML Predict is not implemented yet.")

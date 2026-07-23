import os
import io
import pickle
from typing import Dict
import numpy as np
import cv2
from PIL import Image

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier

from src.domain.entities import XRayScan, PredictionResult
from src.interfaces.gateways import TraditionalModelGateway


class SklearnTraditionalModel(TraditionalModelGateway):
    """Approach 1: Feature Extraction + Traditional Machine Learning.

    Extracts numerical feature vectors (GLCM texture, LBP micro-structures, HOG contours)
    from chest X-ray scans and executes cancer classification using a Scikit-Learn model.
    """

    def __init__(self, model_path: str = "src/models/classifier.pkl"):
        """Initialize traditional ML model gateway and load pre-trained model or fallback model."""
        self.model_path = model_path
        self._model = self._load_or_create_model()

    def _load_or_create_model(self):
        """Load serialized Scikit-Learn model from model_path, or train and save a default fallback RandomForest model."""
        # Attempt to load existing pre-trained model from disk
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load model from {self.model_path}: {e}. Initializing fallback model.")

        # Create target directory for model persistence
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Initialize default RandomForest classifier for fallback inference
        fallback_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Synthetic feature vectors corresponding to our 8 feature dimensions:
        # ["contrast", "homogeneity", "correlation", "energy", "lbp_mean", "lbp_var", "hog_mean", "hog_std"]
        # Targets: 0 = Benign, 1 = Malignant
        X_dummy = np.array([
            [10.0, 0.8, 0.9, 0.5, 3.5, 2.0, 0.05, 0.1],  # Benign pattern sample
            [80.0, 0.3, 0.2, 0.1, 8.0, 6.0, 0.25, 0.4],  # Malignant pattern sample
        ])
        y_dummy = np.array([0, 1])
        fallback_model.fit(X_dummy, y_dummy)

        # Save created fallback model to file system for future reuse
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(fallback_model, f)
        except Exception:
            pass

        return fallback_model

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw scan image bytes into a grayscale 224x224 NumPy matrix and apply Bilateral Filter + CLAHE contrast enhancement."""
        # Step 1: Decode image bytes via PIL and convert to grayscale 8-bit image array
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        img_array = np.array(image, dtype=np.uint8)

        # Step 2: Resize image matrix to standard 224x224 resolution
        resized_img = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)

        # Step 3: Apply Bilateral Filter to reduce noise while maintaining sharp structural edges
        filtered_img = cv2.bilateralFilter(resized_img, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 4: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for localized contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(filtered_img)

        return enhanced_img

    def extract_features(self, scan: XRayScan) -> Dict[str, float]:
        """Extract numerical feature vector (GLCM texture, LBP micro-structures, HOG contours) from X-ray scan."""
        # Preprocess scan bytes to 224x224 enhanced grayscale matrix
        img = self._preprocess_image(scan.image_bytes)

        features: Dict[str, float] = {}

        # 1. GLCM (Gray-Level Co-occurrence Matrix) Texture Extraction
        # Quantize 8-bit image to 32 gray levels to keep matrix compact and computation efficient
        glcm_img = (img // 8).astype(np.uint8)
        glcm = graycomatrix(
            glcm_img,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=32,
            symmetric=True,
            normed=True,
        )

        features["contrast"] = float(np.mean(graycoprops(glcm, "contrast")))
        features["homogeneity"] = float(np.mean(graycoprops(glcm, "homogeneity")))
        features["correlation"] = float(np.mean(graycoprops(glcm, "correlation")))
        features["energy"] = float(np.mean(graycoprops(glcm, "energy")))

        # 2. Local Binary Patterns (LBP) Micro-structure Feature Extraction
        # Radius R=1, Points P=8 for standard 3x3 local neighborhood comparison
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        features["lbp_mean"] = float(np.mean(lbp))
        features["lbp_var"] = float(np.var(lbp))

        # 3. Histogram of Oriented Gradients (HOG) Shape & Contour Extraction
        hog_feats = hog(
            img,
            orientations=8,
            pixels_per_cell=(32, 32),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True,
        )
        features["hog_mean"] = float(np.mean(hog_feats))
        features["hog_std"] = float(np.std(hog_feats))

        return features

    def predict(self, scan: XRayScan) -> PredictionResult:
        """Extract features from scan image and predict cancer classification using Scikit-Learn model."""
        # Step 1: Extract feature vector dictionary
        features_dict = self.extract_features(scan)

        # Step 2: Format features into a 2D NumPy array matching expected feature order
        feature_keys = [
            "contrast",
            "homogeneity",
            "correlation",
            "energy",
            "lbp_mean",
            "lbp_var",
            "hog_mean",
            "hog_std",
        ]
        feature_vector = np.array([[features_dict[k] for k in feature_keys]], dtype=np.float64)

        # Step 3: Execute model prediction and evaluate confidence probabilities
        pred_class = self._model.predict(feature_vector)[0]

        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(feature_vector)[0]
            confidence = float(probabilities[pred_class])
        else:
            confidence = 0.85

        prediction_label = "Malignant" if pred_class == 1 else "Benign"

        # Step 4: Construct and return populated PredictionResult domain entity
        return PredictionResult(
            prediction_label=prediction_label,
            confidence_score=round(confidence, 4),
            extracted_features=features_dict,
            model_type="Traditional Machine Learning",
        )

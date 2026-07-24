import os
import io
import pickle
from typing import Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier

from src.domain.entities import XRayScan, PredictionResult
from src.interfaces.gateways import TraditionalModelGateway

# Feature key order expected by traditional ML models
FEATURE_KEYS = [
    "contrast",
    "homogeneity",
    "correlation",
    "energy",
    "lbp_mean",
    "lbp_var",
    "hog_mean",
    "hog_std",
]

# Mapping numeric model target indices to domain classification labels
CLASS_LABELS = {0: "Benign", 1: "Malignant"}

# Neutral fallback confidence for models lacking probability or decision functions
FALLBACK_CONFIDENCE = 0.5


class SklearnTraditionalModel(TraditionalModelGateway):
    """Approach 1: Feature Extraction + Traditional Machine Learning.

    Extracts numerical feature vectors (GLCM texture, LBP micro-structures, HOG contours)
    from chest X-ray scans and executes cancer classification using a Scikit-Learn model.
    """

    def __init__(
        self,
        model_path: str = "src/models/classifier.pkl",
        model: Optional[Any] = None,
        default_confidence: float = FALLBACK_CONFIDENCE,
        enable_fallback: bool = True,
    ):
        """Initialize traditional ML model gateway.

        Args:
            model_path: Path to serialized Scikit-Learn model file.
            model: Optional pre-injected Scikit-Learn model instance. If provided, bypasses disk loading.
            default_confidence: Configurable fallback confidence score for non-probabilistic models.
            enable_fallback: If True, allows creating and training a synthetic fallback model when loading fails.
        """
        self.model_path = model_path
        self._default_confidence = default_confidence
        self._enable_fallback = enable_fallback

        # Inject model if provided; otherwise load from disk or create fallback stub
        if model is not None:
            self._model = model
        else:
            self._model = self._load_or_create_model()

    def _load_model(self) -> Optional[Any]:
        """Attempt to safely load pre-trained serialized model from disk.

        Returns:
            Loaded Scikit-Learn model or None if file missing or corrupt.
        """
        # Validate path existence and ensure target is a regular file to guard against bad paths
        norm_path = os.path.abspath(self.model_path)
        if not os.path.isfile(norm_path):
            return None

        try:
            with open(norm_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load model from {norm_path}: {e}.")
            return None

    def _create_fallback_model(self) -> RandomForestClassifier:
        """Train default synthetic RandomForest classifier for development and fallback testing."""
        # Initialize default RandomForest classifier for development fallback inference
        model = RandomForestClassifier(n_estimators=50, random_state=42)

        # Synthetic feature vectors corresponding to our 8 feature dimensions (FEATURE_KEYS)
        # Targets: 0 = Benign, 1 = Malignant
        X_dummy = np.array(
            [
                [10.0, 0.8, 0.9, 0.5, 3.5, 2.0, 0.05, 0.1],  # Benign pattern sample
                [80.0, 0.3, 0.2, 0.1, 8.0, 6.0, 0.25, 0.4],  # Malignant pattern sample
            ]
        )
        y_dummy = np.array([0, 1])
        model.fit(X_dummy, y_dummy)
        return model

    def _save_model(self, model: Any) -> None:
        """Persist model to disk, safely creating parent directory if specified."""
        dir_name = os.path.dirname(self.model_path)
        # Guard: only call os.makedirs if directory component is non-empty to avoid FileNotFoundError
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"Warning: Failed to persist model to {self.model_path}: {e}")

    def _load_or_create_model(self) -> Any:
        """Load model from model_path or create development fallback model if permitted."""
        # Attempt to load existing pre-trained model from disk
        model = self._load_model()
        if model is not None:
            return model

        # Check if fallback model creation is enabled (useful for separating dev vs production paths)
        if not self._enable_fallback:
            raise RuntimeError(
                f"Failed to load trained model from '{self.model_path}' "
                f"and fallback model creation is disabled."
            )

        print(f"Initializing synthetic fallback RandomForest model for {self.model_path}")
        fallback_model = self._create_fallback_model()
        self._save_model(fallback_model)
        return fallback_model

    def _decode_to_grayscale(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw image bytes into 8-bit grayscale NumPy matrix.

        Raises:
            ValueError: If bytes are empty or cannot be parsed as a valid image.
        """
        if not image_bytes:
            raise ValueError("Unable to decode X-ray scan image bytes: input byte stream is empty.")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("L")
            return np.array(image, dtype=np.uint8)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            # Wrap low-level PIL decoding errors in a predictable domain exception
            raise ValueError("Unable to decode X-ray scan image bytes") from exc

    def _resize(self, img: np.ndarray, size=(224, 224)) -> np.ndarray:
        """Resize image matrix to standard target dimensions."""
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Apply Bilateral Filter to reduce noise while maintaining sharp structural edges."""
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for localized contrast enhancement."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw scan image bytes into a grayscale 224x224 NumPy matrix and apply Bilateral Filter + CLAHE contrast enhancement."""
        # Execute preprocessing pipeline steps sequentially
        img = self._decode_to_grayscale(image_bytes)
        img = self._resize(img)
        img = self._denoise(img)
        img = self._enhance_contrast(img)
        return img

    def _glcm_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract GLCM (Gray-Level Co-occurrence Matrix) texture parameters, handling degenerate NaN edge cases."""
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

        # Sanitize statistical prop values using np.nan_to_num to prevent NaN values from propagating
        return {
            "contrast": float(np.nan_to_num(np.mean(graycoprops(glcm, "contrast")), nan=0.0)),
            "homogeneity": float(np.nan_to_num(np.mean(graycoprops(glcm, "homogeneity")), nan=0.0)),
            "correlation": float(np.nan_to_num(np.mean(graycoprops(glcm, "correlation")), nan=0.0)),
            "energy": float(np.nan_to_num(np.mean(graycoprops(glcm, "energy")), nan=0.0)),
        }

    def _lbp_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract LBP (Local Binary Patterns) micro-structure features."""
        # Radius R=1, Points P=8 for standard 3x3 local neighborhood comparison
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        return {
            "lbp_mean": float(np.nan_to_num(np.mean(lbp), nan=0.0)),
            "lbp_var": float(np.nan_to_num(np.var(lbp), nan=0.0)),
        }

    def _hog_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract HOG (Histogram of Oriented Gradients) shape & contour features."""
        hog_feats = hog(
            img,
            orientations=8,
            pixels_per_cell=(32, 32),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True,
        )
        return {
            "hog_mean": float(np.nan_to_num(np.mean(hog_feats), nan=0.0)),
            "hog_std": float(np.nan_to_num(np.std(hog_feats), nan=0.0)),
        }

    def extract_features(self, scan: XRayScan) -> Dict[str, float]:
        """Extract numerical feature vector (GLCM texture, LBP micro-structures, HOG contours) from X-ray scan."""
        # Preprocess scan bytes to 224x224 enhanced grayscale matrix
        img = self._preprocess_image(scan.image_bytes)

        # Aggregate feature dictionaries across feature subroutines
        features: Dict[str, float] = {}
        features.update(self._glcm_features(img))
        features.update(self._lbp_features(img))
        features.update(self._hog_features(img))
        return features

    def predict(self, scan: XRayScan) -> PredictionResult:
        """Extract features from scan image and predict cancer classification using Scikit-Learn model."""
        # Step 1: Extract feature vector dictionary
        features_dict = self.extract_features(scan)

        # Step 2: Format features into 2D NumPy array matching explicit FEATURE_KEYS order
        feature_vector = np.array([[features_dict[k] for k in FEATURE_KEYS]], dtype=np.float64)

        # Step 3: Execute model prediction
        pred_class = self._model.predict(feature_vector)[0]

        # Step 4: Derive classification label dynamically using model classes or centralized mapping
        if isinstance(pred_class, str):
            prediction_label = pred_class
        elif hasattr(self._model, "classes_") and pred_class in self._model.classes_:
            # If classes_ contains strings or custom labels
            idx = list(self._model.classes_).index(pred_class)
            raw_cls = self._model.classes_[idx]
            if isinstance(raw_cls, str):
                prediction_label = raw_cls
            else:
                prediction_label = CLASS_LABELS.get(int(raw_cls), str(raw_cls))
        else:
            prediction_label = CLASS_LABELS.get(int(pred_class), str(pred_class))

        # Step 5: Evaluate confidence score derived from probability, decision function, or fallback
        if hasattr(self._model, "classes_") and pred_class in self._model.classes_:
            class_idx = list(self._model.classes_).index(pred_class)
        else:
            class_idx = int(pred_class) if isinstance(pred_class, (int, np.integer)) else 0

        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(feature_vector)[0]
            confidence = float(probabilities[class_idx])
        elif hasattr(self._model, "decision_function"):
            # Derive confidence-like score from decision function using logistic sigmoid transformation
            decision_scores = self._model.decision_function(feature_vector)
            raw_score = float(decision_scores[0])
            confidence = float(1.0 / (1.0 + np.exp(-raw_score)))
        else:
            # Configurable fallback for non-probabilistic models
            confidence = self._default_confidence

        # Step 6: Construct and return populated PredictionResult domain entity
        return PredictionResult(
            prediction_label=prediction_label,
            confidence_score=round(confidence, 4),
            extracted_features=features_dict,
            model_type="Traditional Machine Learning",
        )

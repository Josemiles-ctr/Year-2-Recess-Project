import io
import os
from typing import Dict, Any, Optional

import joblib
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

from src.domain.entities import XRayScan, PredictionResult
from src.interfaces.gateways import TraditionalModelGateway

CLASS_LABELS = {
    0: "Atelectasis",
    1: "Cardiomegaly",
    2: "Consolidation",
    3: "Edema",
    4: "Effusion",
    5: "Emphysema",
    6: "Fibrosis",
    7: "Hernia",
    8: "Infiltration",
    9: "Mass",
    10: "No Finding",
    11: "Nodule",
    12: "Pleural_Thickening",
    13: "Pneumonia",
    14: "Pneumothorax",
}

IMG_SIZE = 96
PIXEL_FEATURES = IMG_SIZE * IMG_SIZE
TABULAR_FEATURES = 13
TOTAL_FEATURES = PIXEL_FEATURES + TABULAR_FEATURES


class SklearnTraditionalModel(TraditionalModelGateway):
    def __init__(self, model_path: str = ""):
        if not model_path:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            model_path = os.path.join(root, "models", "nih_chest_xray_rf_model.joblib")
        norm_path = os.path.abspath(model_path)
        if not os.path.isfile(norm_path):
            raise FileNotFoundError(f"Trained model not found at {norm_path}")
        self._model = joblib.load(norm_path)

    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("Empty image bytes")
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("L")
            return np.array(img, dtype=np.uint8)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise ValueError("Unable to decode image") from e

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return img

    def _pixel_features(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32).reshape(-1) / 255.0

    def _tabular_features(self, img: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(img, 100, 200)
        mean_int = float(img.mean())
        std_int = float(img.std())
        edge_density = float((edges > 0).mean())
        arr = np.array([
            mean_int, std_int, edge_density,
            0.0, 0.0,
            0.0, 1.0,
            0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
        ], dtype=np.float32)
        return arr

    def extract_features(self, scan: XRayScan) -> Dict[str, float]:
        img = self._load_image(scan.image_bytes)
        img = self._preprocess(img)
        pixels = self._pixel_features(img)
        tabs = self._tabular_features(img)
        vec = np.concatenate([pixels, tabs])
        return {"feature_vector": vec}

    def predict(self, scan: XRayScan) -> PredictionResult:
        img = self._load_image(scan.image_bytes)
        img = self._preprocess(img)
        pixels = self._pixel_features(img)
        tabs = self._tabular_features(img)
        feature_vector = np.concatenate([pixels, tabs]).reshape(1, -1)

        probs = self._model.predict_proba(feature_vector)
        prob_dict = {}
        for i, cls_name in CLASS_LABELS.items():
            prob_dict[cls_name] = float(probs[i][0][1]) if isinstance(probs[i], np.ndarray) else 0.0

        detected = {c: p for c, p in prob_dict.items() if p >= 0.5 and c != "No Finding"}
        if detected:
            top_label = max(detected, key=detected.get)
            top_conf = detected[top_label]
        else:
            top_label = "No Finding"
            top_conf = prob_dict.get("No Finding", 0.0)

        features_out = {
            "mean_intensity": float(img.mean()),
            "std_intensity": float(img.std()),
            "edge_density": float(cv2.Canny(img, 100, 200).mean() > 0),
        }

        return PredictionResult(
            prediction_label=top_label,
            confidence_score=round(top_conf, 4),
            extracted_features=features_out,
            model_type="Random Forest (NIH Chest X-Ray)",
            per_class_probabilities=prob_dict,
        )
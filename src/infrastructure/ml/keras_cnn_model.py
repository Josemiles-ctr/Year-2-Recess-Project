import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from src.domain.entities import XRayScan, PredictionResult
from src.interfaces.gateways import CnnModelGateway

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def _conv2d(inp: np.ndarray, kernel: np.ndarray, bias: np.ndarray, stride: int = 1) -> np.ndarray:
    b, h, w, c = inp.shape
    kh, kw, _, f = kernel.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    out = np.zeros((b, oh, ow, f), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            h_start, w_start = i * stride, j * stride
            patch = inp[:, h_start:h_start + kh, w_start:w_start + kw, :]
            for k in range(f):
                out[:, i, j, k] = np.sum(patch * kernel[..., k], axis=(1, 2, 3)) + bias[k]
    return out


def _conv2d_same(inp: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape[:2]
    ph = kh // 2
    pw = kw // 2
    padded = np.pad(inp, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode="constant")
    return _conv2d(padded, kernel, bias, stride=1)


def _max_pool2d(inp: np.ndarray, pool_size: int = 2) -> np.ndarray:
    b, h, w, c = inp.shape
    oh = h // pool_size
    ow = w // pool_size
    out = np.zeros((b, oh, ow, c), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            out[:, i, j, :] = np.max(inp[:, i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size, :], axis=(1, 2))
    return out


def _global_avg_pool(inp: np.ndarray) -> np.ndarray:
    return np.mean(inp, axis=(1, 2))


def _batch_norm(inp: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                mean: np.ndarray, var: np.ndarray, eps: float = 0.001) -> np.ndarray:
    return gamma * (inp - mean) / np.sqrt(var + eps) + beta


def _dense(inp: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return inp @ kernel + bias


class KerasCnnModel(CnnModelGateway):
    def __init__(
        self,
        model_path: str = "models/nih_chest_xray_cnn_model.keras",
        img_size: int = 96,
    ):
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(__file__).resolve().parents[3] / model_path
        self.img_size = img_size
        self._weights: Optional[Dict[str, np.ndarray]] = None

    def _load_weights(self) -> Dict[str, np.ndarray]:
        if self._weights is not None:
            return self._weights
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Keras model not found at {self.model_path}. "
                "Train and save the model first using the notebook."
            )
        try:
            import h5py
        except ImportError:
            raise RuntimeError("h5py is required to load Keras model weights. Install it with: pip install h5py")

        with zipfile.ZipFile(self.model_path) as z:
            with z.open("model.weights.h5") as f:
                with h5py.File(f, "r") as h5:
                    w = {}
                    def _walk(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            w[name] = obj[()]
                        elif isinstance(obj, h5py.Group):
                            for key in obj:
                                _walk(f"{name}/{key}" if name else key, obj[key])
                    for key in h5:
                        _walk(key, h5[key])
        self._weights = w
        self.logger.info("Loaded %d weight tensors from %s", len(w), self.model_path)
        return self._weights

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.reshape(1, self.img_size, self.img_size, 1)
        return arr

    def _forward(self, inp: np.ndarray) -> np.ndarray:
        w = self._load_weights()

        x = _relu(_conv2d_same(inp, w["layers/conv2d/vars/0"], w["layers/conv2d/vars/1"]))
        x = _batch_norm(x, w["layers/batch_normalization/vars/0"], w["layers/batch_normalization/vars/1"],
                        w["layers/batch_normalization/vars/2"], w["layers/batch_normalization/vars/3"])

        x = _max_pool2d(x, 2)

        x = _relu(_conv2d_same(x, w["layers/conv2d_1/vars/0"], w["layers/conv2d_1/vars/1"]))
        x = _batch_norm(x, w["layers/batch_normalization_1/vars/0"], w["layers/batch_normalization_1/vars/1"],
                        w["layers/batch_normalization_1/vars/2"], w["layers/batch_normalization_1/vars/3"])

        x = _max_pool2d(x, 2)

        x = _relu(_conv2d_same(x, w["layers/conv2d_2/vars/0"], w["layers/conv2d_2/vars/1"]))
        x = _batch_norm(x, w["layers/batch_normalization_2/vars/0"], w["layers/batch_normalization_2/vars/1"],
                        w["layers/batch_normalization_2/vars/2"], w["layers/batch_normalization_2/vars/3"])

        x = _global_avg_pool(x)

        x = _dense(x, w["layers/dense/vars/0"], w["layers/dense/vars/1"])
        x = _relu(x)

        x = _dense(x, w["layers/dense_1/vars/0"], w["layers/dense_1/vars/1"])
        x = _sigmoid(x)

        return x[0]

    def _thresholds(self) -> Dict[str, float]:
        return {
            "Atelectasis": 0.05, "Cardiomegaly": 0.05, "Consolidation": 0.75,
            "Edema": 0.05, "Effusion": 0.35, "Emphysema": 0.40,
            "Fibrosis": 0.70, "Hernia": 0.05, "Infiltration": 0.45,
            "Mass": 0.35, "No Finding": 0.55, "Nodule": 0.70,
            "Pleural_Thickening": 0.35, "Pneumonia": 0.95, "Pneumothorax": 0.50,
        }

    def predict(self, scan: XRayScan) -> PredictionResult:
        inp = self._preprocess(scan.image_bytes)
        raw = self._forward(inp)

        probs = {CLASSES[i]: float(raw[i]) for i in range(len(CLASSES))}
        thresh = self._thresholds()
        detected = {c: p for c, p in probs.items() if p >= thresh.get(c, 0.5) and c != "No Finding"}
        top_label = max(detected, key=detected.get) if detected else "No Finding"
        top_conf = probs.get(top_label, 0.0)

        return PredictionResult(
            prediction_label=top_label,
            confidence_score=round(top_conf, 4),
            per_class_probabilities=probs,
            model_type="Keras CNN (NIH Chest X-Ray)",
        )

    def generate_grad_cam(self, scan: XRayScan) -> str:
        return ""

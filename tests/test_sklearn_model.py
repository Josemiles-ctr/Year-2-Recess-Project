import os
import io
import unittest
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

from src.domain.entities import XRayScan
from src.infrastructure.ml.sklearn_model import (
    SklearnTraditionalModel,
    FEATURE_KEYS,
    CLASS_LABELS,
)


class TestSklearnTraditionalModel(unittest.TestCase):
    """Comprehensive unit tests for SklearnTraditionalModel implementation."""

    def setUp(self):
        """Create a synthetic 100x100 grayscale image byte stream for testing."""
        # Generate 100x100 grayscale image array with uint8 data type
        img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.valid_image_bytes = buf.getvalue()

        # Uniform image for GLCM degenerate testing
        uniform_img = Image.fromarray(np.ones((100, 100), dtype=np.uint8) * 128)
        buf_u = io.BytesIO()
        uniform_img.save(buf_u, format="PNG")
        self.uniform_image_bytes = buf_u.getvalue()

        self.temp_model_path = "src/models/test_classifier_tmp.pkl"

    def tearDown(self):
        """Clean up temporary test model files."""
        if os.path.exists(self.temp_model_path):
            try:
                os.remove(self.temp_model_path)
            except OSError:
                pass
        stub_path = "stub_classifier.pkl"
        if os.path.exists(stub_path):
            try:
                os.remove(stub_path)
            except OSError:
                pass

    def test_valid_image_feature_extraction_and_prediction(self):
        """Test feature extraction and prediction output for valid image bytes."""
        model_gateway = SklearnTraditionalModel(model_path=self.temp_model_path)
        scan = XRayScan(filename="test_scan.png", image_bytes=self.valid_image_bytes)

        features = model_gateway.extract_features(scan)
        for key in FEATURE_KEYS:
            self.assertIn(key, features)
            self.assertFalse(np.isnan(features[key]))

        result = model_gateway.predict(scan)
        # Assert prediction label is within configured CLASS_LABELS
        self.assertIn(result.prediction_label, list(CLASS_LABELS.values()))
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)

    def test_corrupt_or_empty_image_bytes_raises_value_error(self):
        """Test defensive handling for empty or corrupt image bytes."""
        model_gateway = SklearnTraditionalModel(model_path=self.temp_model_path)

        # Empty bytes
        empty_scan = XRayScan(filename="empty.png", image_bytes=b"")
        with self.assertRaises(ValueError) as ctx:
            model_gateway.extract_features(empty_scan)
        self.assertIn("Unable to decode", str(ctx.exception))

        # Corrupt bytes
        corrupt_scan = XRayScan(filename="corrupt.png", image_bytes=b"NOT_AN_IMAGE_DATA")
        with self.assertRaises(ValueError) as ctx:
            model_gateway.extract_features(corrupt_scan)
        self.assertIn("Unable to decode", str(ctx.exception))

    def test_uniform_image_glcm_nan_sanitization(self):
        """Test degenerate GLCM case does not propagate NaNs into features."""
        model_gateway = SklearnTraditionalModel(model_path=self.temp_model_path)
        scan = XRayScan(filename="uniform.png", image_bytes=self.uniform_image_bytes)

        features = model_gateway.extract_features(scan)
        for key in ["contrast", "homogeneity", "correlation", "energy"]:
            self.assertFalse(np.isnan(features[key]), f"Feature {key} was NaN")

    def test_model_path_without_directory_component(self):
        """Test that model_path with no directory component (e.g. 'stub_classifier.pkl') creates without error."""
        stub_path = "stub_classifier.pkl"
        model_gateway = SklearnTraditionalModel(model_path=stub_path)
        scan = XRayScan(filename="test.png", image_bytes=self.valid_image_bytes)
        result = model_gateway.predict(scan)
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(stub_path))

    def test_enable_fallback_false_raises_runtime_error(self):
        """Test that enable_fallback=False raises RuntimeError when model file is missing."""
        missing_path = "src/models/non_existent_model_12345.pkl"
        with self.assertRaises(RuntimeError):
            SklearnTraditionalModel(model_path=missing_path, enable_fallback=False)

    def test_injected_model(self):
        """Test injecting an external model directly into gateway."""
        clf = LogisticRegression()
        X_dummy = np.random.rand(4, len(FEATURE_KEYS))
        y_dummy = np.array([0, 1, 0, 1])
        clf.fit(X_dummy, y_dummy)

        gateway = SklearnTraditionalModel(model=clf)
        scan = XRayScan(filename="test.png", image_bytes=self.valid_image_bytes)
        result = gateway.predict(scan)
        # Verify injected model prediction label matches configured CLASS_LABELS
        self.assertIn(result.prediction_label, list(CLASS_LABELS.values()))
        self.assertGreaterEqual(result.confidence_score, 0.0)


if __name__ == "__main__":
    unittest.main()

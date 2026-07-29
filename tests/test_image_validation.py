import io
import unittest
from unittest.mock import MagicMock
from PIL import Image
import numpy as np

from src.domain.entities import (
    DiagnosticReport,
    PredictionResult,
    InvalidImageError,
    NotAnXRayError,
)
from src.use_cases.predict import PredictCancerUseCase


class TestImageValidation(unittest.TestCase):
    """Unit test suite for Layer 1 (PIL decoding) and Layer 2 (Gemini Vision) image validation."""

    def setUp(self):
        """Generate valid synthetic PNG image bytes for testing."""
        # Create a synthetic 100x100 grayscale image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.valid_png_bytes = buf.getvalue()

        # Mock gateway dependencies
        self.traditional_gateway = MagicMock()
        self.cnn_gateway = MagicMock()
        self.llm_gateway = MagicMock()

        self.use_case = PredictCancerUseCase(
            traditional_gateway=self.traditional_gateway,
            cnn_gateway=self.cnn_gateway,
            llm_gateway=self.llm_gateway,
        )

    def test_empty_image_bytes_raises_invalid_image_error(self):
        """Verify Layer 1 rejects empty byte payloads immediately."""
        with self.assertRaises(InvalidImageError) as ctx:
            self.use_case.execute("empty.png", b"")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_corrupt_file_bytes_raises_invalid_image_error(self):
        """Verify Layer 1 rejects non-image binary data immediately."""
        corrupt_bytes = b"PDF-1.4 %DOCUMENT_TEXT_HERE"
        with self.assertRaises(InvalidImageError) as ctx:
            self.use_case.execute("doc.pdf", corrupt_bytes)
        self.assertIn("not a valid or readable image", str(ctx.exception).lower())

    def test_layer2_non_chest_xray_raises_not_an_xray_error(self):
        """Verify Layer 2 raises NotAnXRayError when Gemini Vision rejects the scan."""
        # Mock Gemini Vision rejecting the image as a dog photo
        self.llm_gateway.validate_chest_xray.return_value = (
            False,
            "Image appears to be a photo of a dog, not a human chest X-ray radiograph.",
        )

        with self.assertRaises(NotAnXRayError) as ctx:
            self.use_case.execute("dog.jpg", self.valid_png_bytes)

        self.assertIn("Validation failed", str(ctx.exception))
        self.assertIn("photo of a dog", str(ctx.exception))

    def test_valid_chest_xray_passes_validation_and_executes(self):
        """Verify valid chest X-ray passes validation and triggers downstream ML models."""
        # Mock Gemini Vision accepting the image
        self.llm_gateway.validate_chest_xray.return_value = (
            True,
            "Valid thoracic anatomy detected.",
        )

        # Mock model prediction results
        dummy_pred = PredictionResult(
            prediction_label="No Finding",
            confidence_score=0.95,
            model_type="Test Model",
        )
        self.traditional_gateway.predict.return_value = dummy_pred
        self.cnn_gateway.predict.return_value = dummy_pred
        self.llm_gateway.generate_report_narrative.return_value = ("<p>Normal</p>", "Normal Scan")

        report = self.use_case.execute("patient_xray.png", self.valid_png_bytes)

        self.assertIsInstance(report, DiagnosticReport)
        self.assertEqual(report.session_title, "Normal Scan")
        self.traditional_gateway.predict.assert_called_once()
        self.cnn_gateway.predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()

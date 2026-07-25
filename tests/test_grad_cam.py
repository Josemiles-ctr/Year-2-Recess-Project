"""
Test suite for Grad-CAM implementation.

This module provides unit tests and integration tests for the PyTorchCnnModel
and its Grad-CAM heatmap generation capability.
"""

import os
import io
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from src.domain.entities import XRayScan
from src.infrastructure.ml.pytorch_model import PyTorchCnnModel, CustomCnnArchitecture


def create_sample_xray_bytes(width: int = 224, height: int = 224) -> bytes:
    """Create a sample grayscale X-ray image for testing.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Raw image bytes (PNG format)
    """
    # Create a simple grayscale image with some features
    image_array = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    
    # Add a brighter region (simulating a tumor area)
    image_array[80:130, 80:130] = np.random.randint(150, 255, (50, 50), dtype=np.uint8)
    
    # Convert to PIL Image
    image = Image.fromarray(image_array, mode='L')
    
    # Save to bytes
    bytes_buffer = io.BytesIO()
    image.save(bytes_buffer, format='PNG')
    bytes_buffer.seek(0)
    
    return bytes_buffer.getvalue()


class TestCustomCnnArchitecture(unittest.TestCase):
    """Test suite for CustomCnnArchitecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CustomCnnArchitecture(num_classes=2)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        self.assertIsNotNone(self.model)
        self.assertEqual(len(list(self.model.parameters())), 20)  # Conv + BN + FC layers
    
    def test_forward_pass_output_shape(self):
        """Test forward pass output shape."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 224, 224, device=self.device)
        output = self.model(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, 2))
    
    def test_forward_pass_output_range(self):
        """Test that forward pass produces reasonable logits."""
        input_tensor = torch.randn(1, 1, 224, 224, device=self.device)
        output = self.model(input_tensor)
        
        # Output should be finite (no NaN or Inf)
        self.assertTrue(torch.isfinite(output).all())
    
    def test_model_eval_mode(self):
        """Test that model switches to eval mode correctly."""
        self.model.train()
        self.assertTrue(self.model.training)
        
        self.model.eval()
        self.assertFalse(self.model.training)
    
    def test_conv4_layer_exists(self):
        """Test that conv4 layer (Grad-CAM target) exists."""
        self.assertTrue(hasattr(self.model, 'conv4'))
        self.assertIsInstance(self.model.conv4, torch.nn.Conv2d)


class TestPyTorchCnnModelInitialization(unittest.TestCase):
    """Test suite for PyTorchCnnModel initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_without_weights(self):
        """Test model initialization when weights file doesn't exist."""
        model = PyTorchCnnModel(
            model_path=self.model_path,
            temp_dir=self.temp_dir
        )
        
        self.assertIsNotNone(model.model)
        self.assertEqual(model.image_size, (224, 224))
    
    def test_temp_directory_creation(self):
        """Test that temp directory is created if it doesn't exist."""
        new_temp_dir = os.path.join(self.temp_dir, "nested", "temp")
        _ = PyTorchCnnModel(temp_dir=new_temp_dir)
        
        self.assertTrue(os.path.exists(new_temp_dir))
    
    def test_device_detection(self):
        """Test that device is correctly detected."""
        model = PyTorchCnnModel(temp_dir=self.temp_dir)
        
        if torch.cuda.is_available():
            self.assertIn(model.device, ['cuda', 'cuda:0'])
        else:
            self.assertEqual(model.device, 'cpu')


class TestPreprocessing(unittest.TestCase):
    """Test suite for image preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = PyTorchCnnModel(temp_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_preprocess_image_output_shape(self):
        """Test preprocessing output tensor shape."""
        image_bytes = create_sample_xray_bytes()
        tensor, original_size = self.model._preprocess_image(image_bytes)
        
        self.assertEqual(tensor.shape, (1, 1, 224, 224))
        self.assertEqual(len(original_size), 2)  # (width, height)
    
    def test_preprocess_image_normalization(self):
        """Test that preprocessing normalizes values correctly."""
        image_bytes = create_sample_xray_bytes()
        tensor, _ = self.model._preprocess_image(image_bytes)
        
        # Normalized tensor should have reasonable range (around -2 to +2 after normalization)
        self.assertTrue(tensor.min() < 0)  # Should have negative values after normalization
        self.assertTrue(tensor.max() > 0)
    
    def test_preprocess_grayscale_conversion(self):
        """Test preprocessing with grayscale image."""
        # Create RGB image
        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        rgb_pil = Image.fromarray(rgb_image, mode='RGB')
        
        bytes_buffer = io.BytesIO()
        rgb_pil.save(bytes_buffer, format='PNG')
        bytes_buffer.seek(0)
        
        tensor, _ = self.model._preprocess_image(bytes_buffer.getvalue())
        self.assertEqual(tensor.shape, (1, 1, 224, 224))


class TestPrediction(unittest.TestCase):
    """Test suite for model prediction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = PyTorchCnnModel(temp_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_predict_returns_prediction_result(self):
        """Test that predict returns a valid PredictionResult."""
        image_bytes = create_sample_xray_bytes()
        scan = XRayScan(filename="test.png", image_bytes=image_bytes)
        
        result = self.model.predict(scan)
        
        self.assertIsNotNone(result)
        self.assertIn(result.prediction_label, ["Benign", "Malignant"])
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
    
    def test_predict_confidence_range(self):
        """Test that confidence scores are in valid range."""
        for _ in range(5):
            image_bytes = create_sample_xray_bytes()
            scan = XRayScan(filename="test.png", image_bytes=image_bytes)
            
            result = self.model.predict(scan)
            
            self.assertGreaterEqual(result.confidence_score, 0.0)
            self.assertLessEqual(result.confidence_score, 1.0)
    
    def test_predict_includes_model_type(self):
        """Test that prediction result includes model type."""
        image_bytes = create_sample_xray_bytes()
        scan = XRayScan(filename="test.png", image_bytes=image_bytes)
        
        result = self.model.predict(scan)
        
        self.assertEqual(result.model_type, "PyTorch CNN")


class TestGradCam(unittest.TestCase):
    """Test suite for Grad-CAM heatmap generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = PyTorchCnnModel(temp_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_grad_cam_returns_path(self):
        """Test that generate_grad_cam returns a file path."""
        image_bytes = create_sample_xray_bytes()
        scan = XRayScan(filename="test.png", image_bytes=image_bytes)
        
        grad_cam_path = self.model.generate_grad_cam(scan)
        
        self.assertIsNotNone(grad_cam_path)
        self.assertTrue(grad_cam_path.startswith("/static/temp/"))
    
    def test_grad_cam_creates_file(self):
        """Test that Grad-CAM actually creates a file."""
        image_bytes = create_sample_xray_bytes()
        scan = XRayScan(filename="test.png", image_bytes=image_bytes)
        
        grad_cam_path = self.model.generate_grad_cam(scan)
        
        if grad_cam_path:
            # Convert web path to file path
            file_path = os.path.join("src", grad_cam_path.lstrip("/"))
            file_path_abs = os.path.join(os.getcwd(), file_path)
            
            # File should exist
            self.assertTrue(os.path.exists(file_path_abs) or grad_cam_path == "")
    
    def test_grad_cam_multiple_calls_different_files(self):
        """Test that multiple Grad-CAM calls generate different files."""
        image_bytes = create_sample_xray_bytes()
        scan = XRayScan(filename="test.png", image_bytes=image_bytes)
        
        grad_cam_path1 = self.model.generate_grad_cam(scan)
        grad_cam_path2 = self.model.generate_grad_cam(scan)
        
        # Paths should be different due to UUID
        if grad_cam_path1 and grad_cam_path2:
            self.assertNotEqual(grad_cam_path1, grad_cam_path2)
    
    def test_grad_cam_path_format(self):
        """Test that Grad-CAM path has correct format."""
        image_bytes = create_sample_xray_bytes()
        scan = XRayScan(filename="test.png", image_bytes=image_bytes)
        
        grad_cam_path = self.model.generate_grad_cam(scan)
        
        if grad_cam_path:
            self.assertTrue(grad_cam_path.startswith("/static/temp/"))
            self.assertTrue(grad_cam_path.endswith(".png"))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = PyTorchCnnModel(temp_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_prediction_pipeline(self):
        """Test complete prediction pipeline with Grad-CAM."""
        image_bytes = create_sample_xray_bytes()
        scan = XRayScan(filename="test_case.png", image_bytes=image_bytes)
        
        result = self.model.predict(scan)
        
        # Verify all fields are populated
        self.assertIsNotNone(result.prediction_label)
        self.assertIsNotNone(result.confidence_score)
        self.assertIsNotNone(result.model_type)
        self.assertIn(result.prediction_label, ["Benign", "Malignant"])
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
    
    def test_predictions_deterministic_for_same_image(self):
        """Test that model produces same prediction for same image."""
        image_bytes = create_sample_xray_bytes()
        scan = XRayScan(filename="test.png", image_bytes=image_bytes)
        
        self.model.model.eval()
        with torch.no_grad():
            result1 = self.model.predict(scan)
            result2 = self.model.predict(scan)
        
        # Predictions should be identical for same image in eval mode
        self.assertEqual(result1.prediction_label, result2.prediction_label)
        self.assertAlmostEqual(result1.confidence_score, result2.confidence_score, places=5)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = PyTorchCnnModel(temp_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_invalid_image_bytes(self):
        """Test handling of invalid image bytes."""
        invalid_bytes = b"This is not an image"
        scan = XRayScan(filename="invalid.png", image_bytes=invalid_bytes)
        
        with self.assertRaises(Exception):
            self.model.predict(scan)
    
    def test_empty_image_bytes(self):
        """Test handling of empty image bytes."""
        scan = XRayScan(filename="empty.png", image_bytes=b"")
        
        with self.assertRaises(Exception):
            self.model.predict(scan)


def run_performance_test():
    """Run performance benchmarks for Grad-CAM."""
    import time
    
    print("\n" + "="*60)
    print("GRAD-CAM PERFORMANCE BENCHMARK")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    model = PyTorchCnnModel(temp_dir=temp_dir)
    model.model.eval()
    
    num_runs = 5
    times = []
    
    for i in range(num_runs):
        image_bytes = create_sample_xray_bytes()
        scan = XRayScan(filename=f"test_{i}.png", image_bytes=image_bytes)
        
        start_time = time.time()
        result = model.predict(scan)
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        print(f"Run {i+1}: {elapsed*1000:.2f}ms - {result.prediction_label} ({result.confidence_score:.1%})")
    
    print(f"\nAverage time: {np.mean(times)*1000:.2f}ms")
    print(f"Std deviation: {np.std(times)*1000:.2f}ms")
    print(f"Min: {np.min(times)*1000:.2f}ms")
    print(f"Max: {np.max(times)*1000:.2f}ms")
    
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    if np.mean(times) < 0.025:  # 25ms target
        print("✅ Performance is within acceptable range (<25ms)")
    else:
        print(f"⚠️ Performance may need optimization (avg: {np.mean(times)*1000:.2f}ms)")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmark
    run_performance_test()

import os
import io
import uuid
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

from src.domain.entities import XRayScan, PredictionResult
from src.interfaces.gateways import CnnModelGateway

logger = logging.getLogger(__name__)


class CustomCnnArchitecture(nn.Module):
    """Custom CNN architecture for X-ray classification with Grad-CAM support.

    Architecture:
    - Input: 1 channel (grayscale), 224x224
    - Conv blocks with batch norm and dropout
    - Global average pooling
    - Final classification layer (2 classes: Benign/Malignant)
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__()

        # First convolutional block: Conv -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fourth convolutional block (final conv layer for Grad-CAM)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class PyTorchCnnModel(CnnModelGateway):
    """Approach 2: Deep Learning CNN (PyTorch) with Grad-CAM visualization.

    Implements a custom CNN architecture for X-ray classification with Grad-CAM
    to generate heatmap visualizations highlighting regions influential in predictions.
    """

    def __init__(
        self,
        model_path: str = "src/models/cnn_model.pth",
        temp_dir: str = "src/static/temp",
        device: Optional[str] = None,
    ):
        """Initialize the PyTorch CNN model.

        Args:
            model_path: Path to saved model weights
            temp_dir: Directory to store generated heatmap images
            device: Torch device (auto-detects GPU if available)
        """
        self.model_path = model_path
        self.temp_dir = temp_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Create temp directory if it doesn't exist
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = CustomCnnArchitecture(num_classes=2)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load model weights if available
        self._load_model_weights()

        # Image preprocessing constants
        self.image_size = (224, 224)
        self.mean = 0.485
        self.std = 0.229

        # Grad-CAM hooks
        self.activation = None
        self.gradient = None

    def _load_model_weights(self) -> None:
        """Load model weights from disk if available."""
        if os.path.isfile(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded model weights from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model weights: {e}. Using initialized model.")
        else:
            logger.warning(
                f"Model weights not found at {self.model_path}. Using initialized model."
            )

    def _preprocess_image(self, image_bytes: bytes) -> tuple[torch.Tensor, tuple[int, int]]:
        """Preprocess X-ray scan image.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Tuple of (preprocessed tensor, original image size)
        """
        # Decode image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Store original dimensions for Grad-CAM upsampling
        original_size = image.size  # (width, height)

        # Convert to grayscale if not already
        if image.mode != "L":
            image = image.convert("L")

        # Resize to model input size
        image_resized = image.resize(self.image_size, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image_resized, dtype=np.float32) / 255.0

        # Apply normalization
        image_normalized = (image_array - self.mean) / self.std

        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        return tensor, original_size

    def _register_grad_cam_hooks(self) -> None:
        """Register forward and backward hooks for Grad-CAM on final conv layer."""

        # Hook into the final convolutional layer (conv4)
        def forward_hook(module, input, output):
            self.activation = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()

        self.model.conv4.register_forward_hook(forward_hook)
        self.model.conv4.register_backward_hook(backward_hook)

    def _generate_grad_cam_heatmap(
        self, image_tensor: torch.Tensor, image_bytes: bytes, original_size: tuple[int, int]
    ) -> str:
        """Generate Grad-CAM heatmap and save to disk.

        Args:
            image_tensor: Preprocessed image tensor
            image_bytes: Original image bytes for overlay
            original_size: Original image dimensions (width, height)

        Returns:
            Web URL path to the generated heatmap image
        """
        self.model.eval()

        # Register hooks
        self._register_grad_cam_hooks()

        with torch.set_grad_enabled(True):
            image_tensor_clone = image_tensor.clone().detach().requires_grad_(True)

            # Forward pass
            output = self.model(image_tensor_clone)

            # Get predicted class
            predicted_class = output.argmax(dim=1).item()

            # Compute loss for target class
            loss = output[0, predicted_class]

            # Backward pass to compute gradients
            loss.backward()

        # Compute Grad-CAM
        if self.activation is not None and self.gradient is not None:
            # Get gradients of activation
            gradients = self.gradient[0]  # [C, H, W]
            activations = self.activation[0]  # [C, H, W]

            # Compute channel-wise gradient weights (average across spatial dimensions)
            weights = gradients.mean(dim=(1, 2))  # [C]

            # Weighted sum of activations
            grad_cam = torch.zeros(activations.shape[1:], device=self.device)
            for i in range(len(weights)):
                grad_cam += weights[i] * activations[i]

            # Apply ReLU to keep only positive influences
            grad_cam = F.relu(grad_cam)

            # Normalize to [0, 1]
            if grad_cam.max() > 0:
                grad_cam = grad_cam / grad_cam.max()

            # Convert to numpy
            grad_cam_np = grad_cam.cpu().numpy()

            # Upsample to original image size
            # grad_cam_np is currently [14, 14] (from 224x224 with 4 pooling operations)
            grad_cam_resized = cv2.resize(grad_cam_np, (original_size[0], original_size[1]))

            # Decode original image for overlay
            original_image = Image.open(io.BytesIO(image_bytes))
            if original_image.mode != "L":
                original_image = original_image.convert("L")
            original_image = original_image.resize(original_size, Image.Resampling.LANCZOS)
            original_array = np.array(original_image, dtype=np.uint8)

            # Create color overlay using Jet colormap
            grad_cam_colored = cv2.applyColorMap(
                (grad_cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )

            # Convert original to BGR for overlay
            original_bgr = cv2.cvtColor(original_array, cv2.COLOR_GRAY2BGR)

            # Blend: 50% original image + 50% heatmap
            overlay = cv2.addWeighted(original_bgr, 0.5, grad_cam_colored, 0.5, 0)

            # Save to temp directory
            filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(self.temp_dir, filename)

            # Convert BGR to RGB for PIL
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(overlay_rgb)
            result_image.save(filepath)

            # Return web-accessible URL path
            return f"/static/temp/{filename}"

        logger.warning("Grad-CAM activation/gradient not captured. Returning empty path.")
        return ""

    def predict(self, scan: XRayScan) -> PredictionResult:
        """Predict cancer classification and generate Grad-CAM heatmap.

        Args:
            scan: XRayScan entity containing image bytes and metadata

        Returns:
            PredictionResult with prediction, confidence, and Grad-CAM path
        """
        try:
            # Preprocess image
            image_tensor, original_size = self._preprocess_image(scan.image_bytes)

            # Generate Grad-CAM heatmap
            grad_cam_path = self._generate_grad_cam_heatmap(
                image_tensor, scan.image_bytes, original_size
            )

            # Get prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = output.argmax(dim=1).item()
                confidence = probabilities[0, predicted_class].item()

            # Map class to label
            class_labels = {0: "Benign", 1: "Malignant"}
            prediction_label = class_labels.get(predicted_class, "Unknown")

            return PredictionResult(
                prediction_label=prediction_label,
                confidence_score=confidence,
                grad_cam_path=grad_cam_path,
                model_type="PyTorch CNN",
            )
        except Exception as e:
            logger.error(f"Error during CNN prediction: {e}")
            raise

    def generate_grad_cam(self, scan: XRayScan) -> str:
        """Generate Grad-CAM visualization and return file path.

        Args:
            scan: XRayScan entity containing image bytes

        Returns:
            Web-accessible URL path to the Grad-CAM visualization
        """
        try:
            # Preprocess image
            image_tensor, original_size = self._preprocess_image(scan.image_bytes)

            # Generate and save Grad-CAM heatmap
            grad_cam_path = self._generate_grad_cam_heatmap(
                image_tensor, scan.image_bytes, original_size
            )

            return grad_cam_path
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            return ""

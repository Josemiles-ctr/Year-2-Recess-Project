from src.domain.entities import XRayScan, PredictionResult
from src.interfaces.gateways import CnnModelGateway

# TODO: Define your CNN neural network architecture using PyTorch (nn.Module).
# Ensure it contains Convolutional layers, Max Pooling, and Dense classifiers.


class PyTorchCnnModel(CnnModelGateway):
    """Approach 2: Deep Learning CNN (PyTorch).

    Assignee Guidelines:
    1. Implement a deep learning architecture using PyTorch.
    2. Define normalization, sizing transformations, and tensor pipelines.
    3. Run forward passes to evaluate malignant vs benign probabilities.
    4. Implement Grad-CAM to highlight spatial feature weights on the scan.
    """

    def __init__(
        self, model_path: str = "src/models/cnn_model.pth", temp_dir: str = "src/static/temp"
    ):
        self.model_path = model_path
        self.temp_dir = temp_dir
        # TODO: Initialize transformations, target device (CPU/GPU), and prepare placeholders.

    def predict(self, scan: XRayScan) -> PredictionResult:
        """Task Assignee Implementation steps:
        1. Decode scan.image_bytes using PIL (Pillow).
        2. Apply transformations: convert to Grayscale, resize to 224x224, convert to Tensor, normalize.
        3. Load the model state dictionary from self.model_path.
        4. Run the tensor through the model to obtain predictions and softmax probability confidence scores.
        5. Invoke self.generate_grad_cam(scan) to obtain a heatmap overlay URL.
        6. Return a PredictionResult populated with labels, confidence scores, and grad_cam_path.
        """
        # TODO: Implement CNN prediction logic.
        raise NotImplementedError("PyTorch CNN Predict is not implemented yet.")

    def generate_grad_cam(self, scan: XRayScan) -> str:
        """Task Assignee Implementation steps:
        1. Hook into the gradients of the last convolutional layer.
        2. Perform a forward pass, compute loss gradients back to the target layer activations.
        3. Weight the activations by their gradients and compute a spatial average map.
        4. Resize the map back to the original image dimensions, apply Jet colormap overlays.
        5. Save the generated image file inside self.temp_dir.
        6. Return the local web URL for the image (e.g. /static/temp/gradcam_filename.png).
        """
        # TODO: Implement Grad-CAM heatmap generation.
        raise NotImplementedError("PyTorch CNN Grad-CAM generation is not implemented yet.")

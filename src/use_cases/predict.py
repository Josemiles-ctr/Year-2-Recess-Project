import io
from PIL import Image

from src.domain.entities import DiagnosticReport, XRayScan, InvalidImageError, NotAnXRayError
from src.interfaces.gateways import TraditionalModelGateway, CnnModelGateway, LlmServiceGateway


class PredictCancerUseCase:
    """Usecase to coordinate Dual Model predictions and generate final clinical report.

    Assignee Guidelines:
    1. Receive raw uploads (filename and image bytes).
    2. Validate file format (Layer 1) and anatomical validity (Layer 2 via Gemini Vision).
    3. Coordinate predictions between both gateways (Traditional ML and CNN).
    4. Call the LLM gateway to compile diagnostic text report summaries.
    5. Construct and return a unified DiagnosticReport entity.
    """

    def __init__(
        self,
        traditional_gateway: TraditionalModelGateway,
        cnn_gateway: CnnModelGateway,
        llm_gateway: LlmServiceGateway,
    ):
        self.traditional_gateway = traditional_gateway
        self.cnn_gateway = cnn_gateway
        self.llm_gateway = llm_gateway

    def _validate_image(self, filename: str, image_bytes: bytes) -> None:
        """Helper method to execute Layer 1 (PIL decoding) and Layer 2 (Gemini Vision) checks.

        Raises:
            InvalidImageError: If the file is not a valid, decodable image.
            NotAnXRayError: If Gemini Vision confirms the image is not a chest X-ray radiograph.
        """
        if not filename or not filename.strip():
            raise InvalidImageError("A filename is required.")
        if not image_bytes:
            raise InvalidImageError("The uploaded file is empty.")

        # --- Layer 1: Syntactic & Format Validation (PIL Local Verification) ---
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()  # Verify image headers and format structure
            # Re-open image as verify() clears internal state
            with Image.open(io.BytesIO(image_bytes)) as img:
                valid_formats = {"PNG", "JPEG", "JPG", "WEBP", "TIFF", "BMP", "MPO"}
                if img.format not in valid_formats:
                    raise InvalidImageError(
                        f"Unsupported image format '{img.format}'. Please upload PNG or JPEG images."
                    )
        except Exception as e:
            if isinstance(e, InvalidImageError):
                raise
            raise InvalidImageError("Uploaded file is not a valid or readable image format.") from e

        # --- Layer 2: Anatomical & Domain Validation (Gemini Vision Pre-check) ---
        is_xray, reason = self.llm_gateway.validate_chest_xray(image_bytes, filename)
        if not is_xray:
            raise NotAnXRayError(f"Validation failed: {reason}")

    def execute(self, filename: str, image_bytes: bytes) -> DiagnosticReport:
        """Task Assignee Implementation steps:
        1. Validate file payload using Layer 1 (PIL) and Layer 2 (Gemini Vision).
        2. Instantiate an XRayScan entity using filename and image_bytes.
        3. Call traditional_gateway.predict(scan) to run traditional pipeline.
        4. Call cnn_gateway.predict(scan) to run Deep CNN pipeline.
        5. Call llm_gateway.generate_report_narrative() to get diagnostic summaries.
        6. Compile and return a DiagnosticReport entity.
        """
        # Execute multi-layered validation prior to model predictions
        self._validate_image(filename, image_bytes)

        scan = XRayScan(filename=filename, image_bytes=image_bytes)
        traditional_result = self.traditional_gateway.predict(scan)
        cnn_result = self.cnn_gateway.predict(scan)
        narrative, title = self.llm_gateway.generate_report_narrative(traditional_result, cnn_result)

        return DiagnosticReport(
            scan_details=scan,
            traditional_prediction=traditional_result,
            cnn_prediction=cnn_result,
            llm_narrative=narrative,
            session_title=title,
        )


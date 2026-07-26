from src.domain.entities import DiagnosticReport, XRayScan
from src.interfaces.gateways import TraditionalModelGateway, CnnModelGateway, LlmServiceGateway


class PredictCancerUseCase:
    """Usecase to coordinate Dual Model predictions and generate final clinical report.

    Assignee Guidelines:
    1. Receive raw uploads (filename and image bytes).
    2. Coordinate predictions between both gateways (Traditional ML and CNN).
    3. Call the LLM gateway to compile diagnostic text report summaries.
    4. Construct and return a unified DiagnosticReport entity.
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

    def execute(self, filename: str, image_bytes: bytes) -> DiagnosticReport:
        """Task Assignee Implementation steps:
        1. Instantiate an XRayScan entity using filename and image_bytes.
        2. Call traditional_gateway.predict(scan) to run traditional pipeline.
        3. Call cnn_gateway.predict(scan) to run Deep CNN pipeline.
        4. Develop a resolution logic for cases where model outcomes disagree.
        5. Map findings to a clinical risk tier (e.g., High, Medium, Low).
        6. Call llm_gateway.generate_report_narrative() to get the diagnostic summaries.
        7. Compile and return a DiagnosticReport entity.
        """
        if not filename or not filename.strip():
            raise ValueError("A filename is required.")
        if not image_bytes:
            raise ValueError("The uploaded image is empty.")

        scan = XRayScan(filename=filename, image_bytes=image_bytes)
        traditional_result = self.traditional_gateway.predict(scan)
        cnn_result = self.cnn_gateway.predict(scan)
        narrative = self.llm_gateway.generate_report_narrative(traditional_result, cnn_result)

        return DiagnosticReport(
            scan_details=scan,
            traditional_prediction=traditional_result,
            cnn_prediction=cnn_result,
            llm_narrative=narrative,
        )

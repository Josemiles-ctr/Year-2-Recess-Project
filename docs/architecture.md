# System Architecture & Clean Architecture Specification

This document details the architectural layout of the **AI-Powered Cancer Predictor & LLM Diagnostic Assistant**. We employ Uncle Bob's **Clean Architecture** (also known as Onion Architecture) to decouple our core business rules from external frameworks (Flask), database engines, deep learning backends, and LLM API clients.

---

## Architecture Overview

The system is organized into concentric circles representing different levels of software abstraction. Dependencies point only **inwards**; the core business domain has no knowledge of Flask, web routers, or specific ML model library weights.

```text
    ▲  [External Frameworks & Drivers]  (Flask, PyTorch, Gemini API, HTTP Client)
    │           │
    │  [Interface Adapters]             (Controllers, Model Gateways, LLM Presenters)
    │           │
    │  [Use Cases]                      (PredictCancer, GenerateDiagnosticReport)
    │           │
    ▼  [Domain Entities]                (XRayScan, PredictionResult, DiagnosticConversation)
```

### 1. Domain Layer (src/domain/)
The innermost circle containing the enterprise business entities. These are plain Python classes containing data and basic validations. They represent the core data abstractions.
- **XRayScan**: Holds the binary image data, metadata (patient ID, capture date, scan dimensions), and pre-extracted features.
- **PredictionResult**: Encapsulates the output of both prediction models, including predictions (benign vs malignant), probability confidence scores, and extracted biological features (texture, nodule counts).
- **DiagnosticConversation**: Holds the chat history between the user and the AI assistant, ensuring contextual coherence.

### 2. Use Cases Layer (src/use_cases/)
Contains application-specific business rules. Use cases coordinate the flow of data to and from entities, and direct those entities to use their critical business rules to achieve the goals of the use case.
- **PredictCancerUseCase**: Receives an raw uploaded scan, initiates both model predictions (Traditional ML and CNN), performs comparisons, and produces a consolidated diagnostic output.
- **GenerateDiagnosticReportUseCase**: Takes a PredictionResult and coordinates with an LLM service to synthesize a detailed medical narrative report.
- **ChatWithAssistantUseCase**: Handles conversational follow-ups by forwarding the diagnosis context alongside user queries to the LLM interface.

### 3. Interface Adapters (src/interfaces/)
Translates data between the format most convenient for use cases and the format most convenient for external systems.
- **Model Gateways**: Abstract interfaces defining how use cases trigger models (e.g., TraditionalModelGateway, CnnModelGateway).
- **LLM Gateway**: Abstract interface defining how to interact with Large Language Models (e.g., LlmServiceGateway).
- **Controllers**: Adapts input from the Flask routes (HTTP requests) into Use Case input data structures, and converts Use Case output back into JSON or rendered HTML templates.

### 4. Infrastructure & Frameworks Layer (src/infrastructure/)
The outermost circle, composed of frameworks and tools such as Flask, database adapters, concrete machine learning weights, and actual HTTP clients calling LLM APIs.
- **Flask Web Server**: Registers HTTP routes, handles file uploads, processes cookies/sessions, and renders HTML templates.
- **PyTorch/Scikit-Learn Predictors**: Concrete classes implementing the CnnModelGateway and TraditionalModelGateway boundaries.
- **External LLM Client**: Concrete class implementing the LlmServiceGateway that constructs API payloads, invokes external APIs, and processes responses.

---

## Data Flow (Upload & Diagnosis)

1. **Request Entry**: A user uploads a chest X-ray scan through the Web UI. The request hits the Flask route `/analyze`.
2. **Controller Adaptation**: Flask route extracts the file and passes it to the `AnalyzeController`.
3. **Use Case Trigger**: The controller calls `PredictCancerUseCase.execute()`, passing the raw bytes.
4. **Feature & Neural Inference**:
   - The usecase invokes `TraditionalModelGateway.predict()`. Under the hood, this extracts HOG, LBP, and texture parameters.
   - The usecase invokes `CnnModelGateway.predict()`. Under the hood, this passes the image tensor through the CNN layers.
5. **Entity Creation**: The predictions are combined to construct a `PredictionResult` domain entity.
6. **LLM Context Synthesis**: The `PredictionResult` is passed to the `GenerateDiagnosticReportUseCase`, which prepares the context prompt, calls the `LlmServiceGateway`, and appends the detailed response.
7. **Response Formulating**: The controller receives the domain entities, transforms them into JSON/HTML format, and sends them back to the user's browser.

---

## Key Design Patterns Used

- **Dependency Injection**: Dependencies are injected into use cases via constructors (e.g., passing concrete ML model implementations to the `PredictCancerUseCase`), which permits simple unit testing by mocking gateways.
- **Boundary Interfaces**: Defined as Python abstract base classes (`abc.ABC`) to establish boundaries between layers, enforcing the dependency inversion principle.
- **Repository/Gateway Pattern**: Isolates ML inference libraries from the business core. If the CNN moves backends, only the infrastructure wrapper needs to be updated.

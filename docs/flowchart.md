# Interface and Remote Job Flow

Solid nodes describe the current FastAPI/Modal job and probe architecture.
The canonical manifest gate is required by PLAN Phase 2 and is not yet
implemented end to end. Until that gate exists, the current worker can still
discover a legacy baseline by directory pattern.

```mermaid
flowchart TD
    B["Browser<br/>Jinja HTML + HTMX"] -->|"POST /unlearn<br/>class + method + bounded steps"| F["FastAPI web app"]
    F --> J["Job manager<br/>memory state + persisted results"]
    J --> C{"Candidate record/path<br/>already exists?"}
    C -->|"Yes"| P["Return persisted result"]
    C -->|"No"| R["Modal job runner"]

    R -.->|"Phase 2 required"| I["Verify canonical baseline<br/>id + checkpoint hash + manifest"]
    I -.-> R

    R -->|"HMAC-signed allowlisted spec"| E["Modal HTTP endpoint"]
    E -->|"Validate + spawn + return call id"| G["Detached GPU unlearning call"]
    G --> D["Prepared CIFAR-100 data<br/>and target splits"]
    G --> H["Prepared CLIP cache"]
    G --> A["Baseline adapter artifact"]
    G --> U["Run requested method and steps"]

    R -->|"Poll signed call id"| E
    U -->|"Complete adapter state<br/>as safetensors + metrics"| E
    E -->|"Terminal result"| R
    R --> V["Verify submitted identity,<br/>config, complete keys, shape, dtype"]
    V --> S["Persist candidate + job_result.json"]
    S --> J
    J -->|"Polled status + candidate id"| F
    F --> B

    B -->|"POST /probe<br/>candidate id + uploaded image"| Q["Probe service"]
    Q -.->|"Phase 2 required"| I
    Q --> M["One resident frozen CLIP<br/>and cached text prototypes"]
    M --> X["Baseline prediction"]
    M --> Y["Atomically activate candidate adapter"]
    Y --> Z["Candidate prediction"]
    Z --> RB["Restore loaded baseline adapter"]
    RB --> O["Verdict + raw top-5<br/>+ available recorded metrics"]
    O --> B
```

Clipboard paste is planned for Phase 5. It must feed the same bounded image
validation path as the existing upload route; it does not create a second
probe backend.

In the legacy flow, a matching path is not proof of baseline compatibility.
Until manifest enforcement is implemented, operators must explicitly align the
interface baseline checkpoint with the worker's uploaded checkpoint.

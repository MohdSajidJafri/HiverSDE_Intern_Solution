# Initial Repository Assessment

## Executive Summary
This document provides the baseline inspection of the repository environment prior to executing any code or pipeline implementations.

## 1. Current State
- **Workspace Directory**: `d:\HiverSolution`
- **Initial Contents**: Empty directory. No pre-existing legacy scripts, tests, or configuration files were present.
- **Git Status**: Clean git directory initialized or ready to be initialized.

## 2. Tooling and Runtime Environment
- **Operating System**: Microsoft Windows 11 / Windows Server environment (PowerShell shell).
- **Available Python Interpreters**:
  - Python 3.10.0 (`C:\Program Files\Python310\python.exe`)
  - Python 3.13 (`C:\Program Files\Python313\python.exe`)
- **Pip Version**: pip 21.2.3 (Python 3.10) / pip 25.1.1 (Python 3.13)
- **Pre-installed Python Packages (Python 3.10)**:
  - `pandas` 2.3.3
  - `numpy` 2.2.6
  - `scipy` 1.15.3
  - `pydantic` 2.13.5
  - `fastapi` 0.141.1
  - `uvicorn` 0.52.4
  - `httpx` 0.28.1
- **Tooling Checks**:
  - `git`: version 2.47.1.windows.1 installed and operational.
  - `uv`: Not pre-installed globally; standard Python `venv` + `pip` will serve as the reliable environment manager.
- **Available Storage**: Drive D has **182.79 GB free**, ample for local caching, vector embeddings, and evaluation reports.

## 3. Dataset Inspection & Access Verification
- **Target Dataset**: Customer Support on Twitter (`twcs.csv`), ~3M records, ~516 MB raw.
- **Remote Host**: Verified accessible via Hugging Face Git LFS endpoint (`https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv`).
- **Network Access**: HTTP range streaming tested successfully (HTTP status 206 Partial Content verified). Streaming chunks allows efficient profiling without requiring full in-memory downloads.

## 4. Key Constraints
1. **Windows Pathing and Encoding**:
   - Console stdout must explicitly handle UTF-8 to prevent `UnicodeEncodeError` when handling international characters or emojis from Twitter.
   - Forward slashes or `pathlib.Path` must be utilized for platform-independent paths.
2. **Evaluator Runtime Constraint (Under 15 Minutes)**:
   - Full 3M dataset processing cannot be required on evaluator machines.
   - Pre-processed, validated brand data slices and vector stores must be cached with reproducible provenance.
3. **External API Dependence**:
   - Evaluators cannot be assumed to hold active paid LLM API keys.
   - An offline, deterministic grounded generator must be included alongside the API-driven LLM provider.

## 5. Assumptions
- The Twitter dataset represents real customer support interactions, but includes noisy, abbreviated, or multi-turn conversational quirks that require careful reconstruction.
- Candidate brands will differ substantially in domain coherence, language purity, and response style, requiring an empirical profiling and ranking procedure.

## 6. Risks & Mitigations
| Risk | Impact | Mitigation Strategy |
|---|---|---|
| **Data Leakage** | Contaminated evaluation results | Multi-layer audit: hash duplicate checks, thread isolation, time partitioning, and embedding cosine similarity audit ($>0.92$). |
| **Model Hallucination** | Fabrication of fake support policies | Strict grounding in retrieved historical resolutions; explicit unsupported-claim rate metric. |
| **Arbitrary Thresholds** | Unjustified escalation behavior | Tune confidence and similarity thresholds on a dedicated validation split; freeze all parameters prior to gold evaluation. |
| **Unreproducible LLM Outputs** | Evaluator unable to verify results | Dual-provider design: offline deterministic provider for bit-exact reproduction; external LLM provider for live API verification. |

## 7. Recommended Implementation Approach
Proceed with a modular Python package (`src/hiver_agent`) with clear layer boundaries:
`data/` $\to$ `nlp/` $\to$ `retrieval/` $\to$ `policy/` $\to$ `generation/` $\to$ `evaluation/`.
All configuration is externalized via YAML.

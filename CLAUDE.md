# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedsterPi is a Pi-style autonomous clinical case analysis agent. It performs clinical analysis through an event-driven loop where the LLM autonomously decides which tools to call, when to stop, and how to synthesize results. Uses SYNTHEA/FHIR data and optional MCP server integration for complex document analysis.

## Core Architecture

### Pi-Style Event Loop (agent.py)

The agent implements a minimal event-driven architecture (inspired by Mario Zechner's Pi Agent Framework):

```
User Query
    ↓
┌─────────────────────────────────┐
│  Event Loop                     │
│  ┌───────────────────────────┐  │
│  │ 1. Call LLM with messages │  │
│  │ 2. If tool_use → execute  │  │
│  │ 3. Add result to messages │  │
│  │ 4. Repeat until end_turn  │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    ↓
Final Response
```

**Key Characteristics:**
- Messages list is the only state
- Model decides everything autonomously (which tools, when to stop)
- Full observability via event callbacks (`EventType`: LOOP_START, LLM_REQUEST, LLM_RESPONSE, TOOL_CALL, TOOL_RESULT, LOOP_END, ERROR)
- No complex state machine (unlike LangGraph)

**Safety Mechanisms:**
- `max_iterations` limit (default: 25) prevents runaway loops
- Model decides when tasks are complete (no separate validation module)

### LLM Module (model.py)

The `call_llm` function exclusively uses Claude (Anthropic) models:

- Default model: `claude-sonnet-4-5-20250514`
- Supports vision API via `images` parameter (base64-encoded PNGs)
- Tool binding for autonomous tool selection via `tools` parameter
- Retry logic with exponential backoff (3 attempts)

### Document Store (tools/__init__.py)

Token-efficient document analysis without `exec()`:
- `_document_store: Dict[str, str]` - In-memory storage for uploaded documents
- `store_document(doc_id, content)` - Store document for later searching
- `get_document(doc_id)` - Retrieve stored document
- Documents are searched/extracted server-side; only relevant snippets sent to LLM

### Image Utilities (utils/image_utils.py)

Token-efficient medical image conversion:
- `dicom_to_base64_png()` - Converts DICOM → optimized PNG (~32MB → ~200KB)
- `load_ecg_image_from_csv()` - Extracts base64 ECG from observations CSV
- `find_patient_dicom_files()` - Locates all DICOM files for a patient

Dependencies: `pydicom`, `pillow`

## Data Sources

### Coherent Data Set

Medster uses the **Coherent Data Set** (9 GB synthetic dataset):
- FHIR bundles (1,278 longitudinal patient records)
- DICOM imaging (298 brain MRIs)
- Genomic data (889 CSV files)
- Physiological data (ECGs in observations.csv)
- Clinical notes

**Locations:**
- FHIR: `./coherent_data/fhir/` (via `COHERENT_DATA_PATH`)
- DICOM: `./coherent_data/dicom/` (via `COHERENT_DICOM_PATH`)
- CSV/ECG: `./coherent_data/csv/` (via `COHERENT_CSV_PATH`)
- Genomic: `./coherent_data/dna/` (via `COHERENT_DNA_PATH`)

**Download:** https://synthea.mitre.org/downloads

**Cloud Storage:** Set `USE_GCS=true` and `GCS_COHERENT_BUCKET` for Cloud Run deployments.

### MCP Server Integration (Optional)

Medster can connect to a FastMCP medical analysis server for specialist-level clinical reasoning:

**Recursive AI Architecture:**
- Local: Claude Sonnet 4.5 (Medster) - Orchestration, data extraction
- Remote: Claude Sonnet 4.5 (MCP Server) - Specialist medical document analysis

**Tool:** `analyze_medical_document` in `tools/analysis/mcp_client.py`

**Configuration:** `MCP_SERVER_URL`, `MCP_API_KEY`, `MCP_DEBUG` in `.env`

## Vision Analysis Capabilities

Medster supports vision-based analysis of medical images using Claude's vision API.

### Image Data Structure

**DICOM Images** (`./coherent_data/dicom/`):
- 298 brain MRI scans (~32MB each)
- Naming: `FirstName_LastName_UUID[DICOM_ID].dcm`
- Auto-optimized to ~800x800 PNG (~200KB) for token efficiency

**ECG Waveforms** (`./coherent_data/csv/observations.csv`):
- Base64-encoded PNG images (already optimized)
- LOINC code: 29303009

### Vision Analysis Workflow

**Two-step process:**

1. **Tool loads/converts image** (`analyze_image` in `tools/__init__.py`):
   - Calls vision analyzers in `tools/analysis/vision_analyzer.py`
   - Converts DICOM → base64 PNG or extracts ECG from CSV

2. **LLM analyzes image** (`call_llm` in `model.py`):
   - Vision-capable model (Claude Sonnet 4.5) analyzes the image
   - Returns clinical findings (masses, hemorrhage, rhythm abnormalities, etc.)

### Vision Analyzers (tools/analysis/vision_analyzer.py)

- `analyze_patient_dicom(patient_id, clinical_question)` - DICOM image analysis
- `analyze_patient_ecg(patient_id, clinical_question)` - ECG rhythm analysis

Both use `call_llm` with `images` parameter after converting images via `utils/image_utils.py`.

## Core Tools (tools/__init__.py)

MedsterPi uses **6 core tools** (hybrid approach - removed exec-based code generation):

### 1. `get_patient_data`
Comprehensive patient data retrieval. Combines labs, vitals, medications, conditions, notes.
```python
get_patient_data(
    patient_id: str,
    data_types: List[str] = ["all"],  # Options: labs, vitals, medications, conditions, notes, all
    limit: int = 50
) -> Dict[str, Any]
```

### 2. `search_patients`
Find patients by criteria (condition, medication, lab values).
```python
search_patients(
    condition: Optional[str] = None,
    medication: Optional[str] = None,
    lab_name: Optional[str] = None,
    lab_min: Optional[float] = None,
    lab_max: Optional[float] = None,
    limit: int = 20
) -> Dict[str, Any]
```

### 3. `analyze_image`
Medical image analysis (DICOM, ECG, X-ray) using Claude vision API.
```python
analyze_image(
    patient_id: str,
    image_type: str,  # "dicom", "ecg", "xray"
    clinical_question: Optional[str] = None
) -> Dict[str, Any]
```

### 4. `calculate_score`
Clinical risk scores (MELD, CHA2DS2-VASc, APACHE II, Wells DVT/PE, CURB-65, SOFA).
```python
calculate_score(
    patient_id: str,
    score_type: str  # "meld", "cha2ds2_vasc", "apache_ii", "wells_dvt", "wells_pe", "curb65", "sofa"
) -> Dict[str, Any]
```

### 5. `search_document`
Token-efficient document search (NO exec). Returns only matching lines with context.
```python
search_document(
    content: str,
    search_terms: List[str],
    case_sensitive: bool = False,
    context_lines: int = 1,
    max_matches_per_term: int = 10
) -> Dict[str, Any]
```

### 6. `extract_document_sections`
Token-efficient section extraction (NO exec). Returns only requested sections.
```python
extract_document_sections(
    content: str,
    section_headers: List[str],
    max_section_length: int = 2000
) -> Dict[str, Any]
```

### Supporting Document Tool

**`store_and_summarize_document`** - Store document in `_document_store` for later searching.
```python
store_and_summarize_document(content: str, doc_id: str = "default")
```

### Underlying Tool Implementations

The core tools wrap implementations in:
- `tools/medical/patient_data.py` - Labs, vitals, demographics, conditions
- `tools/medical/clinical_notes.py` - Notes, discharge summaries
- `tools/medical/medications.py` - Medication lists
- `tools/clinical/scores.py` - Risk score calculations
- `tools/analysis/vision_analyzer.py` - DICOM/ECG analysis

### Tool Registry

Tools are registered in `TOOL_REGISTRY` dict in `tools/__init__.py`. Execute via:
```python
from medster.tools import execute_tool
result = execute_tool("get_patient_data", {"patient_id": "123", "data_types": ["labs"]})
```

## Development Commands

### Environment Setup

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .

# Setup environment
cp env.example .env
# Edit .env with API keys and paths
```

### Running the Agent

```bash
# Primary method (uses entry point defined in pyproject.toml)
uv run medster-pi

# Alternative methods
python -m medster.cli
python src/medster/cli.py

# With API key inline
ANTHROPIC_API_KEY=sk-ant-xxx uv run medster-pi
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_scores.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black src/ --line-length 100

# Lint code
ruff check src/

# Both (recommended before committing)
black src/ --line-length 100 && ruff check src/
```

### Web Interface (Optional)

```bash
# Terminal 1: Backend API
uv run uvicorn medster.api:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

Access at http://localhost:3000

## Key Implementation Patterns

### Adding New Tools

1. Create tool function in appropriate submodule (e.g., `tools/medical/`)
2. Import and wrap in `tools/__init__.py` if needed
3. Add to `TOOL_REGISTRY` dict
4. Update `get_tools_schema()` in `prompts.py` to advertise to LLM

### Tool Execution Flow

```python
from medster.tools import execute_tool, get_available_tools

# Execute a tool by name
result = execute_tool("get_patient_data", {"patient_id": "123", "data_types": ["labs"]})

# List available tools
tools = get_available_tools()  # ["get_patient_data", "search_patients", ...]
```

### Document Analysis Pattern

Token-efficient document analysis (no exec):

```python
# 1. Store document first
store_and_summarize_document(content=large_document, doc_id="discharge_summary_001")

# 2. Search for specific terms (only matching lines returned)
search_document(
    content=large_document,
    search_terms=["diabetes", "hypertension", "A1c"],
    context_lines=2
)

# 3. Or extract specific sections
extract_document_sections(
    content=large_document,
    section_headers=["Assessment", "Plan", "Medications"]
)
```

### Vision Analysis Pattern

```python
from medster.tools import analyze_image

# Analyze DICOM/ECG
result = analyze_image(
    patient_id="12345",
    image_type="dicom",  # or "ecg", "xray"
    clinical_question="Look for signs of hemorrhage"
)

# The tool internally uses call_llm with images parameter
```

### Event Handler for Observability

```python
from medster.agent import Agent, Event, EventType

def my_event_handler(event: Event):
    if event.type == EventType.TOOL_CALL:
        print(f"Tool called: {event.data.get('tool_name')}")
    elif event.type == EventType.LOOP_END:
        print(f"Completed in {event.data.get('iterations')} iterations")

agent = Agent(event_handler=my_event_handler, verbose=True)
result = agent.run("Analyze patient 12345")
```

## Environment Variables

Required in `.env` file:

```bash
# Required: Anthropic API key for Claude Sonnet 4.5
ANTHROPIC_API_KEY=sk-ant-...

# Required for Coherent Data Set access (local development)
COHERENT_DATA_PATH=./coherent_data/fhir
COHERENT_DICOM_PATH=./coherent_data/dicom
COHERENT_CSV_PATH=./coherent_data/csv
COHERENT_DNA_PATH=./coherent_data/dna

# Cloud Run deployment (set USE_GCS=true for cloud)
USE_GCS=false
GCS_COHERENT_BUCKET=your-bucket-name

# Optional: MCP server for complex analysis
MCP_SERVER_URL=http://localhost:8000
MCP_API_KEY=...
MCP_DEBUG=false

# Frontend (Next.js) - only needed for web interface
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
NEXTAUTH_SECRET=...
NEXTAUTH_URL=http://localhost:3000
```

## Code Modification Guidelines

### Modifying the Agent Loop

The agent loop in `agent.py` is intentionally minimal:
- `max_iterations` - Safety limit (default: 25)
- `model` - Anthropic model name
- `event_handler` - Optional callback for observability
- `verbose` - Print debug info to console

Key methods:
- `run(query)` - Main entry point, returns final response
- `_emit()` - Emit events for observability

### Changing Models

Default model is set in `Agent.__init__()`:
```python
model: str = "claude-sonnet-4-5-20250514"
```

Supported models (via `call_llm` in `model.py`):
- `claude-sonnet-4-5-20250514` (default, recommended)
- `claude-opus-4-5-20251101` (more capable, more expensive)
- `claude-haiku-4` (faster, cheaper)

### FHIR Data Access

All FHIR parsing happens in `tools/medical/api.py`:
- `load_patient_bundle(patient_id)` - Loads patient JSON file
- `extract_resources(bundle, resource_type)` - Filters by resource type
- Various `format_*` functions - Convert FHIR to readable dicts

### Updating Tool Schemas

Tool schemas for LLM are defined in `prompts.py`:
```python
def get_tools_schema() -> List[Dict]:
    # Returns OpenAI-compatible tool definitions
```

When adding/modifying tools, update this function so the LLM knows how to call them.

## Safety & Disclaimers

**IMPORTANT:** Medster is for research and educational purposes only.
- Not for clinical decision-making without physician review
- Critical value flagging is simplified (not comprehensive)
- Drug interaction checking is basic (not production-grade)
- Always verify findings with appropriate clinical resources

## Citation

When using the Coherent Data Set:

> Walonoski J, et al. The "Coherent Data Set": Combining Patient Data and Imaging in a Comprehensive, Synthetic Health Record. Electronics. 2022; 11(8):1199. https://doi.org/10.3390/electronics11081199
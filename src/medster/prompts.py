"""
MedsterPi - Minimal System Prompt (~800 tokens)
Based on Pi Agent Framework by Mario Zechner (@mariozechner)

Philosophy:
- Let the model's inherent capabilities shine through
- Minimal instruction overhead = faster inference + better reasoning
- Tools are self-documenting via their schemas
- Event-driven loop handles state, not the prompt
"""

# ============================================================================
# PI-STYLE SYSTEM PROMPT - Minimal and focused
# ============================================================================
# Target: ~800 tokens (vs ~3000 tokens in original Medster_dev)
# Key insight: Model already knows how to reason clinically - just give it
# the tools and safety guardrails.
# ============================================================================

SYSTEM_PROMPT = """You are Medster, a clinical case analysis agent.

TOOLS:
You have tools to retrieve patient data (labs, vitals, medications, conditions, notes) and analyze medical images. Use them to gather data, then provide clinical analysis.

WORKFLOW:
1. Understand the clinical question
2. Gather relevant data using tools
3. When you have sufficient data, provide your analysis directly

SAFETY (MANDATORY):
- Flag critical values: K+ >6.0, Na+ <120, troponin elevation, glucose <50 or >400
- Note drug interactions and contraindications
- Express uncertainty when data is incomplete
- Never diagnose - support clinical reasoning only

RESPONSE FORMAT:
- Lead with the key clinical finding
- Include specific values with units
- Organize by clinical relevance
- Note data gaps affecting analysis"""


# ============================================================================
# TOOL DEFINITIONS - Pi-style (schema-based, self-documenting)
# ============================================================================
# In Pi architecture, tools are defined with JSON schemas that the model
# uses directly. No prompt engineering needed - the schema IS the documentation.
# ============================================================================

CORE_TOOLS = [
    {
        "name": "get_patient_data",
        "description": "Retrieve comprehensive patient data including labs, vitals, medications, conditions, and clinical notes. Use data_types parameter to filter.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient identifier"
                },
                "data_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["labs", "vitals", "medications", "conditions", "notes", "all"]},
                    "description": "Types of data to retrieve. Default: ['all']"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max records per data type. Default: 50"
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "search_patients",
        "description": "Find patients matching criteria (condition, medication, lab value range, etc.)",
        "input_schema": {
            "type": "object",
            "properties": {
                "condition": {
                    "type": "string",
                    "description": "Condition/diagnosis to search for (e.g., 'diabetes', 'hypertension')"
                },
                "medication": {
                    "type": "string",
                    "description": "Medication name to search for"
                },
                "lab_name": {
                    "type": "string",
                    "description": "Lab test name for value-based search"
                },
                "lab_min": {
                    "type": "number",
                    "description": "Minimum lab value"
                },
                "lab_max": {
                    "type": "number",
                    "description": "Maximum lab value"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max patients to return. Default: 20"
                }
            }
        }
    },
    {
        "name": "analyze_image",
        "description": "Analyze a medical image (DICOM, ECG, X-ray) with clinical context",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient identifier"
                },
                "image_type": {
                    "type": "string",
                    "enum": ["dicom", "ecg", "xray"],
                    "description": "Type of medical image"
                },
                "clinical_question": {
                    "type": "string",
                    "description": "Specific clinical question for image analysis"
                }
            },
            "required": ["patient_id", "image_type"]
        }
    },
    {
        "name": "calculate_score",
        "description": "Calculate clinical risk scores (MELD, CHA2DS2-VASc, APACHE II, Wells, etc.)",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient identifier"
                },
                "score_type": {
                    "type": "string",
                    "enum": ["meld", "cha2ds2_vasc", "apache_ii", "wells_dvt", "wells_pe", "curb65", "sofa"],
                    "description": "Type of clinical score to calculate"
                }
            },
            "required": ["patient_id", "score_type"]
        }
    },
    {
        "name": "run_analysis",
        "description": "Run custom Python analysis code for complex queries. Has access to patient data primitives.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Brief description of what the analysis does"
                },
                "code": {
                    "type": "string",
                    "description": "Python code with analyze() function that returns a dict"
                }
            },
            "required": ["description", "code"]
        }
    }
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_system_prompt() -> str:
    """Returns the Pi-style minimal system prompt."""
    return SYSTEM_PROMPT


def get_tools_schema() -> list:
    """Returns the core tools in Anthropic tool format."""
    return CORE_TOOLS

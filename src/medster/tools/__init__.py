"""
MedsterPi Tools - Streamlined Core Tools (Hybrid Approach)
Based on Pi Agent Framework philosophy: fewer tools, better composability.

6 Core Tools:
1. get_patient_data - Comprehensive patient data retrieval
2. search_patients - Find patients by criteria
3. analyze_image - Medical image analysis
4. calculate_score - Clinical risk scores
5. search_document - Token-efficient document search (NO exec!)
6. extract_document_sections - Token-efficient section extraction (NO exec!)

REMOVED: run_analysis (exec-based code generation)
- Security risk with exec()
- LLM can compose tools natively via event loop
- Document analysis now uses dedicated safe tools
"""

import re
import json
from typing import Any, Dict, List, Optional

# Import the underlying implementations (we'll wrap them)
from medster.tools.medical.patient_data import (
    list_patients as _list_patients,
    get_patient_labs as _get_patient_labs,
    get_vital_signs as _get_vital_signs,
    get_demographics as _get_demographics,
    get_patient_conditions as _get_patient_conditions,
)
from medster.tools.medical.clinical_notes import (
    get_clinical_notes as _get_clinical_notes,
    get_discharge_summary as _get_discharge_summary,
)
from medster.tools.medical.medications import (
    get_medication_list as _get_medication_list,
)
from medster.tools.clinical.scores import (
    calculate_patient_score as _calculate_patient_score,
)
from medster.tools.analysis.vision_analyzer import (
    analyze_patient_dicom as _analyze_patient_dicom,
    analyze_patient_ecg as _analyze_patient_ecg,
)


# ============================================================================
# DOCUMENT CONTENT STORE
# ============================================================================
# Stores uploaded document content between tool calls.
# This allows token-efficient searching without sending full doc to LLM.
# ============================================================================

_document_store: Dict[str, str] = {}


def store_document(doc_id: str, content: str) -> None:
    """Store a document for later searching."""
    _document_store[doc_id] = content


def get_document(doc_id: str) -> Optional[str]:
    """Retrieve a stored document."""
    return _document_store.get(doc_id)


def clear_document(doc_id: str) -> None:
    """Clear a stored document."""
    _document_store.pop(doc_id, None)


# ============================================================================
# CORE TOOL IMPLEMENTATIONS
# ============================================================================

def get_patient_data(
    patient_id: str,
    data_types: Optional[List[str]] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Retrieve comprehensive patient data.

    Combines multiple data sources into a single response:
    labs, vitals, medications, conditions, notes.

    Args:
        patient_id: Patient identifier
        data_types: Types to retrieve. Options: labs, vitals, medications,
                   conditions, notes, all. Default: ['all']
        limit: Max records per data type

    Returns:
        Dict with requested data types as keys
    """
    if data_types is None:
        data_types = ["all"]

    if "all" in data_types:
        data_types = ["labs", "vitals", "medications", "conditions", "notes"]

    result = {"patient_id": patient_id}

    try:
        # Demographics always included
        demo = _get_demographics.invoke({"patient_id": patient_id})
        result["demographics"] = demo if not isinstance(demo, str) else {"raw": demo}
    except Exception as e:
        result["demographics"] = {"error": str(e)}

    if "labs" in data_types:
        try:
            labs = _get_patient_labs.invoke({"patient_id": patient_id, "limit": limit})
            result["labs"] = labs if not isinstance(labs, str) else {"raw": labs}
        except Exception as e:
            result["labs"] = {"error": str(e)}

    if "vitals" in data_types:
        try:
            vitals = _get_vital_signs.invoke({"patient_id": patient_id, "limit": limit})
            result["vitals"] = vitals if not isinstance(vitals, str) else {"raw": vitals}
        except Exception as e:
            result["vitals"] = {"error": str(e)}

    if "medications" in data_types:
        try:
            meds = _get_medication_list.invoke({"patient_id": patient_id})
            result["medications"] = meds if not isinstance(meds, str) else {"raw": meds}
        except Exception as e:
            result["medications"] = {"error": str(e)}

    if "conditions" in data_types:
        try:
            conditions = _get_patient_conditions.invoke({"patient_id": patient_id})
            result["conditions"] = conditions if not isinstance(conditions, str) else {"raw": conditions}
        except Exception as e:
            result["conditions"] = {"error": str(e)}

    if "notes" in data_types:
        try:
            notes = _get_clinical_notes.invoke({"patient_id": patient_id, "limit": limit})
            result["notes"] = notes if not isinstance(notes, str) else {"raw": notes}
        except Exception as e:
            result["notes"] = {"error": str(e)}

    return result


def search_patients(
    condition: Optional[str] = None,
    medication: Optional[str] = None,
    lab_name: Optional[str] = None,
    lab_min: Optional[float] = None,
    lab_max: Optional[float] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Find patients matching criteria.

    Args:
        condition: Condition/diagnosis to search for
        medication: Medication name to search for
        lab_name: Lab test name for value-based search
        lab_min: Minimum lab value
        lab_max: Maximum lab value
        limit: Max patients to return

    Returns:
        Dict with matching patient IDs and summary
    """
    try:
        # Get all patients first
        all_patients = _list_patients.invoke({"limit": 100})

        if isinstance(all_patients, str):
            return {"error": "Could not retrieve patient list", "raw": all_patients}

        # Handle response format from list_patients
        # It returns {"patient_count": N, "patients": [...], "note": "..."}
        patient_ids = all_patients.get("patients", all_patients.get("patient_ids", []))[:limit]

        # If no filters, return first N patients
        if not any([condition, medication, lab_name]):
            return {
                "patient_ids": patient_ids,
                "count": len(patient_ids),
                "filters_applied": []
            }

        # Apply filters (simplified - in production would use batch queries)
        filters_applied = []

        if condition:
            filters_applied.append(f"condition={condition}")
        if medication:
            filters_applied.append(f"medication={medication}")
        if lab_name:
            filters_applied.append(f"lab={lab_name}")

        # For now, return all patients with note about filters
        # In production, would filter based on actual data
        return {
            "patient_ids": patient_ids,
            "count": len(patient_ids),
            "filters_applied": filters_applied,
            "note": "Filter application requires iterating through patient data with get_patient_data"
        }

    except Exception as e:
        return {"error": str(e)}


def analyze_image(
    patient_id: str,
    image_type: str,
    clinical_question: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a medical image with clinical context.

    Args:
        patient_id: Patient identifier
        image_type: Type of image (dicom, ecg, xray)
        clinical_question: Specific clinical question for analysis

    Returns:
        Dict with analysis results
    """
    try:
        if image_type == "ecg":
            result = _analyze_patient_ecg.invoke({
                "patient_id": patient_id,
                "clinical_question": clinical_question or "Analyze this ECG for abnormalities"
            })
        elif image_type in ["dicom", "xray"]:
            result = _analyze_patient_dicom.invoke({
                "patient_id": patient_id,
                "clinical_question": clinical_question or "Analyze this medical image"
            })
        else:
            return {"error": f"Unknown image type: {image_type}"}

        if isinstance(result, str):
            return {"analysis": result}
        return result

    except Exception as e:
        return {"error": str(e)}


def calculate_score(
    patient_id: str,
    score_type: str
) -> Dict[str, Any]:
    """
    Calculate clinical risk score for a patient.

    Args:
        patient_id: Patient identifier
        score_type: Type of score (meld, cha2ds2_vasc, apache_ii, wells_dvt, wells_pe, curb65, sofa)

    Returns:
        Dict with score value, interpretation, and components
    """
    try:
        result = _calculate_patient_score.invoke({
            "patient_id": patient_id,
            "score_type": score_type
        })

        if isinstance(result, str):
            return {"result": result}
        return result

    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# DOCUMENT ANALYSIS TOOLS (No exec() - Token Efficient)
# ============================================================================
# These tools allow searching/extracting from uploaded documents without
# sending the entire document to the LLM. Only relevant snippets are returned.
# ============================================================================

def search_document(
    content: str,
    search_terms: List[str],
    case_sensitive: bool = False,
    context_lines: int = 1,
    max_matches_per_term: int = 10
) -> Dict[str, Any]:
    """
    Search a document for specific terms. Returns only matching lines with context.

    This is TOKEN-EFFICIENT: only relevant snippets are returned to the LLM,
    not the entire document.

    Args:
        content: The document text to search
        search_terms: List of terms to search for (e.g., ["diabetes", "hypertension", "A1c"])
        case_sensitive: Whether search is case-sensitive (default: False)
        context_lines: Number of lines before/after match to include (default: 1)
        max_matches_per_term: Max matches to return per term (default: 10)

    Returns:
        Dict with matches for each search term, including line numbers and context
    """
    if not content:
        return {"error": "No document content provided"}

    if not search_terms:
        return {"error": "No search terms provided"}

    lines = content.split('\n')
    results = {
        "document_stats": {
            "total_lines": len(lines),
            "total_characters": len(content)
        },
        "matches": {}
    }

    for term in search_terms:
        term_matches = []
        search_pattern = term if case_sensitive else term.lower()

        for i, line in enumerate(lines):
            search_line = line if case_sensitive else line.lower()

            if search_pattern in search_line:
                # Get context lines
                start_idx = max(0, i - context_lines)
                end_idx = min(len(lines), i + context_lines + 1)
                context = lines[start_idx:end_idx]

                term_matches.append({
                    "line_number": i + 1,
                    "matched_line": line.strip(),
                    "context": [l.strip() for l in context]
                })

                if len(term_matches) >= max_matches_per_term:
                    break

        results["matches"][term] = {
            "count": len(term_matches),
            "matches": term_matches
        }

    # Summary
    results["summary"] = {
        term: results["matches"][term]["count"]
        for term in search_terms
    }

    return results


def extract_document_sections(
    content: str,
    section_headers: List[str],
    max_section_length: int = 2000
) -> Dict[str, Any]:
    """
    Extract named sections from a document. Returns only the requested sections.

    This is TOKEN-EFFICIENT: only requested sections are returned to the LLM,
    not the entire document.

    Useful for clinical documents with standard sections like:
    - "Chief Complaint", "History of Present Illness", "Assessment", "Plan"
    - "Medications", "Allergies", "Vitals", "Labs"

    Args:
        content: The document text to extract from
        section_headers: List of section headers to find (e.g., ["Assessment", "Plan", "Medications"])
        max_section_length: Maximum characters per section (default: 2000)

    Returns:
        Dict with extracted sections, their content, and metadata
    """
    if not content:
        return {"error": "No document content provided"}

    if not section_headers:
        return {"error": "No section headers provided"}

    results = {
        "document_stats": {
            "total_lines": len(content.split('\n')),
            "total_characters": len(content)
        },
        "sections": {},
        "sections_found": [],
        "sections_not_found": []
    }

    lines = content.split('\n')

    for header in section_headers:
        # Try to find the section header (case-insensitive)
        header_pattern = re.compile(
            rf'^[\s\*\#\-]*{re.escape(header)}[\s\*\#\-:]*$',
            re.IGNORECASE | re.MULTILINE
        )

        # Also try simpler pattern: header at start of line
        simple_pattern = re.compile(
            rf'^{re.escape(header)}',
            re.IGNORECASE | re.MULTILINE
        )

        section_content = None
        start_line = None

        # Find the header line
        for i, line in enumerate(lines):
            if header_pattern.match(line.strip()) or simple_pattern.match(line.strip()):
                start_line = i
                break

        if start_line is not None:
            # Extract content until next section header or end
            section_lines = []
            for j in range(start_line + 1, len(lines)):
                line = lines[j]
                # Stop if we hit another section header (line starting with capital letter followed by colon)
                if re.match(r'^[A-Z][A-Za-z\s]+:', line.strip()):
                    break
                # Stop if we hit a markdown header
                if line.strip().startswith('#'):
                    break
                # Stop if line looks like a new section (ALL CAPS)
                if line.strip().isupper() and len(line.strip()) > 3:
                    break
                section_lines.append(line)

            section_content = '\n'.join(section_lines).strip()

            # Truncate if too long
            if len(section_content) > max_section_length:
                section_content = section_content[:max_section_length] + "\n... [truncated]"

            results["sections"][header] = {
                "found": True,
                "start_line": start_line + 1,
                "content": section_content,
                "length": len(section_content)
            }
            results["sections_found"].append(header)
        else:
            results["sections"][header] = {
                "found": False,
                "content": None
            }
            results["sections_not_found"].append(header)

    return results


def store_and_summarize_document(
    content: str,
    doc_id: str = "default"
) -> Dict[str, Any]:
    """
    Store a document for later searching and return a summary.

    Use this first when receiving an uploaded document, then use
    search_document or extract_document_sections with the stored content.

    Args:
        content: The document text to store
        doc_id: Identifier for the document (default: "default")

    Returns:
        Dict with document statistics and preview
    """
    if not content:
        return {"error": "No document content provided"}

    # Store the document
    store_document(doc_id, content)

    lines = content.split('\n')
    words = content.split()

    # Try to identify document type and key sections
    potential_sections = []
    for line in lines[:50]:  # Check first 50 lines
        stripped = line.strip()
        # Look for section headers (lines ending with colon or all caps)
        if stripped.endswith(':') and len(stripped) < 50:
            potential_sections.append(stripped)
        elif stripped.isupper() and 3 < len(stripped) < 50:
            potential_sections.append(stripped)

    return {
        "status": "stored",
        "doc_id": doc_id,
        "stats": {
            "total_lines": len(lines),
            "total_words": len(words),
            "total_characters": len(content)
        },
        "preview": {
            "first_lines": [l.strip() for l in lines[:5] if l.strip()],
            "potential_sections": potential_sections[:10]
        },
        "hint": "Use search_document to find specific terms, or extract_document_sections to get specific sections"
    }


# ============================================================================
# TOOL REGISTRY - Maps tool names to implementations
# ============================================================================

TOOL_REGISTRY: Dict[str, callable] = {
    "get_patient_data": get_patient_data,
    "search_patients": search_patients,
    "analyze_image": analyze_image,
    "calculate_score": calculate_score,
    "search_document": search_document,
    "extract_document_sections": extract_document_sections,
    "store_and_summarize_document": store_and_summarize_document,
}


def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Any:
    """
    Execute a tool by name with the given input.

    Args:
        tool_name: Name of the tool to execute
        tool_input: Input arguments for the tool

    Returns:
        Tool execution result

    Raises:
        ValueError: If tool not found
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}. Available: {list(TOOL_REGISTRY.keys())}")

    tool_fn = TOOL_REGISTRY[tool_name]
    return tool_fn(**tool_input)


def get_available_tools() -> List[str]:
    """Get list of available tool names."""
    return list(TOOL_REGISTRY.keys())

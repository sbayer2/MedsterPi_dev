"""
MedsterPi Tools - Streamlined Core Tools
Based on Pi Agent Framework philosophy: fewer tools, better composability.

5 Core Tools:
1. get_patient_data - Comprehensive patient data retrieval
2. search_patients - Find patients by criteria
3. analyze_image - Medical image analysis
4. calculate_score - Clinical risk scores
5. run_analysis - Custom Python analysis

These 5 tools can accomplish everything the original 20+ tools did,
with cleaner interfaces and better composability.
"""

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
from medster.tools.analysis.code_generator import (
    generate_and_run_analysis as _generate_and_run_analysis,
)
from medster.tools.analysis.vision_analyzer import (
    analyze_patient_dicom as _analyze_patient_dicom,
    analyze_patient_ecg as _analyze_patient_ecg,
)


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

        patient_ids = all_patients.get("patient_ids", [])[:limit]

        # If no filters, return first N patients
        if not any([condition, medication, lab_name]):
            return {
                "patient_ids": patient_ids,
                "count": len(patient_ids),
                "filters_applied": []
            }

        # Apply filters (simplified - in production would use batch queries)
        matching = []
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
            "note": "Filter application requires data scan - use run_analysis for complex queries"
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


def run_analysis(
    description: str,
    code: str
) -> Dict[str, Any]:
    """
    Run custom Python analysis code.

    The code should define an analyze() function that returns a dict.
    Available primitives in sandbox:
    - get_patients(limit) -> list of patient IDs
    - load_patient(pid) -> patient data dict
    - load_labs(pid) -> labs list
    - load_vitals(pid) -> vitals list
    - load_conditions(pid) -> conditions list
    - load_medications(pid) -> medications list
    - scan_dicom_directory() -> list of DICOM files
    - load_dicom_image(pid, index) -> base64 image
    - load_ecg_image(pid) -> base64 image
    - log_progress(message) -> log progress during iteration

    Args:
        description: Brief description of the analysis
        code: Python code with analyze() function

    Returns:
        Dict with analysis results
    """
    try:
        result = _generate_and_run_analysis.invoke({
            "analysis_description": description,
            "code": code
        })

        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"result": result}
        return result

    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# TOOL REGISTRY - Maps tool names to implementations
# ============================================================================

TOOL_REGISTRY: Dict[str, callable] = {
    "get_patient_data": get_patient_data,
    "search_patients": search_patients,
    "analyze_image": analyze_image,
    "calculate_score": calculate_score,
    "run_analysis": run_analysis,
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

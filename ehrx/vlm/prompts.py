"""
Prompt templates for Vision-Language Model (VLM) EHR processing.

These prompts guide Gemini models to extract and classify clinical
document elements with semantic understanding.
"""

from typing import Optional, Dict, Any
from ehrx.vlm.models import DocumentContext


# System instruction for all VLM requests
SYSTEM_INSTRUCTION = """You are a clinical document analysis AI specialized in extracting structured information from Electronic Health Record (EHR) PDFs.

Your task is to analyze medical documents and identify all content elements with:
1. Precise bounding box coordinates
2. Semantic element type classification
3. Accurate text extraction
4. Confidence scores for quality assessment

You have deep knowledge of:
- Medical terminology and clinical abbreviations
- EHR document structures and layouts
- Clinical relationships (medications ↔ labs, problems ↔ assessments)
- Healthcare data formats (HL7, FHIR concepts)

Always prioritize accuracy over speed. When uncertain, indicate lower confidence scores."""


def build_element_extraction_prompt(
    context: DocumentContext,
    additional_instructions: Optional[str] = None
) -> str:
    """
    Build prompt for element detection and extraction.

    Args:
        context: Document context (page info, section hierarchy, etc.)
        additional_instructions: Optional additional guidance for VLM

    Returns:
        Formatted prompt string for VLM
    """
    # Build context section
    context_lines = [
        "## Document Context",
        f"- **Page**: {context.page_number + 1} of {context.total_pages}",
    ]

    if context.document_type:
        context_lines.append(f"- **Document Type**: {context.document_type}")

    if context.section_hierarchy:
        hierarchy_str = " → ".join(context.section_hierarchy)
        context_lines.append(f"- **Section Path**: {hierarchy_str}")

    if context.patient_context:
        context_lines.append(f"- **Patient Context**: {context.patient_context}")

    if context.preceding_summary:
        context_lines.append(f"- **Preceding Content**: {context.preceding_summary}")

    context_section = "\n".join(context_lines)

    # Build main prompt
    prompt = f"""{context_section}

## Task

Analyze this EHR page image and extract ALL visible elements with complete information.

For each element, provide:

1. **Bounding Box**: Precise pixel coordinates [x0, y0, x1, y1]
   - x0, y0: Top-left corner
   - x1, y1: Bottom-right corner
   - Use image pixel coordinates (0,0 = top-left of image)

2. **Semantic Type**: Classify element into one of these types:

   **Document Structure:**
   - `document_header`: Hospital/clinic identifying information
   - `patient_demographics`: Name, DOB, MRN, contact info
   - `page_metadata`: Page numbers, dates, document IDs
   - `section_header`: Major section headings (PROBLEMS, MEDICATIONS, LABS)
   - `subsection_header`: Minor headings within sections

   **Clinical Content:**
   - `clinical_paragraph`: Free-text clinical narratives
   - `medication_table`: Structured medication lists
   - `lab_results_table`: Laboratory values with ranges/units
   - `recommendations_table`: Table of reccomended treatments, screenings, specialist referrals, vaccinations, or other actions for the patient
   - `vital_signs_table`: Temperature, BP, pulse, etc.
    - `general_table`:  A multi-column block of aligned text or numbers with a clear header row (May coincide with section headers) and at least two data rows, where each column contains consistent types of information. It should not match a domain-specific table type and must not resemble a list (lists have a single text column, no headers, and use bullets or numbers). 
   - `problem_list`: Diagnoses with ICD codes
   - `assessment_plan`: Clinical reasoning and treatment plans
   - `list_items`: Bullet/numbered lists with clinical content

   **Special Content:**
   - `handwritten_annotation`: Handwritten notes
   - `stamp_signature`: Official stamps or signatures
   - `medical_figure`: Graphs, charts, anatomical diagrams
   - `form_field_group`: Label-value pairs from forms

   **Administrative:**
   - `margin_content`: Headers, footers, confidentiality notices
   - `uncategorized`: Content that doesn't fit other categories

3. **Text Content**:
   - **Clinical content** (patient_demographics, clinical_paragraph, medication_table, lab_results_table, vital_signs_table, problem_list, assessment_plan, list_items): Extract ALL text exactly as written - preserve every detail
   - **Administrative content** (margin_content, page_metadata, confidentiality notices): Provide brief summary or first line only
   - Preserve medical abbreviations and terminology
   - Maintain formatting for clinical content
   - For tables: preserve complete row/column structure
   - For empty/non-text elements: use empty string ""

4. **Confidence Scores**: Provide three scores (0.0-1.0) for each element:

   - `extraction` - Confidence in text accuracy:
     * 0.9-1.0: Clear, legible text with no ambiguity
     * 0.7-0.9: Mostly clear but some minor blur or spacing issues
     * 0.5-0.7: Partially unclear, handwritten, or low quality
     * <0.5: Very difficult to read, severely degraded

   - `classification` - Confidence in semantic type assignment:
     * 0.9-1.0: Unambiguous type (e.g., clearly labeled "MEDICATIONS" section)
     * 0.7-0.9: Strong indicators but some ambiguity (e.g., unlabeled table that's clearly labs)
     * 0.5-0.7: Uncertain type, could fit multiple categories
     * <0.5: Very ambiguous or unclear purpose

   - `clinical_context` - Confidence in clinical metadata understanding:
     * 0.9-1.0: Clear clinical domain and temporal context
     * 0.7-0.9: Can infer clinical context from content
     * 0.5-0.7: Ambiguous clinical significance
     * <0.5: Unable to determine clinical context

   **Be realistic**: Don't default to 0.5 - assess each element independently

5. **Clinical Metadata** (when applicable):
   - `temporal_qualifier`: "current", "historical", "planned"
   - `clinical_domain`: "pharmacology", "laboratory", "vitals", "imaging", etc.
   - `requires_validation`: true/false if human review needed

## Output Format

Return a JSON object with this exact structure:

```json
{{
  "elements": [
    {{
      "element_id": "E_0001",
      "semantic_type": "section_header",
      "bbox": {{
        "x0": 100.0,
        "y0": 50.0,
        "x1": 400.0,
        "y1": 80.0
      }},
      "content": "MEDICATIONS",
      "confidence_scores": {{
        "extraction": 0.98,
        "classification": 0.95,
        "clinical_context": 0.90
      }},
      "clinical_metadata": {{
        "temporal_qualifier": "current",
        "clinical_domain": "pharmacology",
        "requires_validation": false
      }},
      "page": {context.page_number}
    }},
    {{
      "element_id": "E_0002",
      "semantic_type": "medication_table",
      "bbox": {{
        "x0": 100.0,
        "y0": 100.0,
        "x1": 700.0,
        "y1": 300.0
      }},
      "content": "Metformin 500mg PO BID\\nLisinopril 10mg PO daily\\nAspirin 81mg PO daily",
      "confidence_scores": {{
        "extraction": 0.92,
        "classification": 0.88,
        "clinical_context": 0.85
      }},
      "clinical_metadata": {{
        "temporal_qualifier": "current",
        "clinical_domain": "pharmacology",
        "requires_validation": false
      }},
      "page": {context.page_number}
    }}
  ],
  "overall_confidence": 0.91,
  "requires_human_review": false,
  "review_reasons": []
}}
```

## Important Guidelines

1. **Completeness**: Extract EVERY visible text element, no matter how small
2. **Accuracy**: Prioritize correct text extraction over speed
3. **Precision**: Bounding boxes should tightly fit content (no excess whitespace)
4. **Separation**: Keep distinct elements separate (don't merge headers with paragraphs)
5. **Tables**: Extract entire table as single element with structured content
6. **Confidence**: Be honest about uncertainty - lower scores trigger appropriate review
7. **Validation Flags**: Set `requires_validation: true` for:
   - Unclear handwriting
   - Poor image quality regions
   - Ambiguous medical terminology
   - Critical clinical values (dosages, lab results)
8. **Element IDs**: Use sequential IDs starting with E_0001, E_0002, etc.

{additional_instructions if additional_instructions else ""}

Now analyze the provided page image and return the JSON response."""

    return prompt


def build_element_extraction_prompt_concise(
    context: DocumentContext,
    additional_instructions: Optional[str] = None
) -> str:
    """
    Build a concise version of the element extraction prompt for pages with very dense content.
    
    This version requests shorter content summaries and prioritizes structure over full text
    to avoid hitting MAX_TOKENS limits.
    
    Args:
        context: Document context (page info, section hierarchy, etc.)
        additional_instructions: Optional additional guidance for VLM
    
    Returns:
        Formatted concise prompt string for VLM
    """
    # Build context section (same as regular prompt)
    context_lines = [
        "## Document Context",
        f"- **Page**: {context.page_number + 1} of {context.total_pages}",
    ]

    if context.document_type:
        context_lines.append(f"- **Document Type**: {context.document_type}")

    if context.section_hierarchy:
        hierarchy_str = " → ".join(context.section_hierarchy)
        context_lines.append(f"- **Section Path**: {hierarchy_str}")

    context_section = "\n".join(context_lines)

    # Build concise prompt - prioritize structure and key information
    prompt = f"""{context_section}

## Task (CONCISE MODE - Dense Page)

This page has very dense content. Extract ALL elements but use CONCISE text extraction to stay within token limits.

**CRITICAL INSTRUCTIONS FOR CONCISE MODE:**
1. **Extract ALL elements** - don't skip any
2. **Use shorter content** - summarize long paragraphs, extract key points from tables
3. **Prioritize structure** - bounding boxes and types are more important than full text
4. **For tables**: Extract column headers and first 2-3 rows only (indicate total row count)
5. **For paragraphs**: Extract first 100-150 characters + key medical terms
6. **For lists**: Extract first 3-5 items only (indicate total item count)

For each element, provide:

1. **Bounding Box**: Precise pixel coordinates [x0, y0, x1, y1] (same as normal mode)

2. **Semantic Type**: Same classification as normal mode

3. **Text Content** (CONCISE):
   - **Clinical content**: Extract key information only (medication names, dosages, lab values, vital signs)
   - **Tables**: Header row + first 2-3 data rows + note like "[... 15 more rows]"
   - **Paragraphs**: First 100-150 chars + key medical terms + note like "[... truncated]"
   - **Lists**: First 3-5 items + note like "[... 12 more items]"
   - **Administrative**: Brief summary only

4. **Confidence Scores**: Same as normal mode

5. **Clinical Metadata**: Same as normal mode

## Output Format

Same JSON structure as normal mode, but with concise content fields.

## Important Guidelines

1. **Completeness**: Extract EVERY element (don't skip due to conciseness)
2. **Key Information First**: Prioritize critical clinical data (medications, labs, vitals, diagnoses)
3. **Structure Over Detail**: Accurate bounding boxes and types are more important than full text
4. **Indicate Truncation**: Use "[... X more rows/items]" notation when content is truncated
5. **Preserve Critical Values**: Always extract dosages, lab values, vital signs completely even if other text is summarized

{additional_instructions if additional_instructions else ""}

Now analyze the provided page image and return the JSON response with CONCISE content."""

    return prompt


def build_table_extraction_prompt(table_bbox: Optional[Dict[str, float]] = None) -> str:
    """
    Build specialized prompt for table structure extraction.

    Args:
        table_bbox: Optional bounding box to focus on specific table region

    Returns:
        Formatted prompt for table extraction
    """
    bbox_instruction = ""
    if table_bbox:
        bbox_instruction = f"""
Focus specifically on the table region at coordinates:
- x0: {table_bbox['x0']}, y0: {table_bbox['y0']}
- x1: {table_bbox['x1']}, y1: {table_bbox['y1']}
"""

    prompt = f"""## Task: Structured Table Extraction

{bbox_instruction}

Extract this clinical table with complete structural information.

Provide:

1. **Table Type**: medication_table, lab_results_table, or vital_signs_table
2. **Column Headers**: List of column names (preserve medical terminology)
3. **Rows**: Array of row data preserving all values
4. **Structure Metadata**:
   - Number of columns
   - Number of data rows
   - Has header row: true/false
   - Cell alignment (left/center/right)

Return JSON format:

```json
{{
  "table_type": "medication_table",
  "columns": ["Medication", "Dose", "Route", "Frequency", "Start Date"],
  "rows": [
    ["Metformin", "500mg", "PO", "BID", "01/15/2024"],
    ["Lisinopril", "10mg", "PO", "daily", "01/15/2024"]
  ],
  "metadata": {{
    "num_columns": 5,
    "num_rows": 2,
    "has_header": true,
    "confidence": 0.94
  }}
}}
```

**Critical Requirements:**
- Preserve exact values (especially dosages, units, dates)
- Maintain row-column alignment
- Flag uncertain cells with lower confidence
- For lab results: include units and normal ranges if present
- For medications: capture dose, route, frequency accurately"""

    return prompt


def build_figure_interpretation_prompt() -> str:
    """
    Build prompt for medical figure/chart interpretation.

    Returns:
        Formatted prompt for figure analysis
    """
    prompt = """## Task: Medical Figure Interpretation

Analyze this medical figure (chart, graph, diagram) and provide:

1. **Figure Type**:
   - `graph`: Line/bar/scatter plot
   - `anatomical_diagram`: Body part illustration
   - `flowchart`: Clinical pathway or decision tree
   - `other`: Other visualization types

2. **Content Description**: Detailed textual description of:
   - What is being visualized
   - Key data points or trends
   - Clinical significance
   - Axes labels and units (for graphs)
   - Anatomical structures shown (for diagrams)

3. **Extracted Text**: All visible text labels, values, annotations

4. **Clinical Interpretation**: What this figure tells about patient status

Return JSON format:

```json
{
  "figure_type": "graph",
  "description": "Blood glucose trends over 7 days showing values ranging from 90-180 mg/dL",
  "extracted_text": ["Glucose (mg/dL)", "Day 1", "Day 7", "Target Range: 70-130"],
  "clinical_interpretation": "Blood glucose levels elevated but trending downward, approaching target range",
  "confidence": 0.88
}
```

**Guidelines:**
- Focus on clinical relevance
- Extract all text precisely (especially numerical values)
- Note any abnormal findings or trends
- Indicate if image quality affects interpretation"""

    return prompt


def build_validation_prompt(
    element_content: str,
    element_type: str
) -> str:
    """
    Build prompt for validating extracted content.

    Used for quality assurance and correction of low-confidence extractions.

    Args:
        element_content: Previously extracted content
        element_type: Semantic element type

    Returns:
        Formatted validation prompt
    """
    prompt = f"""## Task: Content Validation

Review this previously extracted content for accuracy:

**Element Type**: {element_type}

**Extracted Content**:
```
{element_content}
```

Verify and correct if needed:

1. **Text Accuracy**: Is the extracted text correct?
2. **Medical Terminology**: Are medical terms spelled correctly?
3. **Values**: Are numerical values (dosages, lab results) accurate?
4. **Classification**: Is the semantic type ({element_type}) correct?

Return JSON:

```json
{{
  "is_accurate": true,
  "corrected_content": "{element_content}",
  "corrected_type": "{element_type}",
  "issues_found": [],
  "confidence": 0.95,
  "recommended_action": "accept"
}}
```

Where `recommended_action` is:
- `accept`: Content is accurate, no changes needed
- `correct`: Content corrected (see corrected_content)
- `flag_for_review`: Uncertain, needs human review
- `reject`: Content extraction failed, retry needed"""

    return prompt


# Prompt fragments for context injection
CLINICAL_DOMAIN_HINTS = {
    "pharmacology": "Focus on medication names, dosages, routes, and frequencies.",
    "laboratory": "Focus on test names, values, units, and normal ranges.",
    "vitals": "Focus on vital sign measurements (BP, HR, temp, O2 sat, etc.).",
    "imaging": "Focus on imaging modality, findings, and interpretations.",
    "procedures": "Focus on procedure names, dates, and outcomes.",
    "demographics": "Focus on patient identifiers, contact info, and metadata.",
}


def get_domain_hint(clinical_domain: Optional[str]) -> str:
    """Get additional instructions based on clinical domain."""
    if not clinical_domain:
        return ""

    hint = CLINICAL_DOMAIN_HINTS.get(clinical_domain.lower(), "")
    return f"\n**Domain-Specific Guidance**: {hint}" if hint else ""

"""
Hybrid query agent with liberal filtering for EHR schema querying.

Two-stage process:
1. Flash analyzes query → identifies ALL potentially relevant fields (liberal)
2. Pro reasons over filtered context → returns matched elements with provenance
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai

from ehrx.vlm.config import VLMConfig


logger = logging.getLogger(__name__)


# All available semantic element types
SEMANTIC_TYPES = [
    "document_header",
    "patient_demographics",
    "page_metadata",
    "section_header",
    "subsection_header",
    "clinical_paragraph",
    "medication_table",
    "lab_results_table",
    "vital_signs_table",
    "recommendations_table",
    "general_table",
    "problem_list",
    "assessment_plan",
    "list_items",
    "handwritten_annotation",
    "stamp_signature",
    "medical_figure",
    "form_field_group",
    "margin_content",
    "uncategorized"
]


class HybridQueryAgent:
    """
    Query agent with liberal filtering + Pro reasoning.

    Architecture:
    1. Flash: Analyze query → identify ALL potentially relevant fields (liberal)
    2. Python: Deterministic filter → extract matching elements
    3. Pro: Reason over filtered context → answer with provenance
    """

    def __init__(
        self,
        schema_path: Optional[str] = None,
        vlm_config: Optional[VLMConfig] = None
    ):
        """
        Initialize hybrid query agent.

        Args:
            schema_path: Path to schema JSON file (can be set later)
            vlm_config: VLM configuration (defaults to from_env())
        """
        self.schema_path = schema_path
        self.schema = None
        self.vlm_config = vlm_config or VLMConfig.from_env()
        self.logger = logging.getLogger(__name__)

        # Initialize Vertex AI
        vertexai.init(
            project=self.vlm_config.project_id,
            location=self.vlm_config.location
        )

        # Initialize models
        self.flash_model = GenerativeModel(model_name="gemini-2.5-flash")
        self.pro_model = GenerativeModel(model_name="gemini-2.5-pro")

        # Load schema if provided
        if schema_path:
            self.load_schema(schema_path)

    def load_schema(self, schema_path: str) -> None:
        """
        Load schema JSON file.

        Args:
            schema_path: Path to schema JSON file
        """
        schema_path = Path(schema_path)

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        self.logger.info(f"Loading schema from: {schema_path}")

        with open(schema_path, 'r') as f:
            self.schema = json.load(f)

        # Calculate stats
        total_elements = sum(
            len(page.get("elements", []))
            for page in self.schema.get("pages", [])
        )

        self.logger.info(
            f"Schema loaded: {self.schema.get('total_pages', 0)} pages, "
            f"{total_elements} elements"
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer question using hybrid filtering approach.

        Args:
            question: Natural language question

        Returns:
            Dictionary containing:
                - question: Original question
                - matched_elements: List of relevant elements with provenance
                - reasoning: Explanation of answer
                - filter_stats: Statistics about filtering process
        """
        if self.schema is None:
            raise ValueError("Schema not loaded. Call load_schema() first.")

        self.logger.info(f"Processing query: {question}")

        # Stage 1: Liberal field identification (Flash)
        relevant_fields = self._analyze_query_liberal(question)
        self.logger.info(f"Relevant fields identified: {relevant_fields}")

        # Stage 2: Deterministic filtering
        filtered_schema = self._filter_schema(relevant_fields)
        self.logger.info(
            f"Filtered {len(filtered_schema['elements'])} / "
            f"{self._count_total_elements()} elements"
        )

        # Stage 3: Reason with Pro
        answer = self._reason_with_pro(question, filtered_schema)

        return {
            "question": question,
            "matched_elements": answer.get("elements", []),
            "reasoning": answer.get("reasoning", ""),
            "answer_summary": answer.get("answer_summary", ""),
            "filter_stats": {
                "original_elements": self._count_total_elements(),
                "filtered_elements": len(filtered_schema["elements"]),
                "reduction_ratio": f"{self._count_total_elements() / max(len(filtered_schema['elements']), 1):.1f}x"
            }
        }

    def _analyze_query_liberal(self, question: str) -> Dict[str, Any]:
        """
        Use Flash to identify ALL potentially relevant schema fields (liberal).

        Args:
            question: User's natural language question

        Returns:
            Dictionary with relevant types and sub-documents
        """
        prompt = f"""Analyze this user question about an electronic health record (EHR):

QUESTION: {question}

Available semantic element types:
{', '.join(SEMANTIC_TYPES)}

Available sub-document types:
- laboratory_results: Lab tests, blood work, urinalysis
- medications: Medication lists, prescriptions, pharmacy records
- radiology_imaging: X-rays, CT scans, MRIs, ultrasounds
- vital_signs: Temperature, blood pressure, pulse, oxygen saturation
- progress_notes: Clinical notes, H&P, consultation notes
- procedures: Surgical/procedure notes
- orders: Physician orders, nursing orders
- immunizations: Vaccination records
- allergies: Allergy lists, adverse reactions
- problem_list: Diagnoses, active problems
- appointments: Visit summaries, encounters
- referrals: Specialist referrals
- patient_instructions: Discharge instructions, patient education

TASK: Identify ALL semantic types and sub-document types that are relevant to answering this question.
Err on the the liberal side - better to include an extra type or two than to miss important context.

Return your answer as a JSON object with this structure:
{{
    "relevant_types": ["type1", "type2", ...],
    "relevant_subdocs": ["subdoc1", "subdoc2", ...],
    "temporal_context": "recent|historical|all|null",
    "reasoning": "Brief explanation of your selections"
}}

Only return the JSON object, nothing else."""

        # Use structured output to FORCE valid JSON
        generation_config = GenerationConfig(
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "relevant_types": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "relevant_subdocs": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "temporal_context": {"type": "string"},
                    "reasoning": {"type": "string"}
                },
                "required": ["relevant_types", "relevant_subdocs", "temporal_context", "reasoning"]
            }
        )

        # Retry logic for JSON parsing errors (up to 7 attempts)
        max_retries = 7
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                response = self.flash_model.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                # Parse JSON response (guaranteed valid due to response_schema, but sometimes fails)
                result = json.loads(response.text)

                # Debug logging
                self.logger.info(f"Flash identified {len(result.get('relevant_types', []))} relevant types")
                self.logger.debug(f"Relevant types: {result.get('relevant_types', [])}")

                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < max_retries:
                    self.logger.warning(
                        f"JSON parsing error in Flash response (attempt {attempt}/{max_retries}): {e}. "
                        f"Retrying immediately..."
                    )
                    # No delay - retry immediately since errors are due to LLM output format, not transient issues
                else:
                    self.logger.error(
                        f"JSON parsing failed after {max_retries} attempts: {e}"
                    )
                    raise ValueError(
                        f"Failed to parse Flash model JSON response after {max_retries} retries: {e}"
                    ) from e

    def _filter_schema(self, relevant_fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministically filter schema to include ALL matching elements.

        Liberal filtering = include anything that might be relevant.

        Args:
            relevant_fields: Output from _analyze_query_liberal

        Returns:
            Filtered schema with subset of elements
        """
        relevant_types = set(relevant_fields.get("relevant_types", []))
        relevant_subdocs = set(relevant_fields.get("relevant_subdocs", []))

        # Always include these for context
        always_include_types = {"section_header", "patient_demographics"}
        relevant_types.update(always_include_types)

        filtered_elements = []

        # IMPORTANT: Filter by ELEMENT TYPE first, subdoc type is secondary
        # This handles cases where subdoc grouping is over-generalized
        if "sub_documents" in self.schema:
            for subdoc in self.schema.get("sub_documents", []):
                subdoc_type = subdoc.get("type")

                for page in subdoc.get("pages", []):
                    for element in page.get("elements", []):
                        # PRIMARY: Filter by element type (this is the key!)
                        if element.get("type") in relevant_types:
                            # Add page number and subdoc context
                            element_with_context = {
                                **element,
                                "page_number": page.get("page_number"),
                                "subdoc_type": subdoc_type,
                                "subdoc_title": subdoc.get("title")
                            }
                            filtered_elements.append(element_with_context)
                        # SECONDARY: Also check subdoc type if specified
                        elif relevant_subdocs and subdoc_type in relevant_subdocs:
                            # Include all elements from matching subdoc
                            element_with_context = {
                                **element,
                                "page_number": page.get("page_number"),
                                "subdoc_type": subdoc_type,
                                "subdoc_title": subdoc.get("title")
                            }
                            filtered_elements.append(element_with_context)

        else:
            # No sub-documents, filter pages directly
            for page in self.schema.get("pages", []):
                for element in page.get("elements", []):
                    if element.get("type") in relevant_types:
                        element_with_context = {
                            **element,
                            "page_number": page.get("page_number")
                        }
                        filtered_elements.append(element_with_context)

        self.logger.info(f"Filter found {len(filtered_elements)} matching elements")
        return {"elements": filtered_elements}

    def _reason_with_pro(
        self,
        question: str,
        filtered_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Gemini Pro to reason over filtered schema and answer question.

        Args:
            question: User's question
            filtered_schema: Filtered schema with relevant elements

        Returns:
            Dictionary with answer and reasoning
        """
        # Simplify schema - only include essential fields for reasoning
        # This prevents Pro from getting overwhelmed by verbose metadata
        simplified_elements = []
        for elem in filtered_schema.get("elements", []):
            simplified_elements.append({
                "element_id": elem.get("element_id"),
                "type": elem.get("type"),
                "content": elem.get("content"),  # This is the key field!
                "page_number": elem.get("page_number"),
                "bbox_pixel": elem.get("bbox_pixel"),
                "bbox_pdf": elem.get("bbox_pdf")
            })

        simplified_schema = {"elements": simplified_elements}
        schema_json = json.dumps(simplified_schema, indent=2, ensure_ascii=False)

        # Debug logging
        self.logger.info(f"Sending {len(simplified_elements)} elements to Pro for reasoning")
        self.logger.debug(f"Element types being sent: {[e.get('type') for e in simplified_elements]}")

        prompt = f"""You are analyzing an electronic health record (EHR) to answer a specific question.

Below is a filtered subset of the EHR's structured schema containing potentially relevant elements.

CRITICAL: The actual data is in the "content" field of each element. Read the content field carefully!

Each element has:
- element_id: Unique identifier
- type: Semantic element type (e.g., medication_table, lab_results_table)
- content: THE ACTUAL EXTRACTED TEXT/DATA - THIS IS WHERE THE ANSWER IS!
- page_number: Page where element appears
- bbox_pixel: Pixel coordinates for traceability
- bbox_pdf: PDF coordinates for traceability

FILTERED SCHEMA:
{schema_json}

USER QUESTION: {question}

IMPORTANT INSTRUCTIONS:
1. READ THE "content" FIELD of each element - that's where the actual data is!
2. For tables (medication_table, lab_results_table, etc.), the content field contains the full table data
3. Don't just look at element types - READ THE CONTENT!
4. Extract the actual answer from the content and present it clearly

TASK: Find ALL elements that answer the question and provide a clear natural language answer.

For "answer_summary", extract the ACTUAL DATA from the content fields and present it in a clear, readable format.

Examples:
- For medications: List each medication with dosage
- For lab results: List the key values
- For vital signs: List BP, HR, Temp, etc.

Return your answer as a JSON object:

{{
    "elements": [
        {{
            "element_id": "E_1234",
            "type": "medication_table",
            "content": "aspirin (aspirin 81 mg oral tablet) - Take 1 tab(s) oral daily",
            "page_number": 7,
            "bbox_pixel": [68, 95, 424, 110],
            "bbox_pdf": [24.48, 752.4, 152.64, 757.8],
            "relevance": "Contains medication list with dosages"
        }}
    ],
    "reasoning": "Brief explanation of where the answer was found",
    "answer_summary": "CLEAR NATURAL LANGUAGE ANSWER with the actual extracted data formatted for readability. For example: 'The patient is taking 8 medications: 1) Aspirin 81mg daily, 2) Atorvastatin 80mg nightly, 3) Cephalexin 500mg twice daily...'"
}}

CRITICAL: When including content from the schema in your response:
- ALL quotes in string values MUST be escaped as \\"
- ALL newlines in string values MUST be escaped as \\n
- ALL backslashes MUST be escaped as \\\\
- Ensure all string values are properly quoted and escaped

If no relevant elements are found, return empty elements list with explanation in reasoning.

Only return the JSON object, nothing else."""

        # Use structured output to FORCE valid JSON
        generation_config = GenerationConfig(
            temperature=0.2,
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "element_id": {"type": "string"},
                                "type": {"type": "string"},
                                "content": {"type": "string"},
                                "page_number": {"type": "integer"},
                                "bbox_pixel": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                },
                                "bbox_pdf": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                },
                                "relevance": {"type": "string"}
                            },
                            "required": ["element_id", "type", "content", "page_number"]
                        }
                    },
                    "reasoning": {"type": "string"},
                    "answer_summary": {"type": "string"}
                },
                "required": ["elements", "reasoning", "answer_summary"]
            }
        )

        # Retry logic for JSON parsing errors (up to 7 attempts)
        max_retries = 7
        last_error = None
        
        def repair_json(text: str) -> str:
            """
            Attempt to repair common JSON issues:
            - Unescaped quotes in strings
            - Unescaped newlines
            - Missing commas
            """
            # Try to fix unescaped quotes in string values
            # This is a simple heuristic - look for patterns like: "content": "text with "quotes" here"
            import re
            # Pattern to find string values that might have unescaped quotes
            # This is tricky - we'll try a simple approach: escape quotes that appear inside string values
            # But we need to be careful not to break valid JSON
            
            # For now, just return the text and let json.loads handle it
            # If it fails, we'll try to extract and repair
            return text
        
        for attempt in range(1, max_retries + 1):
            try:
                response = self.pro_model.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                # Try to parse JSON response
                try:
                    result = json.loads(response.text)
                    return result
                except json.JSONDecodeError as parse_error:
                    # If parsing fails, try to repair common issues
                    self.logger.debug(f"JSON parse error: {parse_error}")
                    self.logger.debug(f"Response text (first 500 chars): {response.text[:500]}")
                    
                    # Try to extract JSON from response if it's wrapped in markdown or other text
                    text = response.text.strip()
                    
                    # Remove markdown code blocks if present
                    if text.startswith("```"):
                        # Extract JSON from markdown code block
                        lines = text.split("\n")
                        json_lines = []
                        in_json = False
                        for line in lines:
                            if line.strip().startswith("```"):
                                in_json = not in_json
                                continue
                            if in_json:
                                json_lines.append(line)
                        text = "\n".join(json_lines)
                    
                    # Try parsing again
                    try:
                        result = json.loads(text)
                        return result
                    except json.JSONDecodeError:
                        # If still fails, re-raise the original error to trigger retry
                        raise parse_error
                
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < max_retries:
                    self.logger.warning(
                        f"JSON parsing error in Pro response (attempt {attempt}/{max_retries}): {e}. "
                        f"Retrying immediately..."
                    )
                    # No delay - retry immediately since errors are due to LLM output format, not transient issues
                else:
                    self.logger.error(
                        f"JSON parsing failed after {max_retries} attempts: {e}"
                    )
                    raise ValueError(
                        f"Failed to parse Pro model JSON response after {max_retries} retries: {e}"
                    ) from e

    def _count_total_elements(self) -> int:
        """Count total elements in schema."""
        if "sub_documents" in self.schema:
            return sum(
                len(page.get("elements", []))
                for subdoc in self.schema.get("sub_documents", [])
                for page in subdoc.get("pages", [])
            )
        else:
            return sum(
                len(page.get("elements", []))
                for page in self.schema.get("pages", [])
            )


def run_example_queries(agent: HybridQueryAgent) -> None:
    """
    Run example queries to demonstrate agent capabilities.

    Args:
        agent: Initialized HybridQueryAgent with loaded schema
    """
    example_queries = [
        "What were the patient's blood test results?",
        "What medications is the patient currently taking?",
        "Show me all vital signs readings",
        "What are the patient's active problems or diagnoses?",
        "Were there any radiology or imaging studies?",
        "What were the patient's most recent lab values?",
        "Is there any information about allergies?",
        "What procedures or surgeries were performed?"
    ]

    print("\n" + "=" * 80)
    print("EXAMPLE QUERIES")
    print("=" * 80)

    for i, query in enumerate(example_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 80)

        try:
            result = agent.query(query)

            print(f"Matched Elements: {len(result['matched_elements'])}")
            print(f"Filter Reduction: {result['filter_stats']['reduction_ratio']}")
            print(f"\nAnswer: {result.get('reasoning', 'No answer')}")

            if result['matched_elements']:
                print(f"\nSample Elements:")
                for elem in result['matched_elements'][:2]:  # Show first 2
                    print(f"  - {elem.get('type')}: {elem.get('content', '')[:100]}...")
                    print(f"    Page {elem.get('page_number')}, Bbox: {elem.get('bbox_pixel')}")

        except Exception as e:
            print(f"Error: {e}")

        print()

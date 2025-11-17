"""
Vision-Language Model (VLM) client for EHR document processing.

Provides interface to Google Gemini models via Vertex AI for layout detection,
text extraction, and semantic element classification.
"""

import json
import time
import logging
from datetime import datetime
from typing import Optional, Union, List, Dict, Any
from pathlib import Path

import numpy as np
from PIL import Image
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import vertexai

from ehrx.vlm.config import VLMConfig
from ehrx.vlm.models import (
    VLMRequest,
    VLMResponse,
    ElementDetection,
    ElementType,
    BoundingBox,
    ConfidenceScores,
    ClinicalMetadata,
    ProcessingMetadata,
    VLMError,
)
from ehrx.vlm.prompts import (
    SYSTEM_INSTRUCTION,
    build_element_extraction_prompt,
)


logger = logging.getLogger(__name__)


class VLMClient:
    """
    Client for Vision-Language Model processing using Google Gemini.

    Handles API communication, response parsing, error handling, and cost tracking.
    """

    def __init__(self, config: VLMConfig):
        """
        Initialize VLM client.

        Args:
            config: VLM configuration with API credentials and parameters
        """
        self.config = config
        self._model: Optional[GenerativeModel] = None
        self._total_cost_usd = 0.0
        self._request_count = 0

        # Initialize Vertex AI
        self._initialize_vertex_ai()

    def _initialize_vertex_ai(self) -> None:
        """Initialize Google Cloud Vertex AI client."""
        try:
            logger.info(
                f"Initializing Vertex AI (project={self.config.project_id}, "
                f"location={self.config.location})"
            )

            # Initialize Vertex AI SDK
            vertexai.init(
                project=self.config.project_id,
                location=self.config.location,
            )

            # Initialize generative model
            self._model = GenerativeModel(
                model_name=self.config.model_name,
                system_instruction=[SYSTEM_INSTRUCTION],
            )

            logger.info(f"VLM client initialized with model: {self.config.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise RuntimeError(f"Vertex AI initialization failed: {e}") from e

    def detect_elements(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        request: VLMRequest,
    ) -> VLMResponse:
        """
        Detect and extract elements from document page image.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            request: VLM request with context and parameters

        Returns:
            VLMResponse with detected elements and metadata

        Raises:
            RuntimeError: If VLM API call fails after retries
        """
        start_time = time.time()

        # Convert image to format suitable for Vertex AI
        image_part = self._prepare_image(image)

        # Store request context for potential retry with concise prompt
        self._last_request = request

        # Build prompt with context
        prompt = build_element_extraction_prompt(
            context=request.context,
            additional_instructions=None,
        )

        # Generate response with retry logic
        raw_response = self._generate_with_retry(
            image_part=image_part,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Parse response
        try:
            vlm_response = self._parse_response(
                raw_response=raw_response,
                request=request,
                latency_ms=latency_ms,
            )
            return vlm_response

        except Exception as e:
            logger.error(f"Failed to parse VLM response: {e}")
            logger.debug(f"Raw response: {raw_response}")

            # Return error response
            return self._create_error_response(
                error_message=f"Response parsing failed: {e}",
                raw_response=raw_response,
                latency_ms=latency_ms,
            )

    def _prepare_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Part:
        """
        Convert image to Vertex AI Part format.

        Args:
            image: Input image in various formats

        Returns:
            Part object suitable for Vertex AI API
        """
        # Handle file path (string or Path)
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Load image bytes
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            return Part.from_data(data=image_bytes, mime_type="image/png")

        # Handle PIL Image
        elif isinstance(image, Image.Image):
            # Convert PIL Image to bytes
            import io
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            return Part.from_data(data=image_bytes, mime_type="image/png")

        # Handle numpy array
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image then to bytes
            pil_image = Image.fromarray(image)
            import io
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            return Part.from_data(data=image_bytes, mime_type="image/png")

        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                "Expected str, Path, PIL.Image, or np.ndarray"
            )

    def _generate_with_retry(
        self,
        image_part: Part,
        prompt: str,
        max_tokens: int,
        temperature: float,
        concise_mode: bool = False,
    ) -> str:
        """
        Call Gemini API with retry logic for transient failures.

        Args:
            image_part: Image Part object
            prompt: Text prompt
            max_tokens: Maximum response tokens
            temperature: Sampling temperature

        Returns:
            Raw response text from VLM

        Raises:
            RuntimeError: If all retries fail
        """
        if self._model is None:
            raise RuntimeError("VLM client not initialized")

        # Build generation config with response schema for structured output
        generation_config = GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
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
                                "semantic_type": {"type": "string"},
                                "content": {"type": "string"},
                                "bbox": {
                                    "type": "object",
                                    "properties": {
                                        "x0": {"type": "number"},
                                        "y0": {"type": "number"},
                                        "x1": {"type": "number"},
                                        "y1": {"type": "number"}
                                    },
                                    "required": ["x0", "y0", "x1", "y1"]
                                },
                                "confidence_scores": {
                                    "type": "object",
                                    "properties": {
                                        "extraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "classification": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "clinical_context": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                                    },
                                    "required": ["extraction", "classification", "clinical_context"]
                                },
                                "clinical_metadata": {
                                    "type": "object",
                                    "properties": {
                                        "temporal_qualifier": {"type": "string"},
                                        "clinical_domain": {"type": "string"},
                                        "requires_validation": {"type": "boolean"},
                                        "validation_reason": {"type": "string"}
                                    }
                                }
                            },
                            "required": ["element_id", "semantic_type", "content", "bbox", "confidence_scores"]
                        }
                    },
                    "requires_human_review": {"type": "boolean"},
                    "review_reasons": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["elements", "requires_human_review", "review_reasons"]
            }
        )

        # Retry loop
        max_attempts = self.config.max_retries + 1 if self.config.enable_retry else 1
        last_error = None
        last_finish_reason = None

        for attempt in range(max_attempts):
            try:
                # Call Gemini API
                logger.debug(
                    f"Calling Gemini API (attempt {attempt + 1}/{max_attempts}, concise_mode={concise_mode})"
                )

                response = self._model.generate_content(
                    contents=[image_part, prompt],
                    generation_config=generation_config,
                )

                # Extract text from response
                if not response.candidates:
                    raise RuntimeError("No candidates in response")

                candidate = response.candidates[0]
                response_text = candidate.content.parts[0].text

                # Check if response was truncated due to token limit
                finish_reason = candidate.finish_reason
                if finish_reason and hasattr(finish_reason, 'name'):
                    finish_reason_str = finish_reason.name
                else:
                    finish_reason_str = str(finish_reason)

                last_finish_reason = finish_reason_str

                if "MAX_TOKENS" in finish_reason_str or "LENGTH" in finish_reason_str:
                    if not concise_mode and hasattr(self, '_last_request'):
                        # First time hitting MAX_TOKENS - retry with concise prompt
                        logger.warning(
                            f"Response truncated due to token limit (finish_reason: {finish_reason_str}). "
                            f"Retrying with concise mode..."
                        )
                        # Build concise prompt and retry
                        from ehrx.vlm.prompts import build_element_extraction_prompt_concise
                        concise_prompt = build_element_extraction_prompt_concise(
                            context=self._last_request.context
                        )
                        # Retry with concise prompt (recursive call)
                        return self._generate_with_retry(
                            image_part=image_part,
                            prompt=concise_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            concise_mode=True
                        )
                    else:
                        # Already in concise mode and still hitting limit - log and try to parse partial
                        logger.error(
                            f"Response still truncated in concise mode (finish_reason: {finish_reason_str}). "
                            f"Attempting to parse partial response. Consider page splitting for very dense pages."
                        )
                        # Continue to try parsing - might get partial results

                # Track usage
                self._request_count += 1

                # Estimate cost (approximate - actual token counts from response metadata)
                if self.config.enable_cost_tracking and hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    cost = self.config.estimate_cost(
                        input_tokens=usage.prompt_token_count,
                        output_tokens=usage.candidates_token_count,
                    )
                    self._total_cost_usd += cost
                    logger.debug(
                        f"API call cost: ${cost:.6f} "
                        f"(input: {usage.prompt_token_count}, "
                        f"output: {usage.candidates_token_count} tokens)"
                    )

                return response_text

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Gemini API call failed (attempt {attempt + 1}/{max_attempts}): {e}"
                )

                # Don't retry on last attempt
                if attempt < max_attempts - 1:
                    # Exponential backoff
                    delay = self.config.retry_delay_seconds * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed: {last_error}")

        # All retries failed
        raise RuntimeError(
            f"VLM API call failed after {max_attempts} attempts: {last_error}"
        ) from last_error

    def _parse_response(
        self,
        raw_response: str,
        request: VLMRequest,
        latency_ms: float,
    ) -> VLMResponse:
        """
        Parse raw VLM response into structured VLMResponse.

        Args:
            raw_response: Raw JSON string from VLM
            request: Original request for context
            latency_ms: API call latency in milliseconds

        Returns:
            Parsed VLMResponse with elements and metadata
        """
        # Parse JSON directly - guaranteed valid due to response_schema
        response_text = raw_response.strip()

        try:
            response_data = json.loads(response_text)
            logger.debug(f"Successfully parsed structured JSON response with {len(response_data.get('elements', []))} elements")
        except json.JSONDecodeError as e:
            # This can happen with truncated responses (MAX_TOKENS)
            logger.error(f"JSON parse failed despite structured output: {e}")
            
            # Try to extract partial elements from truncated JSON
            partial_elements = self._extract_partial_elements(response_text, e)
            
            if partial_elements:
                logger.warning(
                    f"Extracted {len(partial_elements)} partial elements from truncated response. "
                    f"Some content may be missing due to token limit."
                )
                # Create response with partial elements
                response_data = {
                    "elements": partial_elements,
                    "requires_human_review": True,
                    "review_reasons": [
                        "Response truncated due to MAX_TOKENS - partial extraction only"
                    ]
                }
            else:
                # Save for debugging if we can't extract anything
                if self.config.save_raw_responses:
                    debug_path = Path(self.config.raw_responses_dir or "./debug")
                    debug_path.mkdir(parents=True, exist_ok=True)
                    error_file = debug_path / f"error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(error_file, 'w') as f:
                        f.write(f"JSON Parse Error: {e}\n\n")
                        f.write("=== RAW RESPONSE ===\n")
                        f.write(raw_response)
                    logger.error(f"Saved problematic response to {error_file}")

                raise ValueError(f"Invalid JSON response: {e}") from e

        # Extract elements
        elements = []
        for elem_data in response_data.get("elements", []):
            element = self._parse_element(elem_data, request.context.page_number)
            elements.append(element)

        # Calculate overall confidence
        if elements:
            overall_confidence = sum(
                elem.confidence_scores.overall() for elem in elements
            ) / len(elements)
        else:
            overall_confidence = 0.0

        # Check if human review needed
        requires_review = response_data.get("requires_human_review", False)
        review_reasons = response_data.get("review_reasons", [])

        # Auto-flag if low confidence
        if overall_confidence < self.config.confidence_threshold_overall:
            requires_review = True
            review_reasons.append(
                f"Overall confidence {overall_confidence:.2f} below threshold "
                f"{self.config.confidence_threshold_overall}"
            )

        # Create processing metadata
        processing_metadata = ProcessingMetadata(
            model_name=self.config.model_name,
            processing_timestamp=datetime.utcnow().isoformat(),
            api_latency_ms=latency_ms,
            cost_estimate_usd=None,  # Updated after response
            human_reviewed=False,
            review_flags=[],
        )

        # Build response
        return VLMResponse(
            elements=elements,
            processing_metadata=processing_metadata,
            overall_confidence=overall_confidence,
            requires_human_review=requires_review,
            review_reasons=review_reasons,
            raw_response=raw_response if self.config.save_raw_responses else None,
        )

    def _extract_partial_elements(
        self,
        response_text: str,
        json_error: json.JSONDecodeError
    ) -> List[Dict[str, Any]]:
        """
        Attempt to extract partial elements from truncated JSON response.
        
        Args:
            response_text: Raw response text (may be truncated)
            json_error: The JSON decode error that occurred
            
        Returns:
            List of partial element dictionaries, or empty list if extraction fails
        """
        try:
            # Try to find the "elements" array in the response
            elements_start = response_text.find('"elements"')
            if elements_start == -1:
                elements_start = response_text.find("'elements'")
            
            if elements_start == -1:
                return []
            
            # Find the opening bracket of the elements array
            bracket_start = response_text.find('[', elements_start)
            if bracket_start == -1:
                return []
            
            # Extract the elements array portion (up to error position or end)
            error_pos = min(json_error.pos, len(response_text))
            partial_text = response_text[bracket_start:error_pos]
            
            # Try to find complete element objects by tracking brace depth
            elements = []
            current_pos = 1  # Skip opening bracket
            brace_count = 0
            element_start = None
            in_string = False
            escape_next = False
            
            while current_pos < len(partial_text):
                char = partial_text[current_pos]
                
                # Handle string escaping
                if escape_next:
                    escape_next = False
                    current_pos += 1
                    continue
                
                if char == '\\':
                    escape_next = True
                    current_pos += 1
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    current_pos += 1
                    continue
                
                if in_string:
                    current_pos += 1
                    continue
                
                # Track braces outside strings
                if char == '{':
                    if brace_count == 0:
                        element_start = current_pos
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and element_start is not None:
                        # Found a complete element object
                        element_text = partial_text[element_start:current_pos + 1]
                        try:
                            element_data = json.loads(element_text)
                            elements.append(element_data)
                        except json.JSONDecodeError:
                            # Skip this element if it's not valid JSON
                            pass
                        element_start = None
                
                current_pos += 1
            
            return elements
            
        except Exception as e:
            logger.debug(f"Failed to extract partial elements: {e}")
            return []

    def _parse_element(
        self,
        elem_data: Dict[str, Any],
        page_number: int,
    ) -> ElementDetection:
        """
        Parse single element from VLM response.

        Args:
            elem_data: Element data dictionary from VLM
            page_number: Page number (0-indexed)

        Returns:
            ElementDetection object
        """
        # Parse bounding box
        bbox_data = elem_data.get("bbox", {})
        bbox = BoundingBox(
            x0=bbox_data.get("x0", 0.0),
            y0=bbox_data.get("y0", 0.0),
            x1=bbox_data.get("x1", 0.0),
            y1=bbox_data.get("y1", 0.0),
        )

        # Parse confidence scores
        conf_data = elem_data.get("confidence_scores", {})
        confidence_scores = ConfidenceScores(
            extraction=conf_data.get("extraction", 0.5),
            classification=conf_data.get("classification", 0.5),
            clinical_context=conf_data.get("clinical_context", 0.5),
        )

        # Parse clinical metadata (optional)
        clinical_metadata = None
        if "clinical_metadata" in elem_data:
            cm_data = elem_data["clinical_metadata"]
            clinical_metadata = ClinicalMetadata(
                temporal_qualifier=cm_data.get("temporal_qualifier"),
                clinical_domain=cm_data.get("clinical_domain"),
                cross_references=cm_data.get("cross_references", []),
                requires_validation=cm_data.get("requires_validation", False),
                validation_reason=cm_data.get("validation_reason"),
            )

        # Parse element type
        semantic_type_str = elem_data.get("semantic_type", "uncategorized")
        try:
            semantic_type = ElementType(semantic_type_str)
        except ValueError:
            logger.warning(f"Unknown element type: {semantic_type_str}, using uncategorized")
            semantic_type = ElementType.UNCATEGORIZED

        # Build element
        element = ElementDetection(
            element_id=elem_data.get("element_id", "E_unknown"),
            semantic_type=semantic_type,
            bbox=bbox,
            content=elem_data.get("content", ""),
            confidence_scores=confidence_scores,
            clinical_metadata=clinical_metadata,
            processing_metadata=None,  # Added at response level
            page=page_number,
            column=None,  # Will be assigned by column detection
            z_order=None,  # Will be assigned by global ordering
        )

        return element

    def _create_error_response(
        self,
        error_message: str,
        raw_response: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> VLMResponse:
        """
        Create error response when VLM processing fails.

        Args:
            error_message: Description of error
            raw_response: Raw response text if available
            latency_ms: API latency before failure

        Returns:
            VLMResponse with error indicators
        """
        processing_metadata = ProcessingMetadata(
            model_name=self.config.model_name,
            processing_timestamp=datetime.utcnow().isoformat(),
            api_latency_ms=latency_ms,
            cost_estimate_usd=None,
            human_reviewed=False,
            review_flags=["error"],
        )

        return VLMResponse(
            elements=[],
            processing_metadata=processing_metadata,
            overall_confidence=0.0,
            requires_human_review=True,
            review_reasons=[error_message],
            raw_response=raw_response,
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get client usage statistics.

        Returns:
            Dictionary with request count, total cost, and other metrics
        """
        return {
            "request_count": self._request_count,
            "total_cost_usd": self._total_cost_usd,
            "average_cost_per_request": (
                self._total_cost_usd / self._request_count
                if self._request_count > 0 else 0.0
            ),
            "model_name": self.config.model_name,
            "project_id": self.config.project_id,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._request_count = 0
        self._total_cost_usd = 0.0
        logger.info("VLM client statistics reset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VLMClient(model={self.config.model_name}, "
            f"requests={self._request_count}, cost=${self._total_cost_usd:.4f})"
        )

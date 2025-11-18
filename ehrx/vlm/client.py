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
        
        If MAX_TOKENS is hit, automatically splits page into regions and processes recursively.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            request: VLM request with context and parameters

        Returns:
            VLMResponse with detected elements and metadata

        Raises:
            RuntimeError: If VLM API call fails after retries
        """
        return self._detect_elements_with_splitting(
            image=image,
            request=request,
            region_offset_x=0,
            region_offset_y=0,
            recursion_depth=0
        )

    def _detect_elements_with_splitting(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        request: VLMRequest,
        region_offset_x: int = 0,
        region_offset_y: int = 0,
        recursion_depth: int = 0
    ) -> VLMResponse:
        """
        Detect elements with automatic region splitting on MAX_TOKENS.
        
        Args:
            image: Input image
            request: VLM request
            region_offset_x: X offset for this region (for bounding box adjustment)
            region_offset_y: Y offset for this region (for bounding box adjustment)
            recursion_depth: Current recursion depth (for logging)
        """
        start_time = time.time()

        # Convert image to format suitable for Vertex AI
        image_part = self._prepare_image(image)

        # Build prompt with context
        prompt = build_element_extraction_prompt(
            context=request.context,
            additional_instructions=None,
        )

        # Generate response with retry logic
        raw_response = None
        finish_reason = None
        try:
            raw_response, finish_reason = self._generate_with_retry(
                image_part=image_part,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except ValueError as e:
            # JSON parse error - check if it's due to MAX_TOKENS truncation
            error_str = str(e).lower()
            if "max_tokens" in error_str or "truncated" in error_str:
                # Try region splitting
                logger.warning(
                    f"MAX_TOKENS hit at recursion depth {recursion_depth}. "
                    f"Splitting page into regions..."
                )
                return self._process_with_region_splitting(
                    image=image,
                    request=request,
                    region_offset_x=region_offset_x,
                    region_offset_y=region_offset_y,
                    recursion_depth=recursion_depth
                )
            raise

        # Check finish reason for MAX_TOKENS
        if finish_reason and ("MAX_TOKENS" in str(finish_reason) or "LENGTH" in str(finish_reason)):
            logger.warning(
                f"MAX_TOKENS detected at recursion depth {recursion_depth}. "
                f"Splitting page into regions..."
            )
            return self._process_with_region_splitting(
                image=image,
                request=request,
                region_offset_x=region_offset_x,
                region_offset_y=region_offset_y,
                recursion_depth=recursion_depth
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
            
            # Adjust bounding boxes for region offset
            if region_offset_x != 0 or region_offset_y != 0:
                for element in vlm_response.elements:
                    if element.bbox:
                        element.bbox.x0 += region_offset_x
                        element.bbox.y0 += region_offset_y
                        element.bbox.x1 += region_offset_x
                        element.bbox.y1 += region_offset_y
            
            return vlm_response

        except (ValueError, json.JSONDecodeError) as e:
            # Check if this is a JSON parse error that might be due to MAX_TOKENS truncation
            # If finish_reason indicates MAX_TOKENS, try region splitting
            if finish_reason and ("MAX_TOKENS" in str(finish_reason) or "LENGTH" in str(finish_reason)):
                # MAX_TOKENS was detected - try region splitting
                logger.warning(
                    f"JSON parse failed due to MAX_TOKENS truncation at recursion depth {recursion_depth}. "
                    f"Splitting page into regions..."
                )
                return self._process_with_region_splitting(
                    image=image,
                    request=request,
                    region_offset_x=region_offset_x,
                    region_offset_y=region_offset_y,
                    recursion_depth=recursion_depth
                )
            
            logger.error(f"Failed to parse VLM response: {e}")
            logger.debug(f"Raw response: {raw_response[:500] if raw_response else 'None'}...")  # Log first 500 chars

            # Return error response
            return self._create_error_response(
                error_message=f"Response parsing failed: {e}",
                raw_response=raw_response,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.error(f"Failed to parse VLM response: {e}")
            logger.debug(f"Raw response: {raw_response[:500] if raw_response else 'None'}...")

            # Return error response
            return self._create_error_response(
                error_message=f"Response parsing failed: {e}",
                raw_response=raw_response,
                latency_ms=latency_ms,
            )

    def _process_with_region_splitting(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        request: VLMRequest,
        region_offset_x: int,
        region_offset_y: int,
        recursion_depth: int
    ) -> VLMResponse:
        """
        Split image into regions and process recursively.
        
        Splits page vertically (top/bottom) or horizontally (left/right) and processes each region separately.
        If a region also hits MAX_TOKENS, recursively splits it further.
        
        Args:
            image: Input image
            request: VLM request
            region_offset_x: X offset for this region
            region_offset_y: Y offset for this region
            recursion_depth: Current recursion depth
            
        Returns:
            Merged VLMResponse from all regions
        """
        # Safety check: prevent infinite recursion
        MAX_RECURSION_DEPTH = 10
        if recursion_depth >= MAX_RECURSION_DEPTH:
            logger.error(
                f"Maximum recursion depth ({MAX_RECURSION_DEPTH}) reached. "
                f"Returning error response for region."
            )
            return self._create_error_response(
                error_message=f"Maximum recursion depth reached - region too dense to process",
                raw_response=None,
                latency_ms=0.0,
            )
        
        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        width, height = pil_image.size
        
        # Safety check: if region is too small, don't split further
        MIN_REGION_SIZE = 100  # pixels
        if width < MIN_REGION_SIZE or height < MIN_REGION_SIZE:
            logger.warning(
                f"Region too small to split further ({width}x{height}). "
                f"Attempting to process as-is (may fail)."
            )
            # Try processing the small region anyway - might work if content is sparse
            return self._detect_elements_with_splitting(
                image=image,
                request=request,
                region_offset_x=region_offset_x,
                region_offset_y=region_offset_y,
                recursion_depth=recursion_depth + 1
            )
        
        # Split vertically (top/bottom) - this is usually better for documents
        # If page is wider than tall, split horizontally instead
        if width > height:
            # Split horizontally (left/right)
            split_coord = width // 2
            region1 = pil_image.crop((0, 0, split_coord, height))
            region2 = pil_image.crop((split_coord, 0, width, height))
            region1_offset_x = region_offset_x
            region1_offset_y = region_offset_y
            region2_offset_x = region_offset_x + split_coord
            region2_offset_y = region_offset_y
            logger.info(
                f"Recursion depth {recursion_depth}: Splitting horizontally at x={split_coord} "
                f"(left: 0-{split_coord}, right: {split_coord}-{width})"
            )
        else:
            # Split vertically (top/bottom)
            split_coord = height // 2
            region1 = pil_image.crop((0, 0, width, split_coord))
            region2 = pil_image.crop((0, split_coord, width, height))
            region1_offset_x = region_offset_x
            region1_offset_y = region_offset_y
            region2_offset_x = region_offset_x
            region2_offset_y = region_offset_y + split_coord
            logger.info(
                f"Recursion depth {recursion_depth}: Splitting vertically at y={split_coord} "
                f"(top: 0-{split_coord}, bottom: {split_coord}-{height})"
            )
        
        # Process each region recursively
        logger.info(f"Processing region 1 (offset: {region1_offset_x}, {region1_offset_y})...")
        response1 = self._detect_elements_with_splitting(
            image=region1,
            request=request,
            region_offset_x=region1_offset_x,
            region_offset_y=region1_offset_y,
            recursion_depth=recursion_depth + 1
        )
        
        logger.info(f"Processing region 2 (offset: {region2_offset_x}, {region2_offset_y})...")
        response2 = self._detect_elements_with_splitting(
            image=region2,
            request=request,
            region_offset_x=region2_offset_x,
            region_offset_y=region2_offset_y,
            recursion_depth=recursion_depth + 1
        )
        
        # Merge responses
        merged_response = self._merge_region_responses(response1, response2)
        
        logger.info(
            f"Recursion depth {recursion_depth}: Merged {len(response1.elements)} + "
            f"{len(response2.elements)} = {len(merged_response.elements)} elements"
        )
        
        return merged_response

    def _merge_region_responses(
        self,
        response1: VLMResponse,
        response2: VLMResponse
    ) -> VLMResponse:
        """
        Merge two VLM responses from different regions into a single response.
        
        Args:
            response1: First region response
            response2: Second region response
            
        Returns:
            Merged VLMResponse
        """
        from ehrx.vlm.models import ProcessingMetadata
        from datetime import datetime
        
        # Combine elements
        merged_elements = list(response1.elements) + list(response2.elements)
        
        # Recalculate overall confidence
        if merged_elements:
            overall_confidence = sum(
                elem.confidence_scores.overall() for elem in merged_elements
            ) / len(merged_elements)
        else:
            overall_confidence = 0.0
        
        # Combine review reasons
        review_reasons = list(response1.review_reasons) + list(response2.review_reasons)
        if "region_split" not in str(review_reasons).lower():
            review_reasons.append("Page split into regions due to MAX_TOKENS")
        
        # Determine if review needed
        requires_review = (
            response1.requires_human_review or 
            response2.requires_human_review or
            overall_confidence < self.config.confidence_threshold_overall
        )
        
        # Merge processing metadata
        merged_metadata = ProcessingMetadata(
            model_name=self.config.model_name,
            processing_timestamp=datetime.utcnow().isoformat(),
            api_latency_ms=(
                (response1.processing_metadata.api_latency_ms or 0) +
                (response2.processing_metadata.api_latency_ms or 0)
            ),
            cost_estimate_usd=None,  # Will be calculated at higher level
            human_reviewed=False,
            review_flags=list(response1.processing_metadata.review_flags) + 
                        list(response2.processing_metadata.review_flags)
        )
        
        return VLMResponse(
            elements=merged_elements,
            processing_metadata=merged_metadata,
            overall_confidence=overall_confidence,
            requires_human_review=requires_review,
            review_reasons=review_reasons,
            raw_response=None  # Don't store raw responses for merged results
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
    ) -> tuple[str, Optional[str]]:
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

        for attempt in range(max_attempts):
            try:
                # Call Gemini API
                logger.debug(
                    f"Calling Gemini API (attempt {attempt + 1}/{max_attempts})"
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

                return response_text, finish_reason_str

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
            # This should rarely happen with structured output, but handle gracefully
            logger.error(f"JSON parse failed despite structured output: {e}")

            # Save for debugging
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

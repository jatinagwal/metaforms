"""LLM Manager Module - Handles Gemini API interactions for Extraction."""

import os
import json
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from google import genai
from google.genai import types

from .schema_analyzer import SchemaAnalyzer, SchemaMetrics, ProcessingStrategy
from .text_processor import TextProcessor, ProcessedText


@dataclass
class ConversionResult:
    json_output: Dict[str, Any]
    confidence_score: float
    processing_time: float
    api_calls_made: int
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    model_name: str = "gemini-2.5-pro"
    temperature: float = 0.0  # Use 0 temperature for maximum consistency
    max_output_tokens: int = None  # Set to None to use maximum allowed by model
    api_key: Optional[str] = None
    timeout: int = 600  # Increase timeout for large inputs (10 minutes)
    use_streaming: bool = False
    thinking_budget: int = 20000  # Reduced from 30000 for large inputs
    
    def adjust_for_large_input(self, input_size: int) -> 'LLMConfig':
        """Return a config adjusted for large input handling."""
        if input_size > 50000:  # If input > 50KB
            return LLMConfig(
                model_name=self.model_name,
                temperature=self.temperature,
                max_output_tokens=None,  # No limit for large inputs
                api_key=self.api_key,
                timeout=900,  # 15 minutes for very large inputs
                use_streaming=True,  # Use streaming for large inputs
                thinking_budget=10000  # Further reduced thinking budget
            )
        elif input_size > 20000:  # If input > 20KB
            return LLMConfig(
                model_name=self.model_name,
                temperature=self.temperature,
                max_output_tokens=None,  # No limit for medium inputs
                api_key=self.api_key,
                timeout=self.timeout,
                use_streaming=True,  # Use streaming for medium inputs
                thinking_budget=15000  # Slightly reduced thinking budget
            )
        else:
            return self


class LLMManager:
    """Manages LLM interactions for structured JSON generation."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM manager with configuration.
        
        Args:
            config: LLM configuration object
        """
        self.config = config or LLMConfig()
        self.schema_analyzer = SchemaAnalyzer()
        self.text_processor = TextProcessor()
        
        # Initialize Gemini API Client
        api_key = self.config.api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it in config.")
        
        self.client = genai.Client(api_key=api_key)
    
    async def convert_to_json(self, 
                            input_text: str, 
                            schema: Dict[str, Any],
                            input_format: Optional[str] = None) -> ConversionResult:
        """
        Convert unstructured text to structured JSON following the provided schema.
        
        Args:
            input_text: Raw input text or data
            schema: JSON schema to follow
            input_format: Input format hint ('text', 'pdf', 'csv', etc.)
            
        Returns:
            ConversionResult with generated JSON and metadata
        """
        start_time = time.time()
        
        # Adjust configuration for large inputs
        input_size = len(input_text)
        if input_size > 20000:
            print(f"📏 Large input detected ({input_size:,} chars) - adjusting configuration")
            original_config = self.config
            self.config = self.config.adjust_for_large_input(input_size)
            print(f"   Adjusted timeout: {self.config.timeout}s")
            print(f"   Adjusted streaming: {self.config.use_streaming}")
            print(f"   Adjusted thinking budget: {self.config.thinking_budget}")
        
        try:
            # Analyze schema complexity
            schema_metrics, strategy = self.schema_analyzer.analyze_schema(schema)
            
            # Process input text
            processed_text = self.text_processor.process_input(input_text, input_format)
            
            # Generate JSON based on processing strategy
            if strategy.use_multi_pass:
                result = await self._multi_pass_conversion(processed_text, schema, schema_metrics, strategy)
            else:
                result = await self._single_pass_conversion(processed_text, schema, schema_metrics, strategy)
            
            # Update timing and metadata
            result.processing_time = time.time() - start_time
            result.metadata.update({
                "schema_complexity": schema_metrics.complexity_level.value,
                "processing_strategy": {
                    "multi_pass": strategy.use_multi_pass,
                    "estimated_calls": strategy.estimated_calls,
                    "actual_calls": result.api_calls_made
                },
                "input_processing": self.text_processor.get_processing_stats(processed_text),
                "input_size": input_size,
                "config_adjusted": input_size > 20000
            })
            
            return result
            
        finally:
            # Restore original configuration if it was adjusted
            if input_size > 20000:
                self.config = original_config
    
    async def _single_pass_conversion(self, 
                                    processed_text: ProcessedText,
                                    schema: Dict[str, Any],
                                    metrics: SchemaMetrics,
                                    strategy: ProcessingStrategy) -> ConversionResult:
        """Perform single-pass conversion for simpler schemas."""
        
        user_prompt, system_instruction = self._build_conversion_prompt(processed_text.content, schema, metrics, is_first_pass=True)
        
        try:
            response_text = await self._call_gemini(user_prompt, system_instruction)
            json_output = self._extract_json_from_response(response_text)
            
            # Validate against schema
            validation_errors = self._validate_json_against_schema(json_output, schema)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                json_output, schema, validation_errors, processed_text.confidence
            )
            
            return ConversionResult(
                json_output=json_output,
                confidence_score=confidence,
                processing_time=0,  # Will be updated later
                api_calls_made=1,
                validation_errors=validation_errors
            )
            
        except Exception as e:
            # If JSON extraction fails completely, try to create a minimal valid structure
            error_msg = str(e)
            fallback_json = self._create_fallback_json(schema, f"JSON extraction failed: {error_msg}")
            
            return ConversionResult(
                json_output=fallback_json,
                confidence_score=0.0,
                processing_time=0,
                api_calls_made=1,
                validation_errors=[f"JSON Extraction Error: {error_msg}"]
            )
    
    async def _multi_pass_conversion(self,
                                   processed_text: ProcessedText,
                                   schema: Dict[str, Any],
                                   metrics: SchemaMetrics,
                                   strategy: ProcessingStrategy) -> ConversionResult:
        """Perform multi-pass conversion for complex schemas."""
        
        api_calls_made = 0
        accumulated_json = {}
        all_validation_errors = []
        
        # If schema chunking is enabled, process schema in parts
        if strategy.chunk_schema:
            schema_chunks = self.schema_analyzer.get_schema_chunks(schema)
        else:
            schema_chunks = [schema]
        
        # Process each chunk
        for i, schema_chunk in enumerate(schema_chunks):
            user_prompt, system_instruction = self._build_conversion_prompt(
                processed_text.content, 
                schema_chunk, 
                metrics,
                is_first_pass=(i == 0),
                existing_json=accumulated_json if i > 0 else None
            )
            
            try:
                response_text = await self._call_gemini(user_prompt, system_instruction)
                chunk_json = self._extract_json_from_response(response_text)
                api_calls_made += 1
                
                # Merge with accumulated JSON
                accumulated_json = self._merge_json_outputs(accumulated_json, chunk_json)
                
            except Exception as e:
                error_msg = f"Chunk {i+1} error: {str(e)}"
                all_validation_errors.append(error_msg)
                api_calls_made += 1
                
                # Try to create a partial fallback for this chunk
                try:
                    chunk_fallback = self._create_fallback_json(schema_chunk, f"Chunk processing failed: {str(e)}")
                    accumulated_json = self._merge_json_outputs(accumulated_json, chunk_fallback)
                except Exception:
                    # If even fallback fails, continue with next chunk
                    continue
        
        # Refinement pass if validation loop is enabled
        if strategy.use_validation_loop:
            validation_errors = self._validate_json_against_schema(accumulated_json, schema)
            
            if validation_errors and api_calls_made < strategy.max_retries:
                refinement_result = await self._refinement_pass(
                    accumulated_json, 
                    schema, 
                    validation_errors, 
                    processed_text.content,
                    strategy.max_retries - api_calls_made
                )
                accumulated_json = refinement_result["json_output"]
                api_calls_made += refinement_result["api_calls"]
                all_validation_errors.extend(refinement_result["validation_errors"])
        
        # Final validation
        final_validation_errors = self._validate_json_against_schema(accumulated_json, schema)
        all_validation_errors.extend(final_validation_errors)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(
            accumulated_json, schema, final_validation_errors, processed_text.confidence
        )
        
        return ConversionResult(
            json_output=accumulated_json,
            confidence_score=confidence,
            processing_time=0,  # Will be updated later
            api_calls_made=api_calls_made,
            validation_errors=all_validation_errors
        )
    
    async def _refinement_pass(self,
                             current_json: Dict[str, Any],
                             schema: Dict[str, Any],
                             validation_errors: List[str],
                             original_text: str,
                             max_attempts: int) -> Dict[str, Any]:
        """Perform refinement pass to fix validation errors."""
        
        api_calls = 0
        refined_json = current_json.copy()
        remaining_errors = validation_errors.copy()
        
        for attempt in range(max_attempts):
            if not remaining_errors:
                break
            
            user_prompt, system_instruction = self._build_refinement_prompt(
                refined_json, schema, remaining_errors, original_text
            )
            
            try:
                response_text = await self._call_gemini(user_prompt, system_instruction)
                refined_json = self._extract_json_from_response(response_text)
                api_calls += 1
                
                # Re-validate
                remaining_errors = self._validate_json_against_schema(refined_json, schema)
                
            except Exception as e:
                remaining_errors.append(f"Refinement attempt {attempt+1} failed: {str(e)}")
                api_calls += 1
                break
        
        return {
            "json_output": refined_json,
            "api_calls": api_calls,
            "validation_errors": remaining_errors
        }
    
    def _build_conversion_prompt(self, 
                               input_text: str, 
                               schema: Dict[str, Any],
                               metrics: SchemaMetrics,
                               is_first_pass: bool = True,
                               existing_json: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Build prompt for JSON conversion, returning (user_prompt, system_instruction)."""
        
        schema_str = json.dumps(schema, indent=2)
        
        # Extract required fields for emphasis
        required_fields = self._extract_required_fields(schema)
        enum_fields = self._extract_enum_fields(schema)
        
        system_instruction = """You are an expert data extraction and conversion system. Your task is to convert unstructured text into structured JSON that strictly follows the provided JSON schema.

CRITICAL REQUIREMENTS:
1. Extract ALL relevant information from the input text
2. Structure it according to the provided JSON schema EXACTLY
3. Use ONLY the field names and types specified in the schema
4. For enum fields, use ONLY the values listed in the enum - never create new values
5. ALL required fields MUST be present in the output - use appropriate default values if information is missing
6. For missing optional information, omit the field entirely rather than using null
7. Maintain data accuracy - don't hallucinate information not present in the text
8. For dates, use the exact format patterns specified in the schema
9. For nested objects, ensure all their required fields are included
10. Numbers should be actual numbers (not strings) when schema specifies "type": "integer" or "type": "number"

JSON FORMATTING RULES:
- All property names must be in double quotes
- All string values must be in double quotes
- No trailing commas
- Proper nesting and bracket matching
- Use proper JSON data types (string, number, boolean, array, object, null)
- Return a SINGLE JSON OBJECT (not an array of objects)
- The root level must be an object starting with { and ending with }

CRITICAL OUTPUT FORMAT:
- Return ONLY a single valid JSON object
- Start with { and end with }
- Do NOT wrap the object in an array
- No explanations, no additional text, no markdown formatting
- No ```json``` code blocks"""

        user_prompt = f"""**INPUT TEXT:**
{input_text}

**TARGET JSON SCHEMA:**
{schema_str}

**SCHEMA ANALYSIS:**
- Complexity Level: {metrics.complexity_level.value}
- Total Fields: {metrics.total_fields}
- Nested Objects: {metrics.nested_objects}
- Maximum Nesting Depth: {metrics.max_nesting_depth}
- Required Fields: {metrics.required_fields}"""

        if required_fields:
            user_prompt += f"\n\n**REQUIRED FIELDS (must be present):**\n"
            for field_path, field_type in required_fields:
                user_prompt += f"- {field_path}: {field_type}\n"

        if enum_fields:
            user_prompt += f"\n**ENUM CONSTRAINTS (use only these exact values):**\n"
            for field_path, enum_values in enum_fields:
                user_prompt += f"- {field_path}: {enum_values}\n"

        if not is_first_pass and existing_json:
            user_prompt += f"""

**EXISTING JSON (to merge with):**
{json.dumps(existing_json, indent=2)}

Please merge the new extraction with the existing JSON, avoiding duplication and maintaining consistency."""

        user_prompt += "\n\n**IMPORTANT REMINDERS:**"
        user_prompt += "\n- Include ALL required fields even if you need to use reasonable defaults"
        user_prompt += "\n- Follow enum constraints exactly - never create new enum values"
        user_prompt += "\n- Return only valid JSON with proper formatting"
        user_prompt += "\n- Use correct data types as specified in the schema"

        user_prompt += "\n\nGenerate the complete JSON output now:"

        return user_prompt, system_instruction
    
    def _extract_required_fields(self, schema: Dict[str, Any], path: str = "") -> List[Tuple[str, str]]:
        """Extract all required fields from schema with their types."""
        required_fields = []
        
        def extract_recursive(obj: Dict[str, Any], current_path: str = "") -> None:
            if isinstance(obj, dict):
                if obj.get("type") == "object" and "properties" in obj:
                    # Check required fields at this level
                    required = obj.get("required", [])
                    for req_field in required:
                        if req_field in obj["properties"]:
                            field_path = f"{current_path}.{req_field}" if current_path else req_field
                            field_type = obj["properties"][req_field].get("type", "unknown")
                            required_fields.append((field_path, field_type))
                    
                    # Recurse into properties
                    for prop_name, prop_def in obj["properties"].items():
                        new_path = f"{current_path}.{prop_name}" if current_path else prop_name
                        extract_recursive(prop_def, new_path)
                
                # Handle arrays
                elif obj.get("type") == "array" and "items" in obj:
                    extract_recursive(obj["items"], current_path)
        
        extract_recursive(schema, path)
        return required_fields
    
    def _extract_enum_fields(self, schema: Dict[str, Any], path: str = "") -> List[Tuple[str, List[str]]]:
        """Extract all enum fields from schema with their allowed values."""
        enum_fields = []
        
        def extract_recursive(obj: Dict[str, Any], current_path: str = "") -> None:
            if isinstance(obj, dict):
                if "enum" in obj:
                    enum_fields.append((current_path, obj["enum"]))
                
                if obj.get("type") == "object" and "properties" in obj:
                    for prop_name, prop_def in obj["properties"].items():
                        new_path = f"{current_path}.{prop_name}" if current_path else prop_name
                        extract_recursive(prop_def, new_path)
                
                elif obj.get("type") == "array" and "items" in obj:
                    extract_recursive(obj["items"], current_path)
        
        extract_recursive(schema, path)
        return enum_fields
    
    def _build_refinement_prompt(self,
                               current_json: Dict[str, Any],
                               schema: Dict[str, Any],
                               validation_errors: List[str],
                               original_text: str) -> Tuple[str, str]:
        """Build prompt for JSON refinement, returning (user_prompt, system_instruction)."""
        
        required_fields = self._extract_required_fields(schema)
        enum_fields = self._extract_enum_fields(schema)
        
        system_instruction = """You are fixing a JSON output that has validation errors against a schema.

CRITICAL FIXING REQUIREMENTS:
1. Fix ALL validation errors while preserving correct data
2. Ensure the output strictly conforms to the schema
3. Don't lose any correctly extracted information
4. ALL required fields must be present
5. Use only allowed enum values
6. Return properly formatted, parseable JSON only

JSON FORMATTING REQUIREMENTS:
- All property names in double quotes
- All string values in double quotes  
- No trailing commas
- Proper bracket matching
- Correct data types (string, number, boolean, array, object, null)
- Return a SINGLE JSON OBJECT (not an array of objects)
- The root level must be an object starting with { and ending with }

CRITICAL OUTPUT FORMAT:
- Return ONLY the corrected single JSON object
- Start with { and end with }
- Do NOT wrap the object in an array
- No explanations, no additional text, no markdown formatting
- No ```json``` code blocks"""
        
        user_prompt = f"""**ORIGINAL INPUT TEXT:**
{original_text}

**TARGET JSON SCHEMA:**
{json.dumps(schema, indent=2)}

**CURRENT JSON (has errors):**
{json.dumps(current_json, indent=2)}

**VALIDATION ERRORS TO FIX:**
{chr(10).join(f"- {error}" for error in validation_errors)}"""

        if required_fields:
            user_prompt += f"\n\n**ENSURE THESE REQUIRED FIELDS ARE PRESENT:**"
            for field_path, field_type in required_fields:
                user_prompt += f"\n- {field_path}: {field_type}"

        if enum_fields:
            user_prompt += f"\n\n**ENUM CONSTRAINTS (use only these values):**"
            for field_path, enum_values in enum_fields:
                user_prompt += f"\n- {field_path}: {enum_values}"

        user_prompt += "\n\n**FIXING INSTRUCTIONS:**"
        user_prompt += "\n- Add any missing required fields with appropriate values from the original text"
        user_prompt += "\n- Fix any enum fields to use only allowed values"
        user_prompt += "\n- Ensure proper JSON formatting (quotes, commas, brackets)"
        user_prompt += "\n- Preserve all correct information already extracted"

        user_prompt += "\n\nGenerate the corrected JSON output now:"

        return user_prompt, system_instruction
    
    async def _call_gemini(self, user_prompt: str, system_instruction: str) -> str:
        """Call Gemini API with error handling and retries."""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Log input size for debugging
                prompt_size = len(user_prompt)
                system_size = len(system_instruction)
                total_size = prompt_size + system_size
                
                print(f"🔍 API Call Debug - Attempt {attempt + 1}:")
                print(f"   Input size: {total_size:,} chars ({prompt_size:,} prompt + {system_size:,} system)")
                print(f"   Model: {self.config.model_name}")
                print(f"   Max output tokens: {self.config.max_output_tokens if self.config.max_output_tokens is not None else 'No limit'}")
                print(f"   Temperature: {self.config.temperature}")
                
                # Prepare the content structure
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=user_prompt),
                        ],
                    ),
                ]
                
                # Prepare the generation config
                thinking_config = types.ThinkingConfig(thinking_budget=self.config.thinking_budget)
                
                # Build generation config with or without max_output_tokens
                generation_config_args = {
                    "thinking_config": thinking_config,
                    "response_mime_type": "application/json",
                    "system_instruction": [types.Part.from_text(text=system_instruction)],
                    "temperature": self.config.temperature,
                }
                
                # Only add max_output_tokens if it's not None
                if self.config.max_output_tokens is not None:
                    generation_config_args["max_output_tokens"] = self.config.max_output_tokens
                
                generate_content_config = types.GenerateContentConfig(**generation_config_args)
                
                # Make the API call
                if self.config.use_streaming:
                    # Use streaming for better responsiveness
                    response_text = ""
                    chunk_count = 0
                    for chunk in self.client.models.generate_content_stream(
                        model=self.config.model_name,
                        contents=contents,
                        config=generate_content_config,
                    ):
                        if chunk.text:
                            response_text += chunk.text
                            chunk_count += 1
                    
                    print(f"   Streaming response: {chunk_count} chunks, {len(response_text)} chars")
                    
                    if response_text:
                        return response_text
                    else:
                        raise Exception(f"Empty response from Gemini API (streaming) - received {chunk_count} chunks but no text content")
                        
                else:
                    # Use non-streaming for simpler handling
                    response = self.client.models.generate_content(
                        model=self.config.model_name,
                        contents=contents,
                        config=generate_content_config,
                    )
                    
                    print(f"   Non-streaming response received")
                    print(f"   Response object type: {type(response)}")
                    print(f"   Has text attribute: {hasattr(response, 'text')}")
                    
                    if hasattr(response, 'text') and response.text:
                        print(f"   Response length: {len(response.text)} chars")
                        return response.text
                    elif hasattr(response, 'candidates') and response.candidates:
                        # Try to extract from candidates
                        print(f"   Found {len(response.candidates)} candidates")
                        for i, candidate in enumerate(response.candidates):
                            print(f"   Candidate {i}: {type(candidate)}")
                            if hasattr(candidate, 'content') and candidate.content:
                                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                    for j, part in enumerate(candidate.content.parts):
                                        print(f"     Part {j}: {type(part)}")
                                        if hasattr(part, 'text') and part.text:
                                            print(f"     Part text length: {len(part.text)}")
                                            return part.text
                    else:
                        print(f"   Response details: {dir(response) if response else 'None'}")
                        raise Exception(f"Empty response from Gemini API - Response object exists but no text content found. Input size: {total_size:,} chars")
                        
            except Exception as e:
                print(f"❌ API call attempt {attempt + 1} failed: {str(e)}")
                if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                    print("   Detected quota/rate limit issue")
                elif "safety" in str(e).lower():
                    print("   Detected safety filter issue")
                elif "token" in str(e).lower() and "limit" in str(e).lower():
                    print("   Detected token limit issue")
                
                if attempt == max_retries - 1:
                    raise e
                
                # Wait before retry (exponential backoff)
                wait_time = 2 ** attempt
                print(f"   Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        
        raise Exception("Failed to get response from Gemini API after all retries")
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with robust error handling."""
        
        # Clean up the response
        response = response.strip()
        
        # Try to parse as JSON directly first (since we set response_mime_type to application/json)
        try:
            parsed_json = json.loads(response)
            
            # If LLM returned an array but we expected an object, extract the first object
            if isinstance(parsed_json, list) and len(parsed_json) > 0 and isinstance(parsed_json[0], dict):
                return parsed_json[0]
            elif isinstance(parsed_json, dict):
                return parsed_json
            else:
                # If it's neither a dict nor a list of dicts, treat as error
                raise json.JSONDecodeError("Unexpected JSON structure", response, 0)
                
        except json.JSONDecodeError:
            pass
        
        # Clean common formatting issues
        response = self._clean_json_response(response)
        
        # Try again after cleaning
        try:
            parsed_json = json.loads(response)
            
            # Handle array vs object issue after cleaning too
            if isinstance(parsed_json, list) and len(parsed_json) > 0 and isinstance(parsed_json[0], dict):
                return parsed_json[0]
            elif isinstance(parsed_json, dict):
                return parsed_json
            else:
                raise json.JSONDecodeError("Unexpected JSON structure after cleaning", response, 0)
                
        except json.JSONDecodeError:
            pass
        
        # Look for JSON block markers if direct parsing fails
        json_str = self._extract_json_block(response)
        
        # Try to parse the extracted block
        try:
            parsed_json = json.loads(json_str)
            
            # Handle array vs object issue for extracted blocks too
            if isinstance(parsed_json, list) and len(parsed_json) > 0 and isinstance(parsed_json[0], dict):
                return parsed_json[0]
            elif isinstance(parsed_json, dict):
                return parsed_json
            else:
                raise json.JSONDecodeError("Unexpected JSON structure in extracted block", json_str, 0)
                
        except json.JSONDecodeError:
            pass
        
        # Try progressive fixing of malformed JSON
        fixed_json = self._fix_malformed_json(json_str)
        try:
            parsed_json = json.loads(fixed_json)
            
            # Handle array vs object issue for fixed JSON too
            if isinstance(parsed_json, list) and len(parsed_json) > 0 and isinstance(parsed_json[0], dict):
                return parsed_json[0]
            elif isinstance(parsed_json, dict):
                return parsed_json
            else:
                raise json.JSONDecodeError("Unexpected JSON structure in fixed JSON", fixed_json, 0)
                
        except json.JSONDecodeError as e:
            raise Exception(f"Could not extract valid JSON from response. Original response: {response[:500]}... Error: {str(e)}")
    
    def _clean_json_response(self, response: str) -> str:
        """Clean common formatting issues in JSON responses."""
        import re
        
        # Remove any leading/trailing non-JSON content
        response = response.strip()
        
        # Remove invalid control characters (ASCII 0-31 except \t, \n, \r)
        # These characters are not allowed in JSON strings and cause parsing errors
        control_char_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
        response = control_char_pattern.sub('', response)
        
        # Find JSON boundaries
        start_idx = -1
        end_idx = len(response)
        
        for i, char in enumerate(response):
            if char in ['{', '[']:
                start_idx = i
                break
        
        if start_idx >= 0:
            # Find matching closing bracket
            bracket_count = 0
            closing_char = '}' if response[start_idx] == '{' else ']'
            
            for i in range(start_idx, len(response)):
                if response[i] in ['{', '[']:
                    bracket_count += 1
                elif response[i] in ['}', ']']:
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
            
            response = response[start_idx:end_idx]
        
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'```\s*$', '', response)
        
        # Remove any trailing commas before closing brackets
        response = re.sub(r',(\s*[}\]])', r'\1', response)
        
        return response.strip()
    
    def _extract_json_block(self, response: str) -> str:
        """Extract JSON block from response."""
        
        # Look for JSON block markers
        if "```json" in response.lower():
            start = response.lower().find("```json") + 7
            end = response.find("```", start)
            if end == -1:
                end = len(response)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end == -1:
                end = len(response)
            json_str = response[start:end].strip()
        else:
            json_str = response
        
        return json_str
    
    def _fix_malformed_json(self, json_str: str) -> str:
        """Attempt to fix common malformed JSON issues."""
        import re
        
        # Remove invalid control characters first (ASCII 0-31 except \t, \n, \r)
        control_char_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
        json_str = control_char_pattern.sub('', json_str)
        
        # Fix unescaped quotes in string values
        def fix_quotes(match):
            content = match.group(1)
            # Escape unescaped quotes
            content = content.replace('\\"', '###ESCAPED_QUOTE###')
            content = content.replace('"', '\\"')
            content = content.replace('###ESCAPED_QUOTE###', '\\"')
            return f'"{content}"'
        
        # Fix quotes in string values (between ": " and next comma/bracket)
        json_str = re.sub(r':\s*"([^"]*(?:"[^"]*)*)"(?=\s*[,}\]])', lambda m: f': "{m.group(1).replace(chr(34), chr(92) + chr(34))}"', json_str)
        
        # Fix trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix missing commas between properties
        json_str = re.sub(r'"\s*\n\s*"', '",\n  "', json_str)
        json_str = re.sub(r'}\s*\n\s*"', '},\n  "', json_str)
        json_str = re.sub(r']\s*\n\s*"', '],\n  "', json_str)
        
        # Try to fix unterminated strings at end
        if json_str.count('"') % 2 != 0:
            # Find the last quote and see if we need to close a string
            last_quote_idx = json_str.rfind('"')
            if last_quote_idx > 0:
                # Check if this is an opening quote by counting quotes before it
                quotes_before = json_str[:last_quote_idx].count('"')
                if quotes_before % 2 == 0:  # Even number means this is an opening quote
                    # Find where to close it
                    after_quote = json_str[last_quote_idx + 1:]
                    close_pos = last_quote_idx + 1
                    for i, char in enumerate(after_quote):
                        if char in [',', '}', ']', '\n']:
                            close_pos = last_quote_idx + 1 + i
                            break
                    json_str = json_str[:close_pos] + '"' + json_str[close_pos:]
        
        return json_str
    
    def _validate_json_against_schema(self, json_data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate JSON against schema and return list of errors."""
        
        try:
            import jsonschema
            validator = jsonschema.validators.validator_for(schema)(schema)
            errors = []
            
            for error in validator.iter_errors(json_data):
                errors.append(f"Path {'.'.join(map(str, error.path))}: {error.message}")
            
            return errors
            
        except ImportError:
            # Fallback validation without jsonschema library
            return self._basic_validation(json_data, schema)
    
    def _basic_validation(self, json_data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Basic validation without jsonschema library."""
        
        errors = []
        
        def validate_object(obj: Any, schema_obj: Dict[str, Any], path: str = "") -> None:
            if schema_obj.get("type") == "object" and "properties" in schema_obj:
                if not isinstance(obj, dict):
                    errors.append(f"{path}: Expected object, got {type(obj).__name__}")
                    return
                
                # Check required fields
                required = schema_obj.get("required", [])
                for req_field in required:
                    if req_field not in obj:
                        errors.append(f"{path}.{req_field}: Required field missing")
                
                # Validate each property
                for prop_name, prop_schema in schema_obj["properties"].items():
                    if prop_name in obj:
                        validate_object(obj[prop_name], prop_schema, f"{path}.{prop_name}")
            
            elif schema_obj.get("type") == "array":
                if not isinstance(obj, list):
                    errors.append(f"{path}: Expected array, got {type(obj).__name__}")
            
            elif "enum" in schema_obj:
                if obj not in schema_obj["enum"]:
                    errors.append(f"{path}: Value '{obj}' not in allowed enum values")
        
        validate_object(json_data, schema)
        return errors
    
    def _merge_json_outputs(self, json1: Dict[str, Any], json2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two JSON outputs, with json2 taking precedence for conflicts."""
        
        result = json1.copy()
        
        for key, value in json2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_json_outputs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _calculate_confidence_score(self, 
                                  json_output: Dict[str, Any], 
                                  schema: Dict[str, Any],
                                  validation_errors: List[str],
                                  input_confidence: float) -> float:
        """Calculate confidence score for the generated JSON."""
        
        # Base confidence from input processing
        confidence = input_confidence
        
        # Penalize validation errors
        if validation_errors:
            error_penalty = min(0.5, len(validation_errors) * 0.1)
            confidence -= error_penalty
        
        # Bonus for completely filled schema
        schema_fields = self._count_schema_fields(schema)
        filled_fields = self._count_filled_fields(json_output)
        
        if schema_fields > 0:
            completion_ratio = min(1.0, filled_fields / schema_fields)
            confidence *= (0.5 + 0.5 * completion_ratio)
        
        return max(0.0, min(1.0, confidence))
    
    def _count_schema_fields(self, schema: Dict[str, Any]) -> int:
        """Count total fields in schema."""
        count = 0
        
        def count_recursive(obj: Any) -> None:
            nonlocal count
            if isinstance(obj, dict):
                if obj.get("type") == "object" and "properties" in obj:
                    count += len(obj["properties"])
                    for prop in obj["properties"].values():
                        count_recursive(prop)
                elif "items" in obj:
                    count_recursive(obj["items"])
        
        count_recursive(schema)
        return count
    
    def _count_filled_fields(self, json_data: Any) -> int:
        """Count filled fields in JSON data."""
        if isinstance(json_data, dict):
            count = len([k for k, v in json_data.items() if v is not None])
            for value in json_data.values():
                if isinstance(value, (dict, list)):
                    count += self._count_filled_fields(value)
            return count
        elif isinstance(json_data, list):
            return sum(self._count_filled_fields(item) for item in json_data)
        else:
            return 1 if json_data is not None else 0 

    def _create_fallback_json(self, schema: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Create a minimal valid JSON structure when extraction fails."""
        fallback = {"_error": error_message}
        
        # Try to add required fields with default values
        try:
            required_fields = self._extract_required_fields(schema)
            for field_path, field_type in required_fields:
                # Only add top-level required fields to avoid complexity
                if "." not in field_path:
                    if field_type == "string":
                        fallback[field_path] = ""
                    elif field_type == "integer":
                        fallback[field_path] = 0
                    elif field_type == "number":
                        fallback[field_path] = 0.0
                    elif field_type == "boolean":
                        fallback[field_path] = False
                    elif field_type == "array":
                        fallback[field_path] = []
                    elif field_type == "object":
                        fallback[field_path] = {}
        except Exception:
            # If even fallback creation fails, just return the error
            pass
        
        return fallback 
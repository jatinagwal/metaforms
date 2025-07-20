"""Schema Analysis Module - Analyzes JSON schema complexity for processing strategy."""

import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class SchemaMetrics:
    total_fields: int
    nested_objects: int
    max_nesting_depth: int
    enum_fields: int
    total_enum_values: int
    required_fields: int
    conditional_fields: int
    complexity_score: float
    complexity_level: ComplexityLevel


@dataclass
class ProcessingStrategy:
    use_multi_pass: bool
    estimated_calls: int
    chunk_schema: bool
    use_validation_loop: bool
    max_retries: int
    confidence_threshold: float


class SchemaAnalyzer:
    """Analyzes JSON schemas to determine processing complexity and strategy."""
    
    def __init__(self):
        self.metrics = None
        self.strategy = None
    
    def analyze_schema(self, schema: Dict[str, Any]) -> Tuple[SchemaMetrics, ProcessingStrategy]:
        """
        Comprehensive schema analysis to determine complexity and processing strategy.
        
        Args:
            schema: JSON schema dictionary
            
        Returns:
            Tuple of (SchemaMetrics, ProcessingStrategy)
        """
        metrics = self._calculate_metrics(schema)
        strategy = self._determine_strategy(metrics)
        
        self.metrics = metrics
        self.strategy = strategy
        
        return metrics, strategy
    
    def _calculate_metrics(self, schema: Dict[str, Any]) -> SchemaMetrics:
        """Calculate detailed metrics about schema complexity."""
        
        total_fields = 0
        nested_objects = 0
        max_depth = 0
        enum_fields = 0
        total_enum_values = 0
        required_fields = 0
        conditional_fields = 0
        
        def analyze_recursive(obj: Any, current_depth: int = 0) -> None:
            nonlocal total_fields, nested_objects, max_depth, enum_fields, total_enum_values, required_fields, conditional_fields
            
            max_depth = max(max_depth, current_depth)
            
            if isinstance(obj, dict):
                if obj.get("type") == "object":
                    nested_objects += 1
                    if "properties" in obj:
                        for prop_name, prop_def in obj["properties"].items():
                            total_fields += 1
                            analyze_recursive(prop_def, current_depth + 1)
                    
                    if "required" in obj:
                        required_fields += len(obj["required"])
                
                elif "enum" in obj:
                    enum_fields += 1
                    total_enum_values += len(obj["enum"])
                
                # Check for conditional fields (oneOf, anyOf, allOf)
                for conditional_key in ["oneOf", "anyOf", "allOf"]:
                    if conditional_key in obj:
                        conditional_fields += 1
                        for condition in obj[conditional_key]:
                            analyze_recursive(condition, current_depth)
                
                # Process definitions and other nested structures
                for key, value in obj.items():
                    if key in ["definitions", "$defs", "properties", "items", "additionalProperties"]:
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                analyze_recursive(sub_value, current_depth + 1)
                        else:
                            analyze_recursive(value, current_depth + 1)
            
            elif isinstance(obj, list):
                for item in obj:
                    analyze_recursive(item, current_depth)
        
        analyze_recursive(schema)
        
        # Calculate complexity score
        complexity_score = (
            (total_fields * 0.1) +
            (nested_objects * 0.5) +
            (max_depth * 1.0) +
            (enum_fields * 0.3) +
            (total_enum_values * 0.01) +
            (conditional_fields * 2.0)
        )
        
        # Determine complexity level
        if complexity_score < 5:
            complexity_level = ComplexityLevel.SIMPLE
        elif complexity_score < 20:
            complexity_level = ComplexityLevel.MODERATE
        elif complexity_score < 50:
            complexity_level = ComplexityLevel.COMPLEX
        else:
            complexity_level = ComplexityLevel.VERY_COMPLEX
        
        return SchemaMetrics(
            total_fields=total_fields,
            nested_objects=nested_objects,
            max_nesting_depth=max_depth,
            enum_fields=enum_fields,
            total_enum_values=total_enum_values,
            required_fields=required_fields,
            conditional_fields=conditional_fields,
            complexity_score=complexity_score,
            complexity_level=complexity_level
        )
    
    def _determine_strategy(self, metrics: SchemaMetrics) -> ProcessingStrategy:
        """Determine processing strategy based on schema metrics."""
        
        # Base strategy parameters
        use_multi_pass = False
        estimated_calls = 1
        chunk_schema = False
        use_validation_loop = True
        max_retries = 2
        confidence_threshold = 0.8
        
        # Adjust based on complexity level
        if metrics.complexity_level == ComplexityLevel.SIMPLE:
            estimated_calls = 1
            max_retries = 1
            confidence_threshold = 0.9
            
        elif metrics.complexity_level == ComplexityLevel.MODERATE:
            estimated_calls = 2
            use_validation_loop = True
            max_retries = 3
            
        elif metrics.complexity_level == ComplexityLevel.COMPLEX:
            use_multi_pass = True
            estimated_calls = 3
            chunk_schema = metrics.total_fields > 100
            max_retries = 4
            confidence_threshold = 0.7
            
        elif metrics.complexity_level == ComplexityLevel.VERY_COMPLEX:
            use_multi_pass = True
            estimated_calls = 5
            chunk_schema = True
            max_retries = 5
            confidence_threshold = 0.6
        
        # Additional adjustments based on specific metrics
        if metrics.max_nesting_depth > 5:
            estimated_calls += 1
            use_multi_pass = True
        
        if metrics.total_enum_values > 500:
            estimated_calls += 1
            chunk_schema = True
        
        if metrics.conditional_fields > 10:
            estimated_calls += 2
            use_validation_loop = True
        
        return ProcessingStrategy(
            use_multi_pass=use_multi_pass,
            estimated_calls=min(estimated_calls, 10),  # Cap at 10 calls
            chunk_schema=chunk_schema,
            use_validation_loop=use_validation_loop,
            max_retries=max_retries,
            confidence_threshold=confidence_threshold
        )
    
    def get_schema_chunks(self, schema: Dict[str, Any], max_chunk_size: int = 50) -> List[Dict[str, Any]]:
        """
        Break down complex schema into manageable chunks for processing.
        
        Args:
            schema: Original JSON schema
            max_chunk_size: Maximum number of fields per chunk
            
        Returns:
            List of schema chunks
        """
        if not self.strategy or not self.strategy.chunk_schema:
            return [schema]
        
        chunks = []
        
        def extract_properties(obj: Dict[str, Any], path: str = "") -> List[Tuple[str, Any]]:
            """Extract all properties with their paths."""
            properties = []
            
            if "properties" in obj:
                for prop_name, prop_def in obj["properties"].items():
                    full_path = f"{path}.{prop_name}" if path else prop_name
                    properties.append((full_path, prop_def))
                    
                    # Recursively extract nested properties
                    if isinstance(prop_def, dict) and prop_def.get("type") == "object":
                        nested_props = extract_properties(prop_def, full_path)
                        properties.extend(nested_props)
            
            return properties
        
        all_properties = extract_properties(schema)
        
        # Group properties into chunks
        for i in range(0, len(all_properties), max_chunk_size):
            chunk_props = all_properties[i:i + max_chunk_size]
            
            # Create a simplified schema chunk
            chunk_schema = {
                "$schema": schema.get("$schema", "http://json-schema.org/draft-07/schema#"),
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for prop_path, prop_def in chunk_props:
                # Flatten the property path for chunk processing
                clean_path = prop_path.split('.')[-1]
                chunk_schema["properties"][clean_path] = prop_def
            
            chunks.append(chunk_schema)
        
        return chunks
    
    def estimate_token_usage(self, input_text: str) -> Dict[str, int]:
        """Estimate token usage for input text and schema."""
        
        # Rough estimation based on character count
        input_tokens = len(input_text.split()) * 1.3  # Account for subword tokens
        
        if self.metrics:
            schema_tokens = (
                self.metrics.total_fields * 5 +
                self.metrics.total_enum_values * 2 +
                self.metrics.nested_objects * 10
            )
        else:
            schema_tokens = 1000  # Default estimate
        
        total_calls = self.strategy.estimated_calls if self.strategy else 1
        
        return {
            "input_tokens": int(input_tokens),
            "schema_tokens": int(schema_tokens),
            "estimated_total_tokens": int((input_tokens + schema_tokens) * total_calls),
            "estimated_calls": total_calls
        } 
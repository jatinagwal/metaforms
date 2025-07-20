#!/usr/bin/env python3
"""Test script to validate the Text-to-JSON Conversion System"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_schema_analyzer():
    """Test schema analyzer with sample schema."""
    print("🔍 Testing Schema Analyzer...")
    
    from src.core.schema_analyzer import SchemaAnalyzer
    
    test_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}, 
            "skills": {"type": "array", "items": {"type": "string"}},
            "experience": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "role": {"type": "string"}
                }
            }
        },
        "required": ["name"]
    }
    
    analyzer = SchemaAnalyzer()
    metrics, strategy = analyzer.analyze_schema(test_schema)
    
    print(f"✅ Schema complexity: {metrics.complexity_level.value}")
    print(f"✅ Total fields: {metrics.total_fields}")
    print(f"✅ Estimated API calls: {strategy.estimated_calls}")


def test_text_processor():
    """Test text processor with sample text."""
    print("📝 Testing Text Processor...")
    
    from src.core.text_processor import TextProcessor
    
    processor = TextProcessor()
    
    # Test different formats
    test_cases = [
        ("Hello world, this is a test.", "text"),
        ("name,age,city\nJohn,30,NYC\nJane,25,LA", "csv"),
        ('{"name": "John", "age": 30}', "json")
    ]
    
    for text, expected_format in test_cases:
        result = processor.process_input(text)
        print(f"✅ Format detection: {result.format_type} (expected: {expected_format})")
        print(f"✅ Token count: {result.total_tokens}")


def test_sample_cases():
    """Test with sample test cases from directory."""
    print("🧪 Testing Sample Cases...")
    
    test_cases_dir = Path("sample test cases")
    if not test_cases_dir.exists():
        print("⚠️  No sample test cases directory found")
        return
    
    from src.core.schema_analyzer import SchemaAnalyzer
    
    analyzer = SchemaAnalyzer()
    schema_files = list(test_cases_dir.glob("*schema*.json"))
    
    print(f"✅ Found {len(schema_files)} test case schemas")
    
    for schema_file in schema_files[:3]:  # Test first 3
        try:
            with open(schema_file, 'r') as f:
                schema = json.load(f)
            
            metrics, strategy = analyzer.analyze_schema(schema)
            print(f"✅ {schema_file.stem}: {metrics.complexity_level.value} ({strategy.estimated_calls} calls)")
            
        except Exception as e:
            print(f"❌ Failed to analyze {schema_file.name}: {e}")


async def test_full_conversion():
    """Test full conversion pipeline (requires API key)."""
    print("🔄 Testing Full Conversion...")
    
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  Skipping full conversion test - no API key")
        return
    
    try:
        from src.core.llm_manager import LLMManager, LLMConfig
        
        config = LLMConfig(api_key=os.getenv('GEMINI_API_KEY'))
        manager = LLMManager(config)
        
        # Simple test case
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        result = await manager.convert_to_json(
            input_text="John Smith is 30 years old",
            schema=schema
        )
        
        print(f"✅ Conversion successful!")
        print(f"✅ Confidence: {result.confidence_score:.2%}")
        print(f"✅ Result: {result.json_output}")
        
    except Exception as e:
        print(f"❌ Full conversion test failed: {e}")


def main():
    """Run all tests."""
    print("🚀 Running Text-to-JSON System Tests\n")
    
    tests = [
        test_schema_analyzer,
        test_text_processor,
        test_sample_cases
    ]
    
    for test in tests:
        try:
            test()
            print()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}\n")
    
    # Run async test
    try:
        asyncio.run(test_full_conversion())
    except Exception as e:
        print(f"❌ Full conversion test failed: {e}")
    
    print("🎉 All tests completed!")


if __name__ == "__main__":
    main() 
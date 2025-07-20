#!/usr/bin/env python3
"""Debug script to test API with different input sizes"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_small_input():
    """Test with a small input to verify API connectivity."""
    print("🧪 Testing API with small input...")
    
    try:
        from src.core.llm_manager import LLMManager, LLMConfig
        
        # Check if API key is available
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable not set")
            return False
        
        config = LLMConfig(api_key=api_key)
        manager = LLMManager(config)
        
        # Simple test case
        small_input = """
        Email from: john@example.com
        Date: 2025-01-05
        Subject: Test Data
        
        Please process the following user information:
        - Name: John Smith
        - Age: 30
        - Department: Engineering
        """
        
        simple_schema = {
            "type": "object",
            "required": ["user"],
            "properties": {
                "user": {
                    "type": "object",
                    "required": ["name", "age"],
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "department": {"type": "string"}
                    }
                }
            }
        }
        
        print(f"✅ Input size: {len(small_input)} characters")
        
        result = await manager.convert_to_json(
            input_text=small_input,
            schema=simple_schema,
            input_format="text"
        )
        
        print("✅ Small input test successful!")
        print(f"✅ Result: {json.dumps(result.json_output, indent=2)}")
        print(f"✅ Confidence: {result.confidence_score:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Small input test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_medium_input():
    """Test with medium input (first 1000 lines of eval data)."""
    print("\n🧪 Testing API with medium input...")
    
    try:
        from src.core.llm_manager import LLMManager, LLMConfig
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable not set")
            return False
        
        config = LLMConfig(api_key=api_key)
        manager = LLMManager(config)
        
        # Load partial eval data
        eval_text_path = Path("eval/test_case_large_extended.txt")
        eval_schema_path = Path("eval/test_schema_large.json")
        
        if not eval_text_path.exists():
            print(f"❌ Eval text file not found: {eval_text_path}")
            return False
            
        if not eval_schema_path.exists():
            print(f"❌ Eval schema file not found: {eval_schema_path}")
            return False
        
        # Read first 1000 lines only
        with open(eval_text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:1000]
            medium_input = ''.join(lines)
        
        with open(eval_schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        print(f"✅ Medium input size: {len(medium_input):,} characters ({len(lines)} lines)")
        
        result = await manager.convert_to_json(
            input_text=medium_input,
            schema=schema,
            input_format="text"
        )
        
        print("✅ Medium input test successful!")
        print(f"✅ Confidence: {result.confidence_score:.2%}")
        print(f"✅ API calls: {result.api_calls_made}")
        
        if "_error" in result.json_output:
            print(f"⚠️ Fallback used: {result.json_output['_error']}")
        else:
            print("✅ No fallback needed")
        
        return True
        
    except Exception as e:
        print(f"❌ Medium input test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_input():
    """Test with full eval data."""
    print("\n🧪 Testing API with full eval input...")
    
    try:
        from src.core.llm_manager import LLMManager, LLMConfig
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable not set")
            return False
        
        config = LLMConfig(api_key=api_key)
        manager = LLMManager(config)
        
        # Load full eval data
        eval_text_path = Path("eval/test_case_large_extended.txt")
        eval_schema_path = Path("eval/test_schema_large.json")
        
        with open(eval_text_path, 'r', encoding='utf-8') as f:
            full_input = f.read()
        
        with open(eval_schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        print(f"✅ Full input size: {len(full_input):,} characters")
        
        result = await manager.convert_to_json(
            input_text=full_input,
            schema=schema,
            input_format="text"
        )
        
        print("✅ Full input test successful!")
        print(f"✅ Confidence: {result.confidence_score:.2%}")
        print(f"✅ API calls: {result.api_calls_made}")
        print(f"✅ Processing time: {result.processing_time:.2f}s")
        
        if "_error" in result.json_output:
            print(f"⚠️ Fallback used: {result.json_output['_error']}")
        else:
            print("✅ No fallback needed - full success!")
        
        return True
        
    except Exception as e:
        print(f"❌ Full input test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run progressive tests to identify the issue."""
    print("🚀 API Debugging - Progressive Input Size Tests")
    print("=" * 60)
    
    # Test 1: Small input
    small_success = await test_small_input()
    
    if not small_success:
        print("\n❌ Basic API connectivity failed. Check your API key and quota.")
        return
    
    # Test 2: Medium input
    medium_success = await test_medium_input()
    
    if not medium_success:
        print("\n❌ Medium input failed. Issue may be with input size or complexity.")
        return
    
    # Test 3: Full input
    full_success = await test_full_input()
    
    if full_success:
        print("\n🎉 All tests passed! The issue may have been resolved.")
    else:
        print("\n⚠️ Full input failed but smaller inputs work. Issue is likely input size related.")
        print("   Consider implementing more aggressive chunking or input preprocessing.")

if __name__ == "__main__":
    asyncio.run(main()) 
# 🔄 Extraction System

Convert unstructured text into structured JSON following any schema using Gemini AI.

## Features

- **Complex Schema Support**: 3-7 levels nesting, 50-150 objects, 1000+ enums
- **Multiple Formats**: PDF, DOCX, CSV, Excel, text
- **Dynamic Processing**: Auto-adjusts based on schema complexity  
- **Web Interface**: Streamlit app + REST API
- **Large Context**: Up to 50-page documents

## Quick Start

1. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Set API key:**
   ```bash
   export GEMINI_API_KEY="your-key-here"
   ```
   Get your key: https://makersuite.google.com/app/apikey

3. **Run:**
   ```bash
   # Web interface
   python3 run_streamlit.py
   
   # REST API  
   python3 run_api.py
   ```

## Usage

### Web Interface
- Schema analysis and complexity metrics
- File upload (PDF, DOCX, CSV, Excel)
- Test cases with sample data
- Real-time conversion

### REST API
```python
# POST /convert
{
  "input_text": "John Doe, age 30, Software Engineer",
  "schema": {"type": "object", "properties": {...}},
  "input_format": "text"
}
```

### Python API
```python
from src.core.llm_manager import LLMManager, LLMConfig

config = LLMConfig(api_key="your-key")
manager = LLMManager(config)

result = await manager.convert_to_json(
    input_text="Your text here",
    schema={"type": "object", ...}
)
```

## Architecture

```
src/
├── core/
│   ├── schema_analyzer.py    # Schema complexity analysis
│   ├── text_processor.py     # Multi-format processing  
│   └── llm_manager.py        # Gemini API integration
├── api/main.py              # FastAPI endpoints
└── web/streamlit_app.py     # Web interface
```

## Schema Complexity

The system automatically analyzes schemas and adjusts processing:

- **Simple**: Direct single-pass conversion
- **Moderate**: Multi-pass with validation  
- **Complex**: Schema chunking + refinement loops
- **Very Complex**: Advanced optimization strategies

## Test Cases

Included sample schemas and data:
- Resume extraction
- Citation parsing  
- GitHub Actions workflow
- Complex nested objects

Run tests: `python3 test_system.py` 
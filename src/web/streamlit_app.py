"""Streamlit Web Interface for Extraction System"""

# Add src to Python path
import sys
from pathlib import Path

# Get the src directory (parent of parent of this file)
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

import streamlit as st
import json
import os
import time
import tempfile
import io
import asyncio
from typing import Dict, Any, List, Optional

# Third-party imports
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import our core modules - now they should be found
from core.llm_manager import LLMManager, LLMConfig
from core.schema_analyzer import SchemaAnalyzer
from core.text_processor import TextProcessor


# Page configuration
st.set_page_config(
    page_title="Text-to-JSON Converter",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'llm_manager' not in st.session_state:
    st.session_state.llm_manager = None
if 'schema_analyzer' not in st.session_state:
    st.session_state.schema_analyzer = SchemaAnalyzer()
if 'text_processor' not in st.session_state:
    st.session_state.text_processor = TextProcessor()
if 'conversion_history' not in st.session_state:
    st.session_state.conversion_history = []


@st.cache_resource
def initialize_llm():
    """Initialize LLM manager with caching."""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return None, "Gemini API key not found. Please set GEMINI_API_KEY environment variable."
        
        config = LLMConfig(
            api_key=api_key, 
            temperature=0.0,  # Use 0 for consistency
            thinking_budget=30000  # Limited thinking for reliability
        )
        manager = LLMManager(config)
        return manager, "LLM Manager initialized successfully"
    except Exception as e:
        return None, f"Failed to initialize LLM: {str(e)}"


@st.cache_data
def load_test_cases():
    """Load available test cases."""
    test_cases_dir = Path("sample test cases")
    if not test_cases_dir.exists():
        return {}
    
    test_cases = {}
    schema_analyzer = SchemaAnalyzer()
    
    for file_path in test_cases_dir.glob("*.json"):
        if "schema" in file_path.name:
            try:
                with open(file_path, 'r') as f:
                    schema = json.load(f)
                
                # Analyze schema
                metrics, strategy = schema_analyzer.analyze_schema(schema)
                
                # Look for sample inputs
                sample_inputs = []
                base_name = file_path.stem.replace("_schema", "").replace(" schema", "")
                
                for input_file in test_cases_dir.glob("*"):
                    if (base_name.lower() in input_file.name.lower() and 
                        "schema" not in input_file.name.lower() and
                        input_file.suffix in ['.md', '.txt', '.bib', '.csv']):
                        
                        try:
                            with open(input_file, 'r') as f:
                                content = f.read()
                            sample_inputs.append({
                                "filename": input_file.name,
                                "content": content
                            })
                        except Exception:
                            pass
                
                test_cases[file_path.stem] = {
                    "schema": schema,
                    "metrics": metrics,
                    "strategy": strategy,
                    "sample_inputs": sample_inputs
                }
            except Exception as e:
                st.error(f"Error loading test case {file_path.name}: {str(e)}")
    
    return test_cases


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🔄 Extraction System")
    st.markdown("Convert unstructured text into structured JSON following desired schemas")
    
    # Initialize LLM Manager
    if st.session_state.llm_manager is None:
        with st.spinner("Initializing LLM Manager..."):
            manager, message = initialize_llm()
            st.session_state.llm_manager = manager
            if manager is None:
                st.error(message)
                st.stop()
            else:
                st.success(message)
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Convert Text", "Schema Analysis", "Test Cases", "File Upload", "System Metrics"]
    )
    
    if page == "Convert Text":
        convert_text_page()
    elif page == "Schema Analysis":
        schema_analysis_page()
    elif page == "Test Cases":
        test_cases_page()
    elif page == "File Upload":
        file_upload_page()
    elif page == "System Metrics":
        system_metrics_page()


def convert_text_page():
    """Text conversion page."""
    st.header("📝 Convert Text to JSON")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        # Text input
        input_text = st.text_area(
            "Enter your text:",
            height=300,
            placeholder="Paste your unstructured text here..."
        )
        
        # Input format hint
        input_format = st.selectbox(
            "Input format hint (optional):",
            ["auto", "text", "csv", "json", "email", "resume"]
        )
        
        if input_format == "auto":
            input_format = None
    
    with col2:
        st.subheader("JSON Schema")
        
        # Schema input options
        schema_option = st.radio(
            "Schema source:",
            ["Paste JSON Schema", "Use Test Case Schema"]
        )
        
        if schema_option == "Paste JSON Schema":
            schema_text = st.text_area(
                "JSON Schema:",
                height=300,
                placeholder='{"type": "object", "properties": {...}}'
            )
            
            try:
                schema = json.loads(schema_text) if schema_text else None
            except json.JSONDecodeError:
                st.error("Invalid JSON schema")
                schema = None
        
        else:
            test_cases = load_test_cases()
            if test_cases:
                selected_test = st.selectbox("Select test case:", list(test_cases.keys()))
                schema = test_cases[selected_test]["schema"] if selected_test else None
                if schema:
                    st.json(schema, expanded=False)
            else:
                st.warning("No test cases available")
                schema = None
    
    # Convert button
    if st.button("🔄 Convert to JSON", type="primary"):
        if not input_text:
            st.error("Please enter input text")
        elif not schema:
            st.error("Please provide a valid JSON schema")
        else:
            with st.spinner("Converting..."):
                try:
                    # Run conversion
                    result = asyncio.run(
                        st.session_state.llm_manager.convert_to_json(
                            input_text=input_text,
                            schema=schema,
                            input_format=input_format
                        )
                    )
                    
                    # Store in history
                    st.session_state.conversion_history.append({
                        "timestamp": time.time(),
                        "result": result,
                        "input_text": input_text[:200] + "..." if len(input_text) > 200 else input_text
                    })
                    
                    # Display results
                    st.success("✅ Conversion completed!")
                    
                    # Results tabs
                    tab1, tab2, tab3 = st.tabs(["JSON Output", "Metadata", "Validation"])
                    
                    with tab1:
                        st.subheader("Generated JSON")
                        st.json(result.json_output)
                        
                        # Download button
                        json_str = json.dumps(result.json_output, indent=2)
                        st.download_button(
                            "📥 Download JSON",
                            json_str,
                            file_name="converted.json",
                            mime="application/json"
                        )
                    
                    with tab2:
                        st.subheader("Processing Metadata")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Confidence Score", f"{result.confidence_score:.2%}")
                        col2.metric("Processing Time", f"{result.processing_time:.2f}s")
                        col3.metric("API Calls", result.api_calls_made)
                        
                        st.json(result.metadata)
                    
                    with tab3:
                        st.subheader("Validation Results")
                        
                        if result.validation_errors:
                            st.error(f"Found {len(result.validation_errors)} validation errors:")
                            for i, error in enumerate(result.validation_errors, 1):
                                st.write(f"{i}. {error}")
                        else:
                            st.success("✅ No validation errors found!")
                
                except Exception as e:
                    st.error(f"Conversion failed: {str(e)}")


def schema_analysis_page():
    """Schema analysis page."""
    st.header("🔍 Schema Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Schema")
        
        # Schema input
        schema_text = st.text_area(
            "Paste JSON Schema:",
            height=400,
            placeholder='{"type": "object", "properties": {...}}'
        )
        
        if st.button("🔍 Analyze Schema"):
            if not schema_text:
                st.error("Please enter a JSON schema")
            else:
                try:
                    schema = json.loads(schema_text)
                    
                    with st.spinner("Analyzing schema..."):
                        metrics, strategy = st.session_state.schema_analyzer.analyze_schema(schema)
                        token_usage = st.session_state.schema_analyzer.estimate_token_usage("Sample text")
                    
                    # Store results in session state for the second column
                    st.session_state.analysis_results = {
                        "metrics": metrics,
                        "strategy": strategy,
                        "token_usage": token_usage
                    }
                    
                except json.JSONDecodeError:
                    st.error("Invalid JSON schema")
    
    with col2:
        st.subheader("Analysis Results")
        
        if hasattr(st.session_state, 'analysis_results'):
            results = st.session_state.analysis_results
            metrics = results["metrics"]
            strategy = results["strategy"]
            token_usage = results["token_usage"]
            
            # Complexity overview
            st.metric("Complexity Level", metrics.complexity_level.value.title())
            st.metric("Complexity Score", f"{metrics.complexity_score:.2f}")
            
            # Metrics dashboard
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Fields", metrics.total_fields)
            col2.metric("Nested Objects", metrics.nested_objects)
            col3.metric("Max Nesting Depth", metrics.max_nesting_depth)
            
            col1.metric("Enum Fields", metrics.enum_fields)
            col2.metric("Total Enum Values", metrics.total_enum_values)
            col3.metric("Required Fields", metrics.required_fields)
            
            # Processing strategy
            st.subheader("Processing Strategy")
            strategy_data = {
                "Multi-pass Processing": "Yes" if strategy.use_multi_pass else "No",
                "Estimated API Calls": strategy.estimated_calls,
                "Schema Chunking": "Yes" if strategy.chunk_schema else "No",
                "Validation Loop": "Yes" if strategy.use_validation_loop else "No",
                "Max Retries": strategy.max_retries,
                "Confidence Threshold": f"{strategy.confidence_threshold:.2%}"
            }
            
            for key, value in strategy_data.items():
                st.write(f"**{key}:** {value}")
            
            # Token usage estimation
            st.subheader("Token Usage Estimation")
            st.write(f"**Schema Tokens:** {token_usage['schema_tokens']}")
            st.write(f"**Estimated Total Tokens:** {token_usage['estimated_total_tokens']}")
            st.write(f"**Estimated Calls:** {token_usage['estimated_calls']}")
            
            # Visualization
            create_complexity_charts(metrics)


def create_complexity_charts(metrics):
    """Create complexity visualization charts."""
    st.subheader("Complexity Visualization")
    
    # Metrics breakdown pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Fields', 'Objects', 'Enums', 'Conditional'],
        values=[metrics.total_fields, metrics.nested_objects, 
               metrics.enum_fields, metrics.conditional_fields],
        hole=.3
    )])
    fig.update_layout(title="Schema Components Breakdown")
    st.plotly_chart(fig, use_container_width=True)


def test_cases_page():
    """Test cases page."""
    st.header("🧪 Test Cases")
    
    test_cases = load_test_cases()
    
    if not test_cases:
        st.warning("No test cases found in 'sample test cases' directory")
        return
    
    # Test case selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Available Test Cases")
        
        for name, data in test_cases.items():
            metrics = data["metrics"]
            strategy = data["strategy"]
            
            with st.expander(f"📋 {name}"):
                st.write(f"**Complexity:** {metrics.complexity_level.value}")
                st.write(f"**Fields:** {metrics.total_fields}")
                st.write(f"**Nesting Depth:** {metrics.max_nesting_depth}")
                st.write(f"**Estimated Calls:** {strategy.estimated_calls}")
                
                if st.button(f"Select {name}", key=f"select_{name}"):
                    st.session_state.selected_test_case = name
    
    with col2:
        if hasattr(st.session_state, 'selected_test_case'):
            test_name = st.session_state.selected_test_case
            test_data = test_cases[test_name]
            
            st.subheader(f"Test Case: {test_name}")
            
            # Show schema
            with st.expander("📄 View Schema", expanded=False):
                st.json(test_data["schema"])
            
            # Sample inputs
            if test_data["sample_inputs"]:
                st.subheader("Sample Inputs")
                
                for i, sample in enumerate(test_data["sample_inputs"]):
                    with st.expander(f"📝 {sample['filename']}"):
                        st.text_area(
                            "Content:",
                            sample["content"],
                            height=200,
                            key=f"sample_{i}",
                            disabled=True
                        )
                        
                        if st.button(f"🔄 Run Test with {sample['filename']}", key=f"run_{i}"):
                            run_test_case(test_name, test_data, sample["content"])
            
            # Custom input
            st.subheader("Custom Input")
            custom_input = st.text_area(
                "Enter your own test input:",
                height=150,
                key="custom_test_input"
            )
            
            if st.button("🔄 Run Custom Test"):
                if custom_input:
                    run_test_case(test_name, test_data, custom_input)
                else:
                    st.error("Please enter test input")


def run_test_case(test_name, test_data, input_text):
    """Run a test case."""
    with st.spinner(f"Running test case: {test_name}"):
        try:
            result = asyncio.run(
                st.session_state.llm_manager.convert_to_json(
                    input_text=input_text,
                    schema=test_data["schema"],
                    input_format="text"
                )
            )
            
            st.success("✅ Test completed!")
            
            # Results
            tab1, tab2 = st.tabs(["Results", "Analysis"])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", f"{result.confidence_score:.2%}")
                col2.metric("Time", f"{result.processing_time:.2f}s")
                col3.metric("API Calls", result.api_calls_made)
                
                st.json(result.json_output)
                
                if result.validation_errors:
                    st.error("Validation Errors:")
                    for error in result.validation_errors:
                        st.write(f"- {error}")
            
            with tab2:
                st.subheader("Expected vs Actual Complexity")
                expected_calls = test_data["strategy"].estimated_calls
                actual_calls = result.api_calls_made
                
                st.write(f"**Expected API Calls:** {expected_calls}")
                st.write(f"**Actual API Calls:** {actual_calls}")
                st.write(f"**Efficiency:** {expected_calls/actual_calls:.2f}x" if actual_calls > 0 else "N/A")
                
        except Exception as e:
            st.error(f"Test failed: {str(e)}")


def file_upload_page():
    """File upload page."""
    st.header("📁 File Upload Conversion")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload File")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'pdf', 'docx', 'csv', 'xlsx', 'json']
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # File info
            st.write(f"**Size:** {len(uploaded_file.getvalue())} bytes")
            st.write(f"**Type:** {uploaded_file.type}")
            
            # Process file
            file_content = uploaded_file.getvalue()
            processed_text = st.session_state.text_processor.process_input(
                input_data=file_content,
                filename=uploaded_file.name
            )
            
            st.write(f"**Processing Confidence:** {processed_text.confidence:.2%}")
            st.write(f"**Total Tokens:** {processed_text.total_tokens}")
            st.write(f"**Chunks:** {len(processed_text.chunks)}")
            
            # Preview processed content
            with st.expander("Preview Processed Content"):
                preview_text = processed_text.content[:1000]
                if len(processed_text.content) > 1000:
                    preview_text += "..."
                st.text(preview_text)
    
    with col2:
        st.subheader("Schema & Conversion")
        
        schema_text = st.text_area(
            "JSON Schema:",
            height=300,
            placeholder='{"type": "object", "properties": {...}}'
        )
        
        if st.button("🔄 Convert File") and uploaded_file is not None:
            if not schema_text:
                st.error("Please provide a JSON schema")
            else:
                try:
                    schema = json.loads(schema_text)
                    
                    with st.spinner("Converting file..."):
                        result = asyncio.run(
                            st.session_state.llm_manager.convert_to_json(
                                input_text=processed_text.content,
                                schema=schema,
                                input_format=processed_text.format_type
                            )
                        )
                    
                    st.success("✅ File converted successfully!")
                    
                    # Results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Confidence", f"{result.confidence_score:.2%}")
                    col2.metric("Time", f"{result.processing_time:.2f}s")
                    col3.metric("API Calls", result.api_calls_made)
                    
                    st.json(result.json_output)
                    
                    # Download
                    json_str = json.dumps(result.json_output, indent=2)
                    st.download_button(
                        "📥 Download Converted JSON",
                        json_str,
                        file_name=f"converted_{uploaded_file.name}.json",
                        mime="application/json"
                    )
                
                except json.JSONDecodeError:
                    st.error("Invalid JSON schema")
                except Exception as e:
                    st.error(f"Conversion failed: {str(e)}")


def system_metrics_page():
    """System metrics page."""
    st.header("📊 System Metrics")
    
    # System status
    col1, col2, col3 = st.columns(3)
    col1.metric("LLM Status", "✅ Available" if st.session_state.llm_manager else "❌ Unavailable")
    col2.metric("Test Cases", len(load_test_cases()))
    col3.metric("Conversions Run", len(st.session_state.conversion_history))
    
    # Conversion history
    if st.session_state.conversion_history:
        st.subheader("Conversion History")
        
        history_data = []
        for entry in st.session_state.conversion_history[-10:]:  # Last 10
            result = entry["result"]
            history_data.append({
                "Timestamp": time.strftime("%H:%M:%S", time.localtime(entry["timestamp"])),
                "Input Preview": entry["input_text"],
                "Confidence": f"{result.confidence_score:.2%}",
                "Processing Time": f"{result.processing_time:.2f}s",
                "API Calls": result.api_calls_made,
                "Validation Errors": len(result.validation_errors)
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df)
        
        # Performance chart
        if len(history_data) > 1:
            fig = px.line(
                df, 
                x="Timestamp", 
                y=["Processing Time"],
                title="Processing Time Trend"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No conversions performed yet.")
    
    # System capabilities
    st.subheader("System Capabilities")
    capabilities = {
        "Supported Input Formats": ["Text", "PDF", "DOCX", "CSV", "Excel", "JSON"],
        "Max Context Tokens": "128,000",
        "Max Output Tokens": "8,192",
        "Schema Complexity Levels": ["Simple", "Moderate", "Complex", "Very Complex"],
        "Multi-pass Processing": "✅ Available",
        "Schema Chunking": "✅ Available",
        "Validation Loop": "✅ Available"
    }
    
    for key, value in capabilities.items():
        if isinstance(value, list):
            st.write(f"**{key}:** {', '.join(value)}")
        else:
            st.write(f"**{key}:** {value}")


if __name__ == "__main__":
    main() 
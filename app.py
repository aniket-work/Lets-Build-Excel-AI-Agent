import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import io
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ExcelProcessor:
    """Enhanced Excel processing with AI capabilities"""
    
    def __init__(self):
        self.dataframes = {}
        self.sheet_info = {}
        self.processed_content = []
        
    def load_excel_file(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Load Excel file and return all sheets as DataFrames"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheets_data = {}
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets_data[sheet_name] = df
                
                # Store metadata
                self.sheet_info[sheet_name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist(),
                    'data_types': df.dtypes.to_dict()
                }
                
            self.dataframes = sheets_data
            return sheets_data
            
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            return {}
    
    def analyze_data_structure(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Analyze DataFrame structure and create text description"""
        analysis = []
        
        # Basic info
        analysis.append(f"Sheet: {sheet_name}")
        analysis.append(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        analysis.append(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Data types
        analysis.append("\nData Types:")
        for col, dtype in df.dtypes.items():
            analysis.append(f"  - {col}: {dtype}")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            analysis.append("\nMissing Values:")
            for col, count in missing.items():
                if count > 0:
                    analysis.append(f"  - {col}: {count} missing")
        
        # Sample data
        analysis.append("\nSample Data (first 3 rows):")
        for i, row in df.head(3).iterrows():
            analysis.append(f"Row {i+1}: {row.to_dict()}")
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis.append("\nNumeric Summary:")
            for col in numeric_cols:
                stats = df[col].describe()
                analysis.append(f"  - {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
        
        return "\n".join(analysis)
    
    def create_searchable_content(self) -> List[Document]:
        """Create searchable documents from Excel content"""
        documents = []
        
        for sheet_name, df in self.dataframes.items():
            # Create comprehensive text representation
            content = self.analyze_data_structure(df, sheet_name)
            
            # Add each row as context
            for idx, row in df.iterrows():
                row_content = f"Sheet: {sheet_name}, Row {idx+1}: "
                row_content += ", ".join([f"{col}={val}" for col, val in row.items() if pd.notna(val)])
                
                doc = Document(
                    page_content=row_content,
                    metadata={
                        "sheet_name": sheet_name,
                        "row_number": idx + 1,
                        "type": "data_row"
                    }
                )
                documents.append(doc)
            
            # Add sheet summary
            summary_doc = Document(
                page_content=content,
                metadata={
                    "sheet_name": sheet_name,
                    "type": "sheet_summary"
                }
            )
            documents.append(summary_doc)
        
        return documents
    
    def generate_insights(self, df: pd.DataFrame, sheet_name: str) -> List[str]:
        """Generate automated insights about the data"""
        insights = []
        
        # Basic insights
        insights.append(f"📊 {sheet_name} contains {len(df)} records with {len(df.columns)} columns")
        
        # Missing data insights
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        high_missing = missing_pct[missing_pct > 10]
        if not high_missing.empty:
            insights.append(f"⚠️ High missing data: {', '.join([f'{col} ({pct}%)' for col, pct in high_missing.items()])}")
        
        # Numeric insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() > 1:
                insights.append(f"📈 {col}: Range {df[col].min():.2f} to {df[col].max():.2f}")
        
        # Categorical insights
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            unique_count = df[col].nunique()
            if unique_count < 10:
                insights.append(f"🏷️ {col}: {unique_count} unique values")
        
        return insights

class ExcelAIAgent:
    """Main AI Agent for Excel analysis"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.processor = ExcelProcessor()
        self.vectorstore = None
        self.conversation_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    def setup_rag_chain(self, documents: List[Document]):
        """Setup RAG chain with Excel documents"""
        if not documents:
            st.error("No documents to process")
            return
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings and vectorstore
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Create conversation chain
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=self.openai_api_key,
            model_name="gpt-4-turbo-preview"
        )
        
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the Excel data using AI"""
        if not self.conversation_chain:
            return {"error": "RAG chain not initialized"}
        
        try:
            result = self.conversation_chain({"question": question})
            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", [])
            }
        except Exception as e:
            return {"error": f"Query failed: {str(e)}"}
    
    def generate_visualization_code(self, query: str, sheet_name: str) -> str:
        """Generate Python code for data visualization"""
        df_ref = f"st.session_state.agent.processor.dataframes['{sheet_name}']"
        
        # Simple visualization suggestions based on query
        if "bar" in query.lower() or "column" in query.lower():
            return f"""
import plotly.express as px
df = {df_ref}
fig = px.bar(df, x=df.columns[0], y=df.columns[1] if len(df.columns) > 1 else df.columns[0])
st.plotly_chart(fig)
"""
        elif "line" in query.lower() or "trend" in query.lower():
            return f"""
import plotly.express as px
df = {df_ref}
fig = px.line(df, x=df.columns[0], y=df.columns[1] if len(df.columns) > 1 else df.columns[0])
st.plotly_chart(fig)
"""
        elif "scatter" in query.lower():
            return f"""
import plotly.express as px
df = {df_ref}
fig = px.scatter(df, x=df.columns[0], y=df.columns[1] if len(df.columns) > 1 else df.columns[0])
st.plotly_chart(fig)
"""
        else:
            return f"""
import plotly.express as px
df = {df_ref}
# Show basic info about the dataframe
st.write(f"Shape: {{df.shape}}")
st.write(f"Columns: {{df.columns.tolist()}}")
st.dataframe(df.head())
"""

def apply_anthropic_theme():
    """Apply Anthropic-inspired theme to Streamlit"""
    st.markdown("""
    <style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles - Dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main content styling */
    .main .block-container {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Text colors */
    .stMarkdown, .stText, p, span, div {
        color: #e2e8f0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Card styling - Dark cards */
    .stContainer, .element-container {
        background: rgba(51, 65, 85, 0.6) !important;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4);
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(51, 65, 85, 0.8) !important;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e2e8f0 !important;
    }
    
    .stChatMessage p {
        color: #e2e8f0 !important;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(51, 65, 85, 0.6) !important;
        border-radius: 12px;
        padding: 2rem;
        border: 2px dashed #64748b;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Metrics styling */
    .stMetric {
        background: rgba(51, 65, 85, 0.8) !important;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e2e8f0 !important;
    }
    
    .stMetric label {
        color: #94a3b8 !important;
    }
    
    .stMetric div {
        color: #f1f5f9 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(51, 65, 85, 0.6);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-weight: 500;
        color: #e2e8f0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border-radius: 8px;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: white !important;
        border-radius: 8px;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: white !important;
        border-radius: 8px;
        border: none;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border-radius: 8px;
        border: none;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(30, 41, 59, 0.8) !important;
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        background: rgba(51, 65, 85, 0.8) !important;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(51, 65, 85, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: rgba(51, 65, 85, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Loading spinner */
    .stSpinner {
        border-color: #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Excel AI Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom theme
    apply_anthropic_theme()
    
    # Main header with dark theme
    st.markdown("""
    <div style="background: linear-gradient(135deg, #374151 0%, #4b5563 100%); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);">
        <h1 style="color: #ffffff; font-weight: 700; font-size: 2.5rem; margin: 0;">🤖 Excel AI Agent</h1>
        <p style="color: #e2e8f0; font-size: 1.1rem; margin: 0.5rem 0 0 0;">Intelligent spreadsheet analysis powered by AI • Upload, ask, analyze</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for OpenAI API key with styled error
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key or openai_api_key == "your_openai_api_key_here":
        st.error("🔑 API Key Required")
        st.markdown("""
        **Please set your OpenAI API key in the .env file to continue.**
        
        **Setup Steps:**
        1. Open the `.env` file in this directory
        2. Replace `your_openai_api_key_here` with your actual OpenAI API key
        3. Save the file and refresh this page
        
        **Get your API key:** https://platform.openai.com/api-keys
        """)
        st.stop()
    
    # File upload section
    st.subheader("📁 Upload Your Excel File")
    st.markdown("Choose an Excel file (.xlsx or .xls) to start analyzing your data with AI")
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload an Excel file (.xlsx or .xls) to start analyzing your data",
        label_visibility="collapsed"
    )
    
    # Always show chat input - even before file upload
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("💬 Chat with Your Data")
    with col2:
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.experimental_rerun()
    
    st.markdown("Ask questions about your Excel data in natural language")
    
    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Show existing chat messages first
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])
    
    # Chat input that's always visible
    if not st.session_state.get('agent') or not st.session_state.get('agent').processor.dataframes:
        # Before file upload - show input but with message
        prompt = st.chat_input("Upload an Excel file first, then ask questions here...")
        if prompt:
            st.warning("Please upload an Excel file first before asking questions!")
            # Add a sample message to show how it works
            if not st.session_state.messages:
                st.session_state.messages.append({"role": "assistant", "content": "👋 Hi! I'm your Excel AI assistant. Upload an Excel file above and I'll help you analyze it. You can ask me questions like 'What columns are in my data?' or 'Show me a summary'."})
                st.experimental_rerun()
    else:
        # After file upload - fully functional chat
        if prompt := st.chat_input("Type your question about the Excel data here..."):
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Show user message immediately
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            
            # Generate and show response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🧠 Analyzing your data..."):
                    try:
                        result = st.session_state.agent.query(prompt)
                        
                        if "error" in result:
                            response = f"❌ Sorry, I encountered an error: {result['error']}"
                        else:
                            response = result["answer"]
                            
                            # Show sources if available
                            if result.get("source_documents"):
                                response += "\n\n📚 **Sources:**"
                                for i, doc in enumerate(result["source_documents"][:2]):
                                    sheet = doc.metadata.get("sheet_name", "Unknown")
                                    response += f"\n- Sheet: {sheet}"
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"❌ Error processing your question: {str(e)}"
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Force refresh to show new messages
            st.experimental_rerun()
    
    if uploaded_file is not None:
        if 'agent' not in st.session_state:
            st.session_state.agent = ExcelAIAgent(openai_api_key)
        
        # Process uploaded file with modern loading
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Load Excel file
            with st.spinner("🔄 Processing your Excel file..."):
                sheets_data = st.session_state.agent.processor.load_excel_file(tmp_file_path)
            
            if sheets_data:
                st.success(f"✅ Excel file loaded successfully! Found {len(sheets_data)} sheet(s). You can now ask questions in the chat above!")
                
                # Setup RAG chain
                with st.spinner("🧠 Setting up AI analysis engine..."):
                    documents = st.session_state.agent.processor.create_searchable_content()
                    st.session_state.agent.setup_rag_chain(documents)
                
                # Show sheet information
                st.subheader("📋 Available Sheets")
                for sheet_name in sheets_data.keys():
                    sheet_info = st.session_state.agent.processor.sheet_info[sheet_name]
                    st.info(f"📄 **{sheet_name}**: {sheet_info['rows']} rows × {sheet_info['columns']} columns")
                
        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)
    
    # Additional tabs for more functionality (only show if file is uploaded)
    if 'agent' in st.session_state and st.session_state.agent.processor.dataframes:
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "📈 Smart Insights", "🔍 Query Builder"])
        
        with tab1:
            st.subheader("📊 Data Explorer")
            st.markdown("Browse and understand your data structure")
            
            # Sheet selector
            sheet_names = list(st.session_state.agent.processor.dataframes.keys())
            selected_sheet = st.selectbox("Select Sheet", sheet_names, key="explorer_sheet")
            
            if selected_sheet:
                df = st.session_state.agent.processor.dataframes[selected_sheet]
                
                # Show basic info in metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                # Show data
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Show column info
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Missing': df.isnull().sum(),
                    'Unique': df.nunique()
                })
                st.dataframe(col_info, use_container_width=True)
        
        with tab2:
            st.markdown("""
            <div class="claude-card">
                <h3 style="margin-top: 0; color: #1e293b;">📈 Smart Insights</h3>
                <p style="color: #64748b;">Automated analysis and visualizations</p>
            </div>
            """, unsafe_allow_html=True)
            
            sheet_names = list(st.session_state.agent.processor.dataframes.keys())
            selected_sheet = st.selectbox("Select Sheet for Insights", sheet_names, key="insights_sheet")
            
            if selected_sheet:
                df = st.session_state.agent.processor.dataframes[selected_sheet]
                insights = st.session_state.agent.processor.generate_insights(df, selected_sheet)
                
                st.markdown("""
                <div class="claude-card">
                    <h4 style="margin-top: 0; color: #1e293b;">Key Insights</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for insight in insights:
                    st.markdown(f"""
                    <div class="claude-card" style="margin: 0.5rem 0; padding: 1rem; border-left: 4px solid #3b82f6;">
                        {insight}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Generate basic visualizations
                st.markdown("""
                <div class="claude-card">
                    <h4 style="margin-top: 0; color: #1e293b;">Smart Visualizations</h4>
                </div>
                """, unsafe_allow_html=True)
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # Correlation heatmap
                    if len(numeric_cols) > 1:
                        fig = px.imshow(df[numeric_cols].corr(), 
                                      title="Correlation Matrix",
                                      color_continuous_scale="RdBu")
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_family="Inter"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Distribution plots
                    for col in numeric_cols[:3]:  # Limit to first 3 columns
                        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_family="Inter"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("""
            <div class="claude-card">
                <h3 style="margin-top: 0; color: #1e293b;">🔍 Query Builder</h3>
                <p style="color: #64748b;">Build custom queries and filters</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Query builder interface
            sheet_names = list(st.session_state.agent.processor.dataframes.keys())
            selected_sheet = st.selectbox("Select Sheet for Query", sheet_names, key="query_sheet")
            
            if selected_sheet:
                df = st.session_state.agent.processor.dataframes[selected_sheet]
                
                # Query type selection
                query_type = st.selectbox(
                    "Query Type",
                    ["Filter Data", "Aggregation", "Custom Query"]
                )
                
                if query_type == "Filter Data":
                    col = st.selectbox("Select Column", df.columns)
                    if df[col].dtype == 'object':
                        unique_vals = df[col].unique()
                        selected_vals = st.multiselect("Select Values", unique_vals)
                        if selected_vals:
                            filtered_df = df[df[col].isin(selected_vals)]
                            st.markdown("""
                            <div class="claude-card">
                                <h4 style="margin-top: 0; color: #1e293b;">Filtered Results</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            st.dataframe(filtered_df, use_container_width=True)
                    else:
                        min_val, max_val = float(df[col].min()), float(df[col].max())
                        range_vals = st.slider("Select Range", min_val, max_val, (min_val, max_val))
                        filtered_df = df[(df[col] >= range_vals[0]) & (df[col] <= range_vals[1])]
                        st.markdown("""
                        <div class="claude-card">
                            <h4 style="margin-top: 0; color: #1e293b;">Filtered Results</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.dataframe(filtered_df, use_container_width=True)
                
                elif query_type == "Aggregation":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        col = st.selectbox("Select Numeric Column", numeric_cols)
                        agg_func = st.selectbox("Aggregation Function", ["sum", "mean", "count", "min", "max"])
                        
                        if st.button("Calculate", type="primary"):
                            result = getattr(df[col], agg_func)()
                            st.markdown("""
                            <div class="claude-card" style="text-align: center;">
                                <h2 style="color: #1e293b; margin: 0;">{:.2f}</h2>
                                <p style="color: #64748b; margin: 0;">{} of {}</p>
                            </div>
                            """.format(result, agg_func.title(), col), unsafe_allow_html=True)
                
                elif query_type == "Custom Query":
                    query = st.text_area("Enter your question about the data:")
                    if st.button("Execute Query", type="primary") and query:
                        with st.spinner("Processing query..."):
                            result = st.session_state.agent.query(query)
                            if "error" not in result:
                                st.markdown("""
                                <div class="claude-card">
                                    <h4 style="margin-top: 0; color: #1e293b;">Query Result</h4>
                                    <p>{}</p>
                                </div>
                                """.format(result["answer"]), unsafe_allow_html=True)
                            else:
                                st.error(result["error"])
    
    else:
        # Welcome screen with proper Streamlit components
        st.markdown("## Welcome to Excel AI Agent")
        st.markdown("Upload an Excel file above to start analyzing your data with artificial intelligence.")
        
        st.markdown("### What this app can do:")
        
        # Create feature cards using columns
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container():
                st.markdown("""
                <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 1rem;">
                    <h4 style="margin-top: 0; color: #1e293b;">📊 Smart Analysis</h4>
                    <p style="color: #64748b; margin: 0;">Load and analyze Excel files with multiple sheets</p>
                </div>
                """, unsafe_allow_html=True)
            
            with st.container():
                st.markdown("""
                <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #f59e0b; margin-bottom: 1rem;">
                    <h4 style="margin-top: 0; color: #1e293b;">� Auto Insights</h4>
                    <p style="color: #64748b; margin: 0;">Generate insights and visualizations automatically</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown("""
                <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #10b981; margin-bottom: 1rem;">
                    <h4 style="margin-top: 0; color: #1e293b;">� Natural Chat</h4>
                    <p style="color: #64748b; margin: 0;">Ask questions about your data in plain English</p>
                </div>
                """, unsafe_allow_html=True)
            
            with st.container():
                st.markdown("""
                <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #ef4444; margin-bottom: 1rem;">
                    <h4 style="margin-top: 0; color: #1e293b;">🔍 Custom Queries</h4>
                    <p style="color: #64748b; margin: 0;">Build advanced queries and filters</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    
    # Add some example usage info
    st.sidebar.markdown("""
    ## 💡 Example Questions:
    - "What columns are in my data?"
    - "Show me a summary"
    - "What's the average of column X?"
    - "Create a chart"
    - "Find records where price > 100"
    """)
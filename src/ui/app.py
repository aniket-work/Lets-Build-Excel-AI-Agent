"""
Advanced Streamlit UI Application for Excel AI Agent

This module provides a sophisticated, modular UI layer with modern design patterns,
intelligent state management, and professional component architecture.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go

from ..core.data_processor import IntelligentDataProcessor, DatasetMetrics
from ..services.ai_analytics import IntelligentAnalyticsEngine, QueryResponse
from config.settings import AdvancedApplicationSettings


class ModernUITheme:
    """Advanced theming system for professional UI design"""
    
    @staticmethod
    def apply_theme():
        """Apply sophisticated dark theme with modern design principles"""
        st.markdown("""
        <style>
        /* Import Inter font for professional typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global application styling */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Enhanced container styling */
        .main .block-container {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        
        /* Professional header design */
        .header-container {
            background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Advanced card components */
        .insight-card {
            background: rgba(51, 65, 85, 0.8);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .insight-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }
        
        /* Enhanced button styling */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 500;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            transform: translateY(-1px);
            box-shadow: 0 8px 32px rgba(59, 130, 246, 0.4);
        }
        
        /* Professional chat interface */
        .stChatMessage {
            background: rgba(51, 65, 85, 0.8);
            border-radius: 12px;
            margin: 0.5rem 0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #e2e8f0;
        }
        
        /* Enhanced metrics display */
        .metric-container {
            background: rgba(51, 65, 85, 0.8);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        
        /* Advanced tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(30, 41, 59, 0.8);
            border-radius: 12px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(51, 65, 85, 0.6);
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-weight: 500;
            color: #e2e8f0;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        </style>
        """, unsafe_allow_html=True)


class ChatInterface:
    """Advanced chat interface with intelligent conversation management"""
    
    def __init__(self, analytics_engine: Optional[IntelligentAnalyticsEngine] = None):
        self.analytics_engine = analytics_engine
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for chat management"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "conversation_context" not in st.session_state:
            st.session_state.conversation_context = {}
    
    def render_chat_history(self):
        """Render existing chat messages with enhanced styling"""
        for message in st.session_state.messages:
            avatar = "🤖" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    
    def handle_user_input(self) -> Optional[str]:
        """Process user input and generate AI response"""
        if not self.analytics_engine or not self.analytics_engine.is_ready():
            prompt = st.chat_input("Upload an Excel file first to start analyzing...")
            if prompt:
                st.warning("🔗 Please upload an Excel file first before asking questions!")
                self._add_welcome_message()
            return None
        
        prompt = st.chat_input("Ask me anything about your Excel data...")
        if prompt:
            return self._process_user_query(prompt)
        return None
    
    def _process_user_query(self, prompt: str) -> str:
        """Process user query and generate intelligent response"""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate and display AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Analyzing your data with advanced AI..."):
                try:
                    response = self.analytics_engine.process_query(prompt)
                    formatted_response = self._format_response(response)
                    st.markdown(formatted_response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": formatted_response
                    })
                    return formatted_response
                except Exception as e:
                    error_msg = f"❌ Analysis Error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
                    return error_msg
    
    def _format_response(self, response: QueryResponse) -> str:
        """Format AI response with rich context and sources"""
        formatted = response.response
        
        if response.confidence_score < 0.7:
            formatted += f"\n\n⚠️ **Confidence: {response.confidence_score:.1%}** - Please verify results"
        
        if response.sources:
            formatted += "\n\n📚 **Data Sources:**"
            for source in response.sources[:3]:  # Limit sources
                sheet = source.get("sheet_name", "Unknown Sheet")
                formatted += f"\n- 📄 {sheet}"
        
        if response.suggested_followups:
            formatted += "\n\n💡 **Try asking:**"
            for suggestion in response.suggested_followups[:2]:
                formatted += f"\n- {suggestion}"
        
        return formatted
    
    def _add_welcome_message(self):
        """Add welcoming message for new users"""
        if not st.session_state.messages:
            welcome_msg = (
                "👋 **Welcome to Excel AI Agent!** I'm your intelligent data assistant.\n\n"
                "Upload an Excel file above and I'll help you:\n"
                "• 📊 Analyze data patterns and trends\n"
                "• 🔍 Answer questions about your data\n"
                "• 📈 Generate insights and visualizations\n"
                "• 🎯 Build custom queries and filters"
            )
            st.session_state.messages.append({
                "role": "assistant", 
                "content": welcome_msg
            })
    
    def reset_conversation(self):
        """Reset chat conversation"""
        st.session_state.messages = []
        st.session_state.conversation_context = {}


class DataExplorer:
    """Advanced data exploration interface"""
    
    def __init__(self, data_processor: IntelligentDataProcessor):
        self.data_processor = data_processor
    
    def render_explorer_tab(self):
        """Render comprehensive data exploration interface"""
        st.markdown("### 📊 Advanced Data Explorer")
        st.markdown("*Explore your data structure with intelligent analysis*")
        
        if not self.data_processor.datasets:
            st.info("🔗 Upload an Excel file to start exploring your data")
            return
        
        # Sheet selection with enhanced info
        sheet_names = list(self.data_processor.datasets.keys())
        selected_sheet = st.selectbox(
            "Select Dataset Sheet", 
            sheet_names, 
            key="explorer_sheet",
            help="Choose a sheet to explore its structure and content"
        )
        
        if selected_sheet:
            metrics = self.data_processor.dataset_metrics[selected_sheet]
            df = self.data_processor.datasets[selected_sheet]
            
            # Enhanced metrics display
            self._render_dataset_overview(metrics, df)
            
            # Advanced data preview
            self._render_data_preview(df, selected_sheet)
            
            # Intelligent column analysis
            self._render_column_analysis(df)
    
    def _render_dataset_overview(self, metrics: DatasetMetrics, df: pd.DataFrame):
        """Render comprehensive dataset overview"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3 style="color: #3b82f6; margin: 0;">{metrics.row_count:,}</h3>
                <p style="color: #94a3b8; margin: 0;">Total Rows</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h3 style="color: #10b981; margin: 0;">{metrics.column_count}</h3>
                <p style="color: #94a3b8; margin: 0;">Columns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            missing_pct = (metrics.missing_values_total / (metrics.row_count * metrics.column_count)) * 100
            st.markdown(f"""
            <div class="metric-container">
                <h3 style="color: #f59e0b; margin: 0;">{missing_pct:.1f}%</h3>
                <p style="color: #94a3b8; margin: 0;">Missing Data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            data_quality = max(0, 100 - missing_pct - (len(metrics.data_quality_issues) * 10))
            color = "#10b981" if data_quality > 80 else "#f59e0b" if data_quality > 60 else "#ef4444"
            st.markdown(f"""
            <div class="metric-container">
                <h3 style="color: {color}; margin: 0;">{data_quality:.0f}%</h3>
                <p style="color: #94a3b8; margin: 0;">Data Quality</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_data_preview(self, df: pd.DataFrame, sheet_name: str):
        """Render intelligent data preview"""
        st.markdown("#### 📋 Data Preview")
        
        # Sample size selector
        sample_size = st.slider("Preview Rows", 5, min(100, len(df)), 10, key=f"preview_{sheet_name}")
        
        # Display enhanced dataframe
        st.dataframe(
            df.head(sample_size), 
            use_container_width=True,
            height=400
        )
    
    def _render_column_analysis(self, df: pd.DataFrame):
        """Render intelligent column analysis"""
        st.markdown("#### 🔍 Column Analysis")
        
        # Create comprehensive column information
        column_info = []
        for col in df.columns:
            col_data = df[col]
            info = {
                'Column': col,
                'Type': str(col_data.dtype),
                'Non-Null': col_data.count(),
                'Null %': f"{(col_data.isnull().sum() / len(df) * 100):.1f}%",
                'Unique': col_data.nunique(),
                'Memory': f"{col_data.memory_usage(deep=True) / 1024:.1f} KB"
            }
            
            # Add type-specific insights
            if col_data.dtype in ['int64', 'float64']:
                info.update({
                    'Min': f"{col_data.min():.2f}" if pd.notna(col_data.min()) else "N/A",
                    'Max': f"{col_data.max():.2f}" if pd.notna(col_data.max()) else "N/A",
                    'Mean': f"{col_data.mean():.2f}" if pd.notna(col_data.mean()) else "N/A"
                })
            else:
                info.update({
                    'Min': "N/A",
                    'Max': "N/A", 
                    'Mean': "N/A"
                })
            
            column_info.append(info)
        
        # Display as enhanced dataframe
        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df, use_container_width=True)


class VisualizationEngine:
    """Advanced visualization generation and rendering"""
    
    def __init__(self, data_processor: IntelligentDataProcessor):
        self.data_processor = data_processor
    
    def render_insights_tab(self):
        """Render intelligent insights and visualizations"""
        st.markdown("### 📈 Intelligent Insights")
        st.markdown("*AI-powered analysis and automated visualizations*")
        
        if not self.data_processor.datasets:
            st.info("🔗 Upload an Excel file to generate intelligent insights")
            return
        
        # Sheet selection for insights
        sheet_names = list(self.data_processor.datasets.keys())
        selected_sheet = st.selectbox(
            "Select Sheet for Analysis", 
            sheet_names, 
            key="insights_sheet"
        )
        
        if selected_sheet:
            self._render_sheet_insights(selected_sheet)
            self._render_automated_visualizations(selected_sheet)
    
    def _render_sheet_insights(self, sheet_name: str):
        """Render AI-generated insights for a sheet"""
        metrics = self.data_processor.dataset_metrics[sheet_name]
        
        st.markdown("#### 🧠 AI-Generated Insights")
        
        insights = [
            f"📊 Dataset contains **{metrics.row_count:,} records** across **{metrics.column_count} dimensions**",
            f"🔢 **{len(metrics.numeric_columns)} numeric columns** available for quantitative analysis",
            f"📝 **{len(metrics.categorical_columns)} categorical columns** for grouping and segmentation",
            f"⚠️ **{metrics.missing_values_total:,} missing values** detected across the dataset"
        ]
        
        # Add data quality insights
        if metrics.data_quality_issues:
            insights.append(f"🔍 **{len(metrics.data_quality_issues)} data quality issues** require attention")
        
        # Display insights in cards
        for insight in insights:
            st.markdown(f"""
            <div class="insight-card">
                <p style="margin: 0; font-size: 1.1rem;">{insight}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_automated_visualizations(self, sheet_name: str):
        """Generate and render automated visualizations"""
        st.markdown("#### 📊 Automated Visualizations")
        
        df = self.data_processor.datasets[sheet_name]
        metrics = self.data_processor.dataset_metrics[sheet_name]
        
        # Correlation analysis for numeric data
        if len(metrics.numeric_columns) > 1:
            st.markdown("##### Correlation Matrix")
            numeric_df = df[metrics.numeric_columns]
            
            fig = px.imshow(
                numeric_df.corr(),
                title="Feature Correlation Analysis",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Inter",
                title_font_size=16
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution analysis
        if metrics.numeric_columns:
            st.markdown("##### Distribution Analysis")
            selected_column = st.selectbox(
                "Analyze Column Distribution", 
                metrics.numeric_columns,
                key=f"dist_{sheet_name}"
            )
            
            fig = px.histogram(
                df, 
                x=selected_column,
                title=f"Distribution of {selected_column}",
                marginal="box",
                color_discrete_sequence=["#3b82f6"]
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Inter"
            )
            st.plotly_chart(fig, use_container_width=True)


class ExcelAIApplication:
    """Main application controller with advanced architecture"""
    
    def __init__(self):
        self.settings = AdvancedApplicationSettings()
        self.data_processor = IntelligentDataProcessor(self.settings)
        self.analytics_engine = None
        self.chat_interface = None
        self.data_explorer = None
        self.visualization_engine = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all application components"""
        # Initialize UI components
        self.data_explorer = DataExplorer(self.data_processor)
        self.visualization_engine = VisualizationEngine(self.data_processor)
        
        # Initialize analytics engine if API key is available
        if self.settings.openai_api_key and self.settings.openai_api_key != "your_openai_api_key_here":
            self.analytics_engine = IntelligentAnalyticsEngine(
                self.settings, 
                self.data_processor
            )
            self.chat_interface = ChatInterface(self.analytics_engine)
        else:
            self.chat_interface = ChatInterface(None)
    
    def run(self):
        """Run the main application"""
        # Configure page
        st.set_page_config(
            page_title="Excel AI Agent",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Apply modern theme
        ModernUITheme.apply_theme()
        
        # Render header
        self._render_header()
        
        # Check API key
        if not self._check_api_key():
            return
        
        # Render main interface
        self._render_main_interface()
    
    def _render_header(self):
        """Render professional application header"""
        st.markdown("""
        <div class="header-container">
            <h1 style="color: #ffffff; font-weight: 700; font-size: 2.5rem; margin: 0;">
                🤖 Excel AI Agent
            </h1>
            <p style="color: #e2e8f0; font-size: 1.2rem; margin: 0.5rem 0 0 0;">
                Intelligent spreadsheet analysis powered by advanced AI • Upload, explore, analyze
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _check_api_key(self) -> bool:
        """Validate OpenAI API key configuration"""
        if not self.settings.openai_api_key or self.settings.openai_api_key == "your_openai_api_key_here":
            st.error("🔑 **OpenAI API Key Required**")
            st.markdown("""
            **Configuration Steps:**
            1. Open the `.env` file in the project directory
            2. Replace `your_openai_api_key_here` with your actual OpenAI API key
            3. Save the file and refresh this page
            
            **Get your API key:** https://platform.openai.com/api-keys
            """)
            return False
        return True
    
    def _render_main_interface(self):
        """Render the main application interface"""
        # File upload section
        self._render_file_upload()
        
        # Chat interface
        self._render_chat_section()
        
        # Advanced features tabs
        if self.data_processor.datasets:
            self._render_advanced_features()
    
    def _render_file_upload(self):
        """Render intelligent file upload interface"""
        st.markdown("### 📁 Upload Excel File")
        st.markdown("*Select an Excel file (.xlsx or .xls) to begin intelligent analysis*")
        
        uploaded_file = st.file_uploader(
            "Choose Excel file",
            type=['xlsx', 'xls'],
            help="Upload Excel files with one or multiple sheets for AI analysis",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            self._process_uploaded_file(uploaded_file)
    
    def _process_uploaded_file(self, uploaded_file):
        """Process uploaded Excel file with advanced handling"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            with st.spinner("🔄 Processing Excel file with AI..."):
                success = self.data_processor.load_excel_file(tmp_file_path)
                
                if success:
                    st.success(
                        f"✅ **File processed successfully!** "
                        f"Loaded {len(self.data_processor.datasets)} sheet(s) "
                        f"with advanced AI analysis capabilities."
                    )
                    
                    # Initialize analytics if not already done
                    if self.analytics_engine and not self.analytics_engine.is_ready():
                        with st.spinner("🧠 Initializing AI analytics engine..."):
                            self.analytics_engine.initialize_vector_store()
                    
                    # Display sheet information
                    self._display_sheet_info()
                else:
                    st.error("❌ Failed to process Excel file. Please check file format.")
        
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
        finally:
            os.unlink(tmp_file_path)
    
    def _display_sheet_info(self):
        """Display comprehensive sheet information"""
        st.markdown("#### 📋 Available Datasets")
        
        for sheet_name, metrics in self.data_processor.dataset_metrics.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**📄 {sheet_name}**")
            with col2:
                st.markdown(f"*{metrics.row_count:,} rows*")
            with col3:
                st.markdown(f"*{metrics.column_count} columns*")
    
    def _render_chat_section(self):
        """Render intelligent chat interface"""
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown("### 💬 Chat with Your Data")
        with col2:
            if st.button("🔄 Reset Chat", help="Clear conversation history"):
                self.chat_interface.reset_conversation()
                st.experimental_rerun()
        
        st.markdown("*Ask questions about your Excel data using natural language*")
        
        # Render chat interface
        self.chat_interface.render_chat_history()
        response = self.chat_interface.handle_user_input()
        
        if response:
            st.experimental_rerun()
    
    def _render_advanced_features(self):
        """Render advanced feature tabs"""
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs([
            "📊 Data Explorer", 
            "📈 Smart Insights", 
            "🔍 Query Builder"
        ])
        
        with tab1:
            self.data_explorer.render_explorer_tab()
        
        with tab2:
            self.visualization_engine.render_insights_tab()
        
        with tab3:
            self._render_query_builder()
    
    def _render_query_builder(self):
        """Render advanced query builder interface"""
        st.markdown("### 🔍 Advanced Query Builder")
        st.markdown("*Build sophisticated data queries and custom analysis*")
        
        if not self.data_processor.datasets:
            st.info("🔗 Upload an Excel file to access query building features")
            return
        
        # Implementation would continue with query building interface
        st.info("🚧 Advanced query builder interface coming soon...")


def main():
    """Main application entry point"""
    app = ExcelAIApplication()
    app.run()


if __name__ == "__main__":
    main()

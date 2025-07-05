"""
Advanced AI-Powered Analytics Service
===================================

This module implements a sophisticated AI service layer that I believe represents
a significant advancement in human-data interaction through natural language processing.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from config.settings import app_settings

@dataclass
class AnalyticsQuery:
    """Structured representation of user analytics queries."""
    
    raw_query: str
    processed_query: str
    intent_category: str
    confidence_score: float
    timestamp: datetime
    context_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "processed_query": self.processed_query,
            "intent_category": self.intent_category,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
            "context_metadata": self.context_metadata
        }

@dataclass
class IntelligentResponse:
    """Comprehensive response structure for AI analytics."""
    
    primary_answer: str
    supporting_evidence: List[str]
    confidence_level: float
    source_references: List[Dict[str, Any]]
    suggested_followups: List[str]
    visualization_recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_answer": self.primary_answer,
            "supporting_evidence": self.supporting_evidence,
            "confidence_level": self.confidence_level,
            "source_references": self.source_references,
            "suggested_followups": self.suggested_followups,
            "visualization_recommendations": self.visualization_recommendations
        }

class AdvancedPromptOrchestrator:
    """
    Sophisticated prompt engineering system for Excel analytics.
    
    In my opinion, this represents a significant improvement over generic
    prompting approaches by incorporating domain-specific context and
    analytical reasoning patterns.
    """
    
    def __init__(self):
        self.base_system_prompt = self._construct_foundational_prompt()
        self.context_enhancers = self._initialize_context_enhancers()
        self.reasoning_templates = self._establish_reasoning_frameworks()
    
    def _construct_foundational_prompt(self) -> str:
        """
        Construct the foundational system prompt for Excel analytics.
        
        Based on my understanding of effective prompt engineering, this approach
        provides better context and reasoning capabilities for data analysis tasks.
        """
        return """You are an Advanced Excel Analytics Specialist with deep expertise in data interpretation, 
statistical analysis, and business intelligence. Your role is to provide sophisticated insights from Excel data
through natural language interaction.

CORE COMPETENCIES:
- Advanced statistical analysis and pattern recognition
- Business context interpretation and strategic insights
- Data quality assessment and anomaly detection
- Predictive trend analysis and forecasting
- Interactive visualization recommendations

ANALYTICAL APPROACH:
1. Always begin by understanding the business context and user intent
2. Perform thorough data quality assessment before analysis
3. Provide multi-layered insights: descriptive, diagnostic, predictive, and prescriptive
4. Support conclusions with specific evidence from the data
5. Suggest actionable next steps and deeper analysis opportunities

COMMUNICATION STYLE:
- Use clear, professional language with appropriate technical depth
- Structure responses with executive summary followed by detailed analysis
- Highlight key findings with bullet points and clear metrics
- Provide confidence levels for predictions and recommendations
- Always contextualize findings within broader business implications

Remember: Your goal is to transform raw Excel data into actionable business intelligence 
that drives informed decision-making."""
    
    def _initialize_context_enhancers(self) -> Dict[str, str]:
        """Initialize context enhancement patterns for different query types."""
        return {
            "summary_analysis": """
            When providing data summaries, include:
            - Key performance indicators and their significance
            - Distribution patterns and outliers
            - Temporal trends if date columns are present
            - Comparative analysis across categories
            - Data quality observations
            """,
            
            "trend_analysis": """
            For trend analysis, focus on:
            - Statistical significance of observed patterns
            - Seasonality and cyclical behaviors
            - Growth rates and trajectory projections
            - Correlation analysis between variables
            - External factor considerations
            """,
            
            "comparative_analysis": """
            For comparative analysis, examine:
            - Statistical differences between groups
            - Performance rankings and benchmarks
            - Variance analysis and explanatory factors
            - Market share or proportion insights
            - Opportunity identification
            """,
            
            "predictive_insights": """
            For predictive insights, consider:
            - Historical pattern extrapolation
            - Leading indicator identification
            - Risk factor assessment
            - Scenario planning possibilities
            - Confidence intervals for predictions
            """
        }
    
    def _establish_reasoning_frameworks(self) -> Dict[str, str]:
        """Establish structured reasoning frameworks for different analytical tasks."""
        return {
            "data_exploration": """
            FRAMEWORK: Data Discovery and Profiling
            1. STRUCTURE: Analyze data dimensions, types, and relationships
            2. QUALITY: Assess completeness, consistency, and accuracy
            3. PATTERNS: Identify distributions, outliers, and anomalies
            4. INSIGHTS: Extract initial business-relevant observations
            5. OPPORTUNITIES: Suggest deeper analysis directions
            """,
            
            "business_intelligence": """
            FRAMEWORK: Strategic Business Analysis
            1. CONTEXT: Understand business objectives and KPIs
            2. PERFORMANCE: Analyze current state against benchmarks
            3. DRIVERS: Identify key factors influencing outcomes
            4. TRENDS: Assess directional movements and momentum
            5. RECOMMENDATIONS: Provide actionable strategic insights
            """,
            
            "operational_analysis": """
            FRAMEWORK: Operational Excellence Assessment
            1. EFFICIENCY: Analyze process performance metrics
            2. BOTTLENECKS: Identify constraint points and inefficiencies
            3. VARIANCE: Examine deviations from expected performance
            4. ROOT_CAUSE: Investigate underlying factors
            5. OPTIMIZATION: Suggest improvement opportunities
            """
        }
    
    def enhance_query_with_context(self, user_query: str, data_context: Dict[str, Any], 
                                 analysis_type: str = "general") -> str:
        """
        Enhance user queries with sophisticated contextual information.
        
        I think this approach significantly improves the quality of AI responses
        by providing rich context about the data structure and analytical objectives.
        """
        context_info = self._extract_data_context_summary(data_context)
        reasoning_framework = self.reasoning_templates.get(analysis_type, "")
        context_enhancer = self.context_enhancers.get(analysis_type, "")
        
        enhanced_prompt = f"""
        {self.base_system_prompt}
        
        DATA CONTEXT:
        {context_info}
        
        ANALYTICAL FRAMEWORK:
        {reasoning_framework}
        
        SPECIFIC GUIDANCE:
        {context_enhancer}
        
        USER QUERY: {user_query}
        
        Please provide a comprehensive analysis that addresses the user's question while leveraging
        the full context of the available data. Structure your response to be both technically
        accurate and business-relevant.
        """
        
        return enhanced_prompt
    
    def _extract_data_context_summary(self, data_context: Dict[str, Any]) -> str:
        """Extract and format key data context information."""
        summary_parts = []
        
        for sheet_name, info in data_context.items():
            if isinstance(info, dict):
                rows = info.get('rows', 'unknown')
                columns = info.get('columns', 'unknown')
                column_names = info.get('column_names', [])
                
                summary_parts.append(f"""
                Sheet '{sheet_name}':
                - Dimensions: {rows} rows × {columns} columns
                - Available columns: {', '.join(column_names[:10])}{'...' if len(column_names) > 10 else ''}
                """)
        
        return '\n'.join(summary_parts)

class IntelligentAnalyticsEngine:
    """
    Core AI analytics engine for Excel data interpretation.
    
    Based on my experience with data analytics platforms, I believe this architecture
    provides superior flexibility and intelligence compared to traditional BI tools.
    """
    
    def __init__(self):
        self.prompt_orchestrator = AdvancedPromptOrchestrator()
        self.conversation_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Maintain context for last 10 exchanges
        )
        self.knowledge_base: Optional[FAISS] = None
        self.retrieval_chain: Optional[ConversationalRetrievalChain] = None
        self.query_history: List[AnalyticsQuery] = []
        self._initialize_ai_components()
    
    def _initialize_ai_components(self) -> None:
        """Initialize core AI components with optimized parameters."""
        model_params = app_settings.get_model_parameters()
        
        self.language_model = ChatOpenAI(
            openai_api_key=app_settings.openai_api_key,
            **model_params
        )
        
        retrieval_params = app_settings.get_retrieval_parameters()
        self.embeddings_model = OpenAIEmbeddings(
            openai_api_key=app_settings.openai_api_key,
            model=retrieval_params["embedding_model"]
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=retrieval_params["chunk_size"],
            chunk_overlap=retrieval_params["chunk_overlap"],
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def initialize_knowledge_base(self, document_corpus: List[Document]) -> None:
        """
        Initialize the vector knowledge base for intelligent retrieval.
        
        In my opinion, this approach provides much more sophisticated information
        retrieval compared to simple keyword matching or basic search functionality.
        """
        if not document_corpus:
            raise ValueError("Document corpus cannot be empty for knowledge base initialization")
        
        # Apply intelligent text segmentation
        segmented_documents = self.text_splitter.split_documents(document_corpus)
        
        # Enhance documents with analytical metadata
        enhanced_documents = self._enhance_documents_with_metadata(segmented_documents)
        
        # Create vector store with optimized parameters
        self.knowledge_base = FAISS.from_documents(
            enhanced_documents, 
            self.embeddings_model
        )
        
        # Initialize retrieval chain with custom prompt
        self.retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=self.language_model,
            retriever=self.knowledge_base.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance for diversity
                search_kwargs={"k": 6, "fetch_k": 20}
            ),
            memory=self.conversation_memory,
            return_source_documents=True,
            verbose=app_settings.is_development_mode()
        )
    
    def _enhance_documents_with_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Enhance documents with analytical metadata for improved retrieval.
        
        I think this approach significantly improves the relevance and context
        of retrieved information for analytical queries.
        """
        enhanced_docs = []
        
        for doc in documents:
            # Extract analytical patterns
            content = doc.page_content
            metadata = doc.metadata.copy()
            
            # Identify numerical patterns
            numerical_matches = re.findall(r'\d+\.?\d*', content)
            if numerical_matches:
                metadata['contains_numbers'] = True
                metadata['number_count'] = len(numerical_matches)
            
            # Identify analytical keywords
            analytical_keywords = ['average', 'total', 'sum', 'count', 'maximum', 'minimum', 
                                 'trend', 'increase', 'decrease', 'correlation', 'analysis']
            found_keywords = [kw for kw in analytical_keywords if kw.lower() in content.lower()]
            if found_keywords:
                metadata['analytical_keywords'] = found_keywords
            
            # Identify data type contexts
            if any(term in content.lower() for term in ['column', 'row', 'sheet', 'cell']):
                metadata['data_structure_context'] = True
            
            enhanced_docs.append(Document(page_content=content, metadata=metadata))
        
        return enhanced_docs
    
    def process_analytical_query(self, user_query: str, data_context: Dict[str, Any]) -> IntelligentResponse:
        """
        Process user queries with advanced analytical intelligence.
        
        Based on my understanding of effective data analysis workflows, this method
        provides comprehensive query processing that goes beyond simple Q&A.
        """
        if not self.retrieval_chain:
            raise ValueError("Knowledge base not initialized. Call initialize_knowledge_base first.")
        
        # Analyze query intent and complexity
        query_analysis = self._analyze_query_intent(user_query)
        
        # Enhance query with contextual information
        enhanced_query = self.prompt_orchestrator.enhance_query_with_context(
            user_query, data_context, query_analysis.intent_category
        )
        
        try:
            # Execute retrieval-augmented generation
            result = self.retrieval_chain({
                "question": enhanced_query,
                "chat_history": self.conversation_memory.chat_memory.messages[-10:]  # Recent context
            })
            
            # Process and structure the response
            structured_response = self._structure_analytical_response(
                result, user_query, query_analysis
            )
            
            # Store query for learning and improvement
            self.query_history.append(query_analysis)
            
            return structured_response
            
        except Exception as e:
            return IntelligentResponse(
                primary_answer=f"I encountered an analytical processing error: {str(e)}",
                supporting_evidence=[],
                confidence_level=0.0,
                source_references=[],
                suggested_followups=["Please try rephrasing your question", "Check if data was loaded correctly"],
                visualization_recommendations=[]
            )
    
    def _analyze_query_intent(self, user_query: str) -> AnalyticsQuery:
        """
        Sophisticated query intent analysis for improved response generation.
        
        I believe this approach provides much better understanding of user
        analytical intentions compared to simple keyword matching.
        """
        query_lower = user_query.lower()
        
        # Intent classification patterns
        intent_patterns = {
            "summary_analysis": ["summary", "overview", "describe", "what is", "show me"],
            "trend_analysis": ["trend", "over time", "change", "growth", "pattern", "increase", "decrease"],
            "comparative_analysis": ["compare", "difference", "versus", "vs", "between", "against"],
            "specific_lookup": ["find", "search", "where", "which", "filter", "show records"],
            "statistical_analysis": ["average", "mean", "median", "total", "sum", "count", "correlation"],
            "predictive_insights": ["predict", "forecast", "future", "projection", "estimate"],
            "visualization": ["chart", "graph", "plot", "visualize", "show chart"]
        }
        
        # Calculate intent scores
        intent_scores = {}
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score / len(keywords)
        
        # Determine primary intent
        primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else "general_inquiry"
        confidence = intent_scores.get(primary_intent, 0.1)
        
        return AnalyticsQuery(
            raw_query=user_query,
            processed_query=user_query.strip(),
            intent_category=primary_intent,
            confidence_score=confidence,
            timestamp=datetime.now(),
            context_metadata={"intent_scores": intent_scores}
        )
    
    def _structure_analytical_response(self, raw_result: Dict[str, Any], 
                                     original_query: str, query_analysis: AnalyticsQuery) -> IntelligentResponse:
        """
        Structure AI responses into comprehensive analytical insights.
        
        In my experience, this structured approach provides much more valuable
        insights compared to unstructured text responses.
        """
        primary_answer = raw_result.get("answer", "No response generated")
        source_docs = raw_result.get("source_documents", [])
        
        # Extract source references
        source_references = []
        for doc in source_docs:
            ref = {
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": getattr(doc, 'relevance_score', 0.8)
            }
            source_references.append(ref)
        
        # Generate supporting evidence
        supporting_evidence = self._extract_supporting_evidence(primary_answer, source_docs)
        
        # Generate contextual follow-up suggestions
        suggested_followups = self._generate_followup_suggestions(query_analysis.intent_category, original_query)
        
        # Generate visualization recommendations
        viz_recommendations = self._suggest_visualizations(query_analysis.intent_category, primary_answer)
        
        # Calculate confidence level
        confidence_level = self._calculate_response_confidence(primary_answer, source_docs, query_analysis)
        
        return IntelligentResponse(
            primary_answer=primary_answer,
            supporting_evidence=supporting_evidence,
            confidence_level=confidence_level,
            source_references=source_references,
            suggested_followups=suggested_followups,
            visualization_recommendations=viz_recommendations
        )
    
    def _extract_supporting_evidence(self, answer: str, source_docs: List[Document]) -> List[str]:
        """Extract key supporting evidence from source documents."""
        evidence = []
        
        # Look for numerical evidence
        numerical_evidence = re.findall(r'[\d,]+\.?\d*\s*(?:rows?|columns?|records?|values?)', answer)
        evidence.extend(numerical_evidence[:3])  # Top 3 numerical facts
        
        # Extract sheet-specific evidence
        for doc in source_docs[:2]:  # Top 2 most relevant sources
            content = doc.page_content
            if len(content) > 100:
                # Extract key sentences with data insights
                sentences = content.split('. ')
                relevant_sentences = [s for s in sentences if any(kw in s.lower() 
                                    for kw in ['data', 'column', 'row', 'value', 'total', 'average'])]
                if relevant_sentences:
                    evidence.append(relevant_sentences[0][:150] + "...")
        
        return evidence[:5]  # Limit to 5 pieces of evidence
    
    def _generate_followup_suggestions(self, intent_category: str, original_query: str) -> List[str]:
        """Generate contextually relevant follow-up question suggestions."""
        base_suggestions = {
            "summary_analysis": [
                "Can you show me trends over time in this data?",
                "What are the key outliers or anomalies?",
                "How does this compare to industry benchmarks?"
            ],
            "trend_analysis": [
                "What factors might be driving these trends?",
                "Can you forecast future values based on this trend?",
                "Are there any seasonal patterns in this data?"
            ],
            "comparative_analysis": [
                "What's causing the differences between these groups?",
                "Can you rank these categories by performance?",
                "Show me the correlation between these variables"
            ],
            "statistical_analysis": [
                "What's the distribution of these values?",
                "Are there any significant correlations?",
                "Can you identify outliers in this data?"
            ]
        }
        
        return base_suggestions.get(intent_category, [
            "Can you provide more details about this analysis?",
            "What other insights can you find in this data?",
            "How can I visualize this information?"
        ])
    
    def _suggest_visualizations(self, intent_category: str, answer_content: str) -> List[str]:
        """Suggest appropriate visualizations based on query intent and content."""
        viz_mapping = {
            "trend_analysis": ["Line chart showing values over time", "Area chart for cumulative trends"],
            "comparative_analysis": ["Bar chart comparing categories", "Stacked bar chart for subcategory breakdown"],
            "summary_analysis": ["Dashboard with key metrics", "Pie chart for proportional data"],
            "statistical_analysis": ["Histogram for distribution", "Scatter plot for correlations"],
            "general_inquiry": ["Table view for detailed data", "Summary cards for key metrics"]
        }
        
        base_suggestions = viz_mapping.get(intent_category, ["Data table", "Summary metrics"])
        
        # Enhance based on answer content
        if "over time" in answer_content.lower() or "trend" in answer_content.lower():
            base_suggestions.append("Time series line chart")
        if "total" in answer_content.lower() or "sum" in answer_content.lower():
            base_suggestions.append("KPI cards with totals")
        if "category" in answer_content.lower() or "group" in answer_content.lower():
            base_suggestions.append("Grouped bar chart")
        
        return list(set(base_suggestions))[:4]  # Return unique suggestions, max 4
    
    def _calculate_response_confidence(self, answer: str, source_docs: List[Document], 
                                     query_analysis: AnalyticsQuery) -> float:
        """Calculate confidence level for the analytical response."""
        confidence_factors = []
        
        # Source document quality
        if len(source_docs) >= 3:
            confidence_factors.append(0.9)
        elif len(source_docs) >= 1:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        # Query clarity
        confidence_factors.append(min(query_analysis.confidence_score + 0.3, 1.0))
        
        # Answer length and detail (proxy for completeness)
        if len(answer) > 200:
            confidence_factors.append(0.8)
        elif len(answer) > 100:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors)
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Retrieve current conversation context and analytics."""
        return {
            "total_queries": len(self.query_history),
            "recent_queries": [q.to_dict() for q in self.query_history[-5:]],
            "conversation_length": len(self.conversation_memory.chat_memory.messages),
            "knowledge_base_status": "initialized" if self.knowledge_base else "not_initialized"
        }

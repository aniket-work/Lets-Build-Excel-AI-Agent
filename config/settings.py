"""
Advanced Configuration Management for Intelligent Excel Analytics Platform
=========================================================================

This configuration module provides a sophisticated approach to managing
application settings, environment variables, and system parameters.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class AIModelConfiguration:
    """Configuration parameters for AI model interactions."""
    
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    chunk_size: int = 1500
    chunk_overlap: int = 300
    embedding_model: str = "text-embedding-3-small"
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.chunk_size < 100:
            raise ValueError("Chunk size must be at least 100 characters")

@dataclass
class DataProcessingConfiguration:
    """Configuration for data processing and analysis operations."""
    
    max_file_size_mb: int = 100
    supported_formats: tuple = ('.xlsx', '.xls')
    max_sheets_per_file: int = 50
    max_rows_per_sheet: int = 100000
    cache_expiry_hours: int = 24
    
    def validate_file_size(self, file_size_bytes: int) -> bool:
        """Validate if file size is within acceptable limits."""
        max_bytes = self.max_file_size_mb * 1024 * 1024
        return file_size_bytes <= max_bytes

@dataclass
class UIConfiguration:
    """User interface configuration and styling parameters."""
    
    page_title: str = "IntelliSheet Analytics"
    page_icon: str = "🧠"
    layout: str = "wide"
    sidebar_state: str = "collapsed"
    theme_variant: str = "dark"
    
    # Advanced styling parameters
    primary_color: str = "#3b82f6"
    secondary_color: str = "#10b981"
    background_gradient: tuple = ("#0f172a", "#1e293b")
    text_colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.text_colors is None:
            self.text_colors = {
                "primary": "#f1f5f9",
                "secondary": "#e2e8f0",
                "muted": "#94a3b8"
            }

class AdvancedApplicationSettings:
    """
    Centralized configuration management system for the Excel AI Analytics platform.
    
    This class implements a sophisticated configuration pattern that I believe
    provides better maintainability and flexibility compared to scattered
    configuration variables throughout the codebase.
    """
    
    def __init__(self):
        self._api_key = self._retrieve_api_key()
        self.ai_config = AIModelConfiguration()
        self.data_config = DataProcessingConfiguration()
        self.ui_config = UIConfiguration()
        self._validate_configuration()
    
    def _retrieve_api_key(self) -> str:
        """
        Securely retrieve OpenAI API key from environment variables.
        
        In my experience, this approach provides better security than
        hardcoding API keys in the source code.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError(
                "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."
            )
        return api_key
    
    def _validate_configuration(self) -> None:
        """Perform comprehensive validation of all configuration parameters."""
        required_dirs = ["src", "config"]
        project_root = Path(__file__).parent.parent
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    @property
    def openai_api_key(self) -> str:
        """Provide secure access to OpenAI API key."""
        return self._api_key
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Retrieve optimized model parameters for AI interactions.
        
        Based on my understanding of language model behavior, these parameters
        provide a good balance between creativity and consistency.
        """
        return {
            "model": self.ai_config.model_name,
            "temperature": self.ai_config.temperature,
            "max_tokens": self.ai_config.max_tokens,
        }
    
    def get_retrieval_parameters(self) -> Dict[str, Any]:
        """Configure parameters for document retrieval and embedding."""
        return {
            "chunk_size": self.ai_config.chunk_size,
            "chunk_overlap": self.ai_config.chunk_overlap,
            "embedding_model": self.ai_config.embedding_model,
        }
    
    def is_development_mode(self) -> bool:
        """Determine if application is running in development mode."""
        return os.getenv("ENVIRONMENT", "production").lower() == "development"

# Global configuration instance
app_settings = AdvancedApplicationSettings()

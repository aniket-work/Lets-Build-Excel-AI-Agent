"""
Excel AI Agent - Advanced Modular Application Entry Point

This is the main entry point for the refactored, professional Excel AI Agent.
It orchestrates all components in a clean, modular architecture.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path for proper imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.ui.app import main as run_ui
    from config.settings import AdvancedApplicationSettings
    from src.utils.file_utils import LoggingUtils
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required dependencies are installed.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


def setup_application_environment():
    """Setup application environment and logging"""
    # Create necessary directories
    directories = [
        "logs",
        "exports", 
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Setup logging
    logger = LoggingUtils.setup_logger("excel_ai_agent")
    logger.info("Excel AI Agent application starting...")
    
    return logger


def validate_environment():
    """Validate application environment and dependencies"""
    logger = LoggingUtils.setup_logger("excel_ai_agent.validation")
    
    # Check for required environment variables
    settings = AdvancedApplicationSettings()
    
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        logger.warning("OpenAI API key not configured in environment")
        return False
    
    logger.info("Environment validation completed successfully")
    return True


def main():
    """Main application entry point"""
    try:
        # Setup environment
        logger = setup_application_environment()
        
        # Validate environment (non-blocking)
        validate_environment()
        
        # Log application startup
        logger.info("Starting Excel AI Agent UI...")
        
        # Run the Streamlit application
        run_ui()
        
    except KeyboardInterrupt:
        logger = LoggingUtils.setup_logger("excel_ai_agent")
        logger.info("Application terminated by user")
    except Exception as e:
        logger = LoggingUtils.setup_logger("excel_ai_agent")
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

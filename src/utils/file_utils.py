"""
Advanced Utility Functions for Excel AI Agent

This module provides sophisticated utility functions for data manipulation,
file operations, validation, and system integration with robust error handling.
"""

import os
import tempfile
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import asdict
import logging


class FileOperationUtils:
    """Advanced file operation utilities with security and validation"""
    
    @staticmethod
    def validate_excel_file(file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """
        Validate Excel file with comprehensive checks
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            file_path = Path(file_path)
            
            # Check file existence
            if not file_path.exists():
                return False, "File does not exist"
            
            # Check file extension
            if file_path.suffix.lower() not in ['.xlsx', '.xls']:
                return False, "Invalid file format. Only .xlsx and .xls files are supported"
            
            # Check file size (max 100MB)
            file_size = file_path.stat().st_size
            if file_size > 100 * 1024 * 1024:
                return False, "File too large. Maximum size is 100MB"
            
            # Try to read file structure
            excel_file = pd.ExcelFile(file_path)
            if not excel_file.sheet_names:
                return False, "No readable sheets found in Excel file"
            
            return True, None
            
        except Exception as e:
            return False, f"File validation error: {str(e)}"
    
    @staticmethod
    def create_secure_temp_file(data: bytes, suffix: str = '.xlsx') -> str:
        """
        Create a secure temporary file with proper cleanup handling
        
        Args:
            data: File data as bytes
            suffix: File extension
            
        Returns:
            Path to temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=suffix,
            prefix='excel_ai_'
        )
        
        try:
            temp_file.write(data)
            temp_file.flush()
            return temp_file.name
        finally:
            temp_file.close()
    
    @staticmethod
    def cleanup_temp_file(file_path: str) -> bool:
        """
        Safely cleanup temporary file
        
        Args:
            file_path: Path to temporary file
            
        Returns:
            True if successfully cleaned up
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                return True
            return False
        except Exception:
            return False
    
    @staticmethod
    def generate_file_hash(file_path: Union[str, Path]) -> str:
        """Generate SHA-256 hash of file for integrity checking"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()


class DataValidationUtils:
    """Advanced data validation and quality assessment utilities"""
    
    @staticmethod
    def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Dictionary with quality metrics and issues
        """
        quality_report = {
            'overall_score': 0.0,
            'completeness': {},
            'consistency': {},
            'validity': {},
            'accuracy': {},
            'issues': []
        }
        
        # Completeness assessment
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells
        
        quality_report['completeness'] = {
            'score': completeness_score,
            'missing_cells': missing_cells,
            'total_cells': total_cells
        }
        
        # Column-level analysis
        for column in df.columns:
            col_data = df[column]
            
            # Check for consistency issues
            if col_data.dtype == 'object':
                # Check for mixed data types in text columns
                numeric_count = sum(pd.to_numeric(col_data, errors='coerce').notna())
                if 0 < numeric_count < len(col_data) * 0.9:
                    quality_report['issues'].append(
                        f"Mixed data types in column '{column}'"
                    )
            
            # Check for validity issues
            if col_data.dtype in ['int64', 'float64']:
                # Check for extreme outliers
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                
                if iqr > 0:
                    outliers = col_data[
                        (col_data < q1 - 3 * iqr) | (col_data > q3 + 3 * iqr)
                    ]
                    
                    if len(outliers) > len(col_data) * 0.05:  # More than 5% outliers
                        quality_report['issues'].append(
                            f"High number of outliers in column '{column}'"
                        )
        
        # Calculate overall quality score
        scores = [completeness_score]
        if len(quality_report['issues']) == 0:
            consistency_score = 1.0
        else:
            consistency_score = max(0, 1 - len(quality_report['issues']) * 0.1)
        
        scores.append(consistency_score)
        quality_report['overall_score'] = sum(scores) / len(scores)
        
        return quality_report
    
    @staticmethod
    def infer_column_semantics(series: pd.Series) -> Dict[str, Any]:
        """
        Infer semantic meaning and characteristics of a data column
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            Dictionary with inferred semantics
        """
        semantics = {
            'name': series.name,
            'dtype': str(series.dtype),
            'semantic_type': 'unknown',
            'patterns': [],
            'suggestions': []
        }
        
        # Basic statistics
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            semantics['semantic_type'] = 'empty'
            return semantics
        
        # Numeric analysis
        if series.dtype in ['int64', 'float64']:
            semantics['semantic_type'] = 'numeric'
            
            # Check for ID patterns
            if series.nunique() == len(non_null_series) and series.min() > 0:
                semantics['patterns'].append('potential_id')
                semantics['suggestions'].append('This column appears to be an identifier')
            
            # Check for percentage patterns
            if series.min() >= 0 and series.max() <= 100:
                semantics['patterns'].append('potential_percentage')
                semantics['suggestions'].append('Values range 0-100, might be percentages')
        
        # Text analysis
        elif series.dtype == 'object':
            semantics['semantic_type'] = 'categorical'
            
            # Check for date patterns
            try:
                pd.to_datetime(non_null_series.head(10))
                semantics['patterns'].append('potential_date')
                semantics['suggestions'].append('Contains date-like strings')
            except:
                pass
            
            # Check cardinality
            unique_ratio = series.nunique() / len(non_null_series)
            
            if unique_ratio < 0.05:
                semantics['patterns'].append('low_cardinality')
                semantics['suggestions'].append('Low cardinality - good for grouping')
            elif unique_ratio > 0.95:
                semantics['patterns'].append('high_cardinality')
                semantics['suggestions'].append('High cardinality - might be identifiers')
        
        return semantics
    
    @staticmethod
    def detect_relationships(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect potential relationships between columns
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of detected relationships
        """
        relationships = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Correlation analysis for numeric columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    correlation = corr_matrix.loc[col1, col2]
                    
                    if abs(correlation) > 0.7:  # Strong correlation
                        relationships.append({
                            'type': 'correlation',
                            'column1': col1,
                            'column2': col2,
                            'strength': abs(correlation),
                            'direction': 'positive' if correlation > 0 else 'negative',
                            'description': f"Strong {('positive' if correlation > 0 else 'negative')} correlation ({correlation:.2f})"
                        })
        
        # Check for hierarchical relationships
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Check if this could be a parent-child relationship
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            
            if 2 <= unique_count <= total_count * 0.1:  # Between 2 and 10% unique values
                relationships.append({
                    'type': 'grouping',
                    'column': col,
                    'groups': unique_count,
                    'description': f"Column '{col}' can group data into {unique_count} categories"
                })
        
        return relationships


class FormattingUtils:
    """Advanced formatting utilities for data presentation"""
    
    @staticmethod
    def format_number(value: Union[int, float], precision: int = 2) -> str:
        """
        Format numbers with appropriate precision and comma separation
        
        Args:
            value: Number to format
            precision: Decimal places for floats
            
        Returns:
            Formatted number string
        """
        if pd.isna(value):
            return "N/A"
        
        if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
            return f"{int(value):,}"
        else:
            return f"{value:,.{precision}f}"
    
    @staticmethod
    def format_percentage(value: Union[int, float], precision: int = 1) -> str:
        """Format value as percentage"""
        if pd.isna(value):
            return "N/A"
        return f"{value:.{precision}f}%"
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
        """Truncate text with ellipsis if too long"""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix


class LoggingUtils:
    """Advanced logging utilities for application monitoring"""
    
    @staticmethod
    def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """
        Setup a comprehensive logger with proper formatting
        
        Args:
            name: Logger name
            level: Logging level
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        
        if not logger.handlers:  # Avoid duplicate handlers
            logger.setLevel(level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    @staticmethod
    def log_operation(logger: logging.Logger, operation: str, **kwargs):
        """
        Log operation with structured data
        
        Args:
            logger: Logger instance
            operation: Operation name
            **kwargs: Additional data to log
        """
        log_data = {
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        logger.info(f"Operation: {operation}", extra=log_data)


class ExportUtils:
    """Utilities for exporting data and results"""
    
    @staticmethod
    def export_dataframe_to_excel(
        df: pd.DataFrame, 
        filename: str,
        sheet_name: str = "Data"
    ) -> bool:
        """
        Export DataFrame to Excel with proper formatting
        
        Args:
            df: DataFrame to export
            filename: Output filename
            sheet_name: Sheet name in Excel file
            
        Returns:
            True if export successful
        """
        try:
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Get workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets[sheet_name]
                
                # Add formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Apply header formatting
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Auto-adjust column widths
                for column in df:
                    column_length = max(df[column].astype(str).map(len).max(), len(column))
                    col_idx = df.columns.get_loc(column)
                    worksheet.set_column(col_idx, col_idx, min(column_length + 2, 50))
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def export_analysis_report(
        analysis_data: Dict[str, Any], 
        filename: str
    ) -> bool:
        """
        Export analysis report as JSON
        
        Args:
            analysis_data: Analysis results to export
            filename: Output filename
            
        Returns:
            True if export successful
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            return True
        except Exception:
            return False


# Create convenience function for commonly used utilities
def validate_and_process_file(file_data: bytes, filename: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Convenience function to validate and create temporary file
    
    Returns:
        Tuple of (is_valid, error_message, temp_file_path)
    """
    # Create temporary file
    temp_path = FileOperationUtils.create_secure_temp_file(file_data)
    
    # Validate file
    is_valid, error_msg = FileOperationUtils.validate_excel_file(temp_path)
    
    if not is_valid:
        FileOperationUtils.cleanup_temp_file(temp_path)
        return False, error_msg, None
    
    return True, None, temp_path

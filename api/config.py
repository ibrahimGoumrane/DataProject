"""
Configuration management for RAG Pipeline
Centralizes all environment variable handling and provides defaults.
"""

import os
from typing import Union
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class RAGConfig:
    """
    Centralized configuration for the RAG pipeline.
    Loads settings from environment variables with sensible defaults.
    """
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # OpenAI Configuration
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        
        # RAG Pipeline Settings
        self.MAX_CONTEXT_LENGTH = self._get_int('RAG_MAX_CONTEXT_LENGTH', 10000)
        self.MAX_TOKENS = self._get_int('RAG_MAX_TOKENS', 10000)
        self.TEMPERATURE = self._get_float('RAG_TEMPERATURE', 0.7)
        self.MODEL_NAME = os.getenv('RAG_MODEL_NAME', 'gpt-4.1-nano')
        self.TOP_K_RESULTS = self._get_int('RAG_TOP_K_RESULTS', 10)
        
        # Data Processing Settings
        self.CHUNK_SIZE = self._get_int('DATA_CHUNK_SIZE', 1000)
        self.CHUNK_OVERLAP = self._get_int('DATA_CHUNK_OVERLAP', 100)
        self.DATA_MAX_CONTEXT_LENGTH = self._get_int('DATA_MAX_CONTEXT_LENGTH', 10000)
        
        # Scraper Settings
        self.SCRAPER_TIMEOUT = self._get_int('SCRAPER_TIMEOUT', 60)
        self.SCRAPER_MAX_DEPTH = self._get_int('SCRAPER_MAX_DEPTH', 4)
        self.SCRAPER_MAX_ANCHOR_LINKS = self._get_int('SCRAPER_MAX_ANCHOR_LINKS', 10)
        self.SCRAPER_SPARSE_CONTENT_THRESHOLD = self._get_int('SCRAPER_SPARSE_CONTENT_THRESHOLD', 750)
        self.SCRAPER_MEAN_SIMILARITY_THRESHOLD = self._get_float('SCRAPER_MEAN_SIMILARITY_THRESHOLD', 0.3)
        # Storage Settings
        self.STORAGE_DIR = os.getenv('STORAGE_DIR', 'storage')

        
    def _get_int(self, key: str, default: int) -> int:
        """Get integer value from environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default
    
    def _get_float(self, key: str, default: float) -> float:
        """Get float value from environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def validate(self) -> dict:
        """
        Validate configuration and return status.
        
        Returns:
            dict: Validation results with warnings and errors
        """
        warnings = []
        errors = []
        
        # Check critical configurations
        if not self.OPENAI_API_KEY:
            warnings.append("OPENAI_API_KEY not set - OpenAI features will be disabled")
        
        if self.MAX_CONTEXT_LENGTH < 100:
            warnings.append(f"MAX_CONTEXT_LENGTH ({self.MAX_CONTEXT_LENGTH}) is very low")
        
        if self.TEMPERATURE < 0 or self.TEMPERATURE > 1:
            errors.append(f"TEMPERATURE ({self.TEMPERATURE}) must be between 0 and 1")
        
        if self.TOP_K_RESULTS < 1:
            errors.append(f"TOP_K_RESULTS ({self.TOP_K_RESULTS}) must be at least 1")
        
        if self.CHUNK_SIZE < 50:
            warnings.append(f"CHUNK_SIZE ({self.CHUNK_SIZE}) is very small")
        
        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors
        }
    
    def get_summary(self) -> dict:
        """
        Get a summary of current configuration.
        
        Returns:
            dict: Configuration summary
        """
        return {
            "openai": {
                "api_key_set": bool(self.OPENAI_API_KEY),
                "model": self.MODEL_NAME,
                "max_tokens": self.MAX_TOKENS,
                "temperature": self.TEMPERATURE
            },
            "rag": {
                "max_context_length": self.MAX_CONTEXT_LENGTH,
                "top_k_results": self.TOP_K_RESULTS
            },
            "data_processing": {
                "chunk_size": self.CHUNK_SIZE,
                "chunk_overlap": self.CHUNK_OVERLAP,
                "max_context_length": self.DATA_MAX_CONTEXT_LENGTH
            },
            "scraper": {
                "max_pages": self.SCRAPER_MAX_PAGES,
                "timeout": self.SCRAPER_TIMEOUT,
                "max_depth": self.SCRAPER_MAX_DEPTH,
                "max_anchor_links": self.MAX_ANCHOR_LINKS,
                "sparse_content_threshold": self.SCRAPER_SPARSE_CONTENT_THRESHOLD,
                "mean_similarity_threshold": self.SCRAPER_MEAN_SIMILARITY_THRESHOLD
            },
            "storage": {
                "directory": self.STORAGE_DIR
            }
        }

# Global configuration instance
config = RAGConfig()

def load_env_file(env_path: str = '.env') -> bool:
    """
    Load environment variables from a .env file.
    
    Args:
        env_path (str): Path to the .env file
        
    Returns:
        bool: True if file was loaded successfully
    """
    try:
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            return True
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")
    return False

def get_config() -> RAGConfig:
    """
    Get the global configuration instance.
    
    Returns:
        RAGConfig: Configuration instance
    """
    return config

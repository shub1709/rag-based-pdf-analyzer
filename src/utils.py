import os
import yaml
import hashlib
from typing import Dict, Any, List
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logging.warning(f"Config file not found at {self.config_path}. Using defaults.")
            return self.get_default_config()
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "llm": {
                "base_url": "http://localhost:11434",
                "model_name": "qwen3",
                "temperature": 0.3,
                "max_tokens": 2000
            },
            "embeddings": {
                "model_name": "all-MiniLM-L6-v2",
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "vector_store": {
                "collection_name": "pdf_documents",
                "persist_directory": "./data/vector_store"
            }
        }

class FileManager:
    """Manages file operations"""
    
    @staticmethod
    def ensure_directory(path: str) -> None:
        """Ensure directory exists"""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Generate hash for a file"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    @staticmethod
    def save_uploaded_file(uploaded_file, save_path: str) -> str:
        """Save uploaded file and return path"""
        FileManager.ensure_directory(os.path.dirname(save_path))
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return save_path

def format_timestamp() -> str:
    """Get formatted timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
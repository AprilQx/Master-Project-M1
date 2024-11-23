import tomli
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Load configuration from TOML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "rb") as f:
        return tomli.load(f)
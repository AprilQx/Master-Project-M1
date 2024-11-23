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
    try:
        with open(config_path, "rb") as f:
            return tomli.load(f)
    except FileNotFoundError:
        print(f"Config file not found at: {config_path}")
        print(f"Current working directory: {Path.cwd()}")
        raise
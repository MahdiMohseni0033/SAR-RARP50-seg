# utils.py
# Utility functions for the training pipeline.

import torch
import platform
import logging
from typing import Optional, Dict, Any
import yaml # For potentially logging the config nicely

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Get logger for this module

def get_device(requested_device: Optional[str] = None) -> str:
    """
    Selects the appropriate compute device based on availability and user request.
    Prioritizes CUDA > MPS > CPU.

    Args:
        requested_device: The device string requested via config (e.g., '0', '0,1', 'cpu', 'mps', None).

    Returns:
        The validated device string to be used by YOLO (e.g., 'cuda:0', 'cpu', 'mps').
        Returns the specific request if valid and available, otherwise auto-detects.
    """
    if requested_device:
        req_lower = str(requested_device).lower()
        if 'cuda' in req_lower or req_lower.isdigit() or ',' in req_lower:
            if not torch.cuda.is_available():
                logger.warning(f"CUDA device '{requested_device}' requested but CUDA not available! Falling back.")
                # Fall through to auto-detection below
            else:
                # Basic check: If it's just digits or commas, assume CUDA indices
                # More complex validation (e.g., checking specific indices) is handled by torch/ultralytics
                logger.info(f"Using requested CUDA device(s): {requested_device}")
                return str(requested_device) # Return original string format
        elif req_lower == 'mps':
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                 # Check platform just to be sure, though torch checks should suffice
                 if platform.system() == "Darwin":
                     logger.info("Using requested MPS device.")
                     return 'mps'
                 else:
                     logger.warning("MPS requested but not on MacOS! Falling back.")
            else:
                logger.warning("MPS requested but not available/built! Falling back.")
            # Fall through to auto-detection below
        elif req_lower == 'cpu':
            logger.info("Using requested CPU device.")
            return 'cpu'
        else:
             logger.warning(f"Unrecognized device request: '{requested_device}'. Attempting auto-detection.")
             # Fall through

    # Auto-detection if no valid device requested or fallback needed
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Auto-detected {device_count} CUDA device(s). Using 'cuda:0'.")
        return 'cuda:0' # Default to the first GPU

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        if platform.system() == "Darwin":
            logger.info("Auto-detected MPS device.")
            return 'mps'

    logger.info("Auto-detected CPU device.")
    return 'cpu'


def log_config(config: Dict[str, Any]):
    """Logs the training configuration dictionary in a readable format."""
    logger.info("----- Loaded Training Configuration -----")
    # Pretty print the dictionary using yaml dump for readability
    try:
        import yaml
        config_str = yaml.dump(config, indent=2, default_flow_style=False, sort_keys=False)
        # Log line by line to avoid truncation issues in some log viewers
        for line in config_str.splitlines():
            logger.info(line)
    except ImportError:
        # Fallback if PyYAML not installed (though it's needed for loading)
        for key, value in config.items():
            logger.info(f"{key:<25}: {value}")
    except Exception as e:
        logger.error(f"Error formatting config for logging: {e}")
        # Fallback to basic print
        for key, value in config.items():
            logger.info(f"{key:<25}: {value}")
    logger.info("-----------------------------------------")
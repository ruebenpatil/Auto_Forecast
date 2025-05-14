from loguru import logger
from pathlib import Path

WORKING_DIR = Path.cwd().resolve()
LOG_PATH = WORKING_DIR.joinpath("logs")

# Global logging configuration
def setup_logger(name=None):
    """
    Set up a logger with consistent formatting
    
    Args:
        name: The name of the logger (typically __name__ from the calling module)
        
    Returns:
        A configured logger instance
    """
    # Use the module name if no name is provided
    logger.remove()
    file_path = LOG_PATH.joinpath(r"app.log")
    logger.add(file_path, rotation="500 MB", retention="10 days", compression="zip")
    return logger
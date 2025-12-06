import sys
from pathlib import Path
from loguru import logger

def setup_logger(script_name: str):
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")
    logger.add(log_dir / f"{script_name}.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="DEBUG", rotation="10 MB", encoding="utf-8")
    
    return logger


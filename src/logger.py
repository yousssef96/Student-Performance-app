import sys
from loguru import logger
from datetime import datetime
import os

LOG_FILE = os.path.join("logs", f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log")

# Remove the default stderr sink
logger.remove()

# Console output (colorized, clean)
logger.add(sys.stderr, level="INFO")

logger.add(
    LOG_FILE,
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} - {message}",
    enqueue=True,           
)

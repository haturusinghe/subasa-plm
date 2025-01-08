import logging
from pathlib import Path

def setup_logging(log_dir: str = "logs"):
    Path(log_dir).mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/training.log")
        ]
    )

    return logging.getLogger(__name__)

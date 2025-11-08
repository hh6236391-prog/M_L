import os
import logging
from dotenv import load_dotenv

load_dotenv()

def setup_logger(name="face_system"):
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logger = logging.getLogger(name)
    logger.info("Logger initialized.")
    return logger

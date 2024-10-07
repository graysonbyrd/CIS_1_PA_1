import logging
import os

log_folder_path = "logs"

if not os.path.exists(log_folder_path):
    os.makedirs(log_folder_path)

logging.basicConfig(
    filename=os.path.join(log_folder_path, "program.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

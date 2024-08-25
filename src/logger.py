import datetime
import os
import logging

# Get the current date and time
now = datetime.datetime.now()

# Format the date and time and replace spaces and colons with underscores
formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")

# Create the directory path
log_dir = os.path.join("Log", formatted_now)

# Create the directory if it does not exist
os.makedirs(log_dir, exist_ok=True)

# Define the log file path
log_file = os.path.join(log_dir, "logfile.log")

# Configure logging
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
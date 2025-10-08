#!/usr/bin/env python3
"""
Centralized logging configuration for LoRA Easy Training system.
Creates log files that are easy to copy/paste and share for debugging.
"""

import logging
import os
from datetime import datetime
import sys

def setup_file_logging():
    """
    Set up file-based logging for the LoRA training system.
    Creates timestamped log files in the logs/ directory.
    """
    
    # Create logs directory
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"lora_training_{timestamp}.log")
    
    # Configure logging format
    log_format = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    
    # Create console handler (still show in Jupyter, but also log to file)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add both handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging initialized - log file: {log_file}")
    logger.info(f"🔍 To view logs: tail -f {log_file}")
    
    return log_file

def get_latest_log_file():
    """Get the path to the most recent log file"""
    log_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(log_dir):
        return None
        
    log_files = [f for f in os.listdir(log_dir) if f.startswith("lora_training_") and f.endswith(".log")]
    if not log_files:
        return None
        
    # Sort by modification time and return the newest
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
    return os.path.join(log_dir, log_files[0])

def tail_log_file(lines=50):
    """Print the last N lines of the current log file"""
    log_file = get_latest_log_file()
    if not log_file:
        print("No log files found")
        return
        
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            tail_lines = all_lines[-lines:]
            print(f"📝 Last {len(tail_lines)} lines from {os.path.basename(log_file)}:")
            print("=" * 80)
            for line in tail_lines:
                print(line.rstrip())
            print("=" * 80)
    except Exception as e:
        print(f"Error reading log file: {e}")

if __name__ == "__main__":
    # Test the logging setup
    log_file = setup_file_logging()
    logger = logging.getLogger("test")
    logger.info("Test log message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    print(f"\nLog file created: {log_file}")
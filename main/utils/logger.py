# -*- coding: utf-8 -*-
"""
Centralized Logger Setup for PSD Optimization System

This module provides a centralized logging system with color-coded console output
that supports different log levels with appropriate colors and formatting.

Why (Purpose and Necessity):
The system needs a unified logging approach to replace all print() statements
with proper logging that includes timestamps, module names, and color coding
for better debugging and monitoring of the optimization process.

What (Implementation Details):
- Uses Python's logging module with custom colored formatters
- Supports ERROR (red), WARNING (orange), INFO (white), DEBUG (light green)
- Provides timestamps and module names in the output format
- Uses colorama for cross-platform colored terminal support
- Centralized configuration to ensure consistent logging across all modules
"""

import logging
import sys
from datetime import datetime
from typing import Optional

try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback colors for systems without colorama
    class Fore:
        RED = ''
        YELLOW = ''
        WHITE = ''
        GREEN = ''
        RESET = ''
    class Style:
        RESET_ALL = ''


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log messages based on their level.
    
    Why (Purpose and Necessity):
    Provides visual distinction between different log levels, making it easier
    to quickly identify errors, warnings, and debug information in the console output.
    
    What (Implementation Details):
    - Maps log levels to specific colors using colorama
    - Formats messages with timestamp, module name, level, and colored message
    - Handles cases where colorama is not available with fallback formatting
    """
    
    # Color mapping for different log levels
    COLORS = {
        'DEBUG': Fore.GREEN,
        'INFO': Fore.WHITE,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize the colored formatter.
        
        Args:
            use_colors (bool): Whether to use colors in the output. Defaults to True.
        """
        self.use_colors = use_colors and COLORAMA_AVAILABLE
        super().__init__()
    
    def format(self, record):
        """
        Format the log record with appropriate colors and structure.
        
        Args:
            record (logging.LogRecord): The log record to format.
            
        Returns:
            str: The formatted log message with colors and timestamp.
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get module name (remove package prefixes for cleaner output)
        module_name = record.name.split('.')[-1]
        
        # Get log level
        level_name = record.levelname
        
        # Apply color if enabled
        if self.use_colors and level_name in self.COLORS:
            colored_level = f"{self.COLORS[level_name]}{level_name}{Style.RESET_ALL}"
            colored_message = f"{self.COLORS[level_name]}{record.getMessage()}{Style.RESET_ALL}"
        else:
            colored_level = level_name
            colored_message = record.getMessage()
        
        # Format the complete message
        formatted_message = f"[{timestamp}] [{module_name}] [{colored_level}] {colored_message}"
        
        return formatted_message


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance for the specified module.
    
    Why (Purpose and Necessity):
    Provides a centralized way to create loggers with consistent configuration
    across all modules in the system, ensuring uniform formatting and behavior.
    
    What (Implementation Details):
    - Creates a logger with the specified name
    - Configures console handler with colored formatter
    - Sets appropriate log level
    - Prevents duplicate handlers if logger already exists
    - Returns the configured logger instance
    
    Args:
        name (str): The name of the logger (typically __name__ from the calling module).
        level (int, optional): The logging level. Defaults to logging.INFO.
        
    Returns:
        logging.Logger: A configured logger instance ready for use.
    """
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers to the same logger
    if logger.handlers:
        return logger
    
    # Set the log level
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create and set the colored formatter
    formatter = ColoredFormatter(use_colors=True)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to parent loggers to avoid duplicate messages
    logger.propagate = False
    
    return logger


def configure_root_logger(level: int = logging.INFO) -> None:
    """
    Configure the root logger for the entire application.
    
    Why (Purpose and Necessity):
    Ensures that the root logger is properly configured to handle any
    unhandled logging messages and provides a fallback configuration.
    
    What (Implementation Details):
    - Sets the root logger level
    - Creates a console handler with colored formatting
    - Removes any existing handlers to prevent duplicates
    - Configures the root logger for consistent behavior
    
    Args:
        level (int, optional): The logging level for the root logger. Defaults to logging.INFO.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create and set the colored formatter
    formatter = ColoredFormatter(use_colors=True)
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)


# Configure the root logger when this module is imported
configure_root_logger()

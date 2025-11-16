# utils/logger.py

## Role in the System

`utils/logger.py` provides a centralized logging system with color-coded console output. It replaces all print() statements with proper logging that includes timestamps, module names, and color coding for better debugging and monitoring.

## Responsibilities

- Provides consistent logging interface across all modules
- Formats log messages with timestamps and module names
- Adds color coding to log levels for visual distinction
- Configures root logger for unhandled messages

## Dependencies

**Imports:**
- `logging` - Python's standard logging module
- `sys` - Standard output stream
- `datetime` - Timestamp generation
- `typing.Optional` - Type hints
- `colorama` (optional) - Cross-platform colored terminal support

**Used In:**
- All modules in the project import and use `get_logger(__name__)`

## Classes

### Class: `ColoredFormatter`

**Location:** `utils/logger.py`

**Purpose:**  
Custom formatter that adds colors to log messages based on their level. Provides visual distinction between different log levels, making it easier to quickly identify errors, warnings, and debug information.

**Class Variables:**
- `COLORS (dict)` - Maps log levels to colorama colors:
  - 'DEBUG': Fore.GREEN
  - 'INFO': Fore.WHITE
  - 'WARNING': Fore.YELLOW
  - 'ERROR': Fore.RED
  - 'CRITICAL': Fore.RED + Style.BRIGHT

**Methods:**

#### Method: `__init__(self, use_colors=True)`

**Purpose:**  
Initializes the colored formatter.

**Parameters:**
- `use_colors (bool, optional)` - Whether to use colors (default: True)

**Returns:**
None

**Side Effects:**
None

**Error Handling:**
- Falls back to no colors if colorama not available

#### Method: `format(self, record)`

**Purpose:**  
Formats the log record with appropriate colors and structure.

**Parameters:**
- `record (logging.LogRecord)` - The log record to format

**Returns:**
- `str` - Formatted log message with colors and timestamp

**Side Effects:**
None

**Error Handling:**
- Handles cases where colorama is not available (uses plain text)
- Handles missing log level in COLORS dict (uses plain text)

## Functions

### Function: `get_logger(name, level=logging.INFO)`

**Location:** `utils/logger.py`

**Purpose:**  
Get a configured logger instance for the specified module. Provides centralized way to create loggers with consistent configuration across all modules.

**Parameters:**
- `name (str)` - Name of the logger (typically `__name__` from calling module)
- `level (int, optional)` - Logging level (default: logging.INFO)

**Returns:**
- `logging.Logger` - Configured logger instance ready for use

**Side Effects:**
- Creates logger with specified name
- Configures console handler with colored formatter
- Sets log level
- Prevents duplicate handlers if logger already exists
- Sets `propagate = False` to prevent duplicate messages

**Error Handling:**
- Returns existing logger if handlers already exist (prevents duplicates)

**Used In:**
- All modules call this at module level: `logger = get_logger(__name__)`

### Function: `configure_root_logger(level=logging.INFO)`

**Location:** `utils/logger.py`

**Purpose:**  
Configure the root logger for the entire application. Ensures root logger is properly configured to handle any unhandled logging messages.

**Parameters:**
- `level (int, optional)` - Logging level for root logger (default: logging.INFO)

**Returns:**
None

**Side Effects:**
- Sets root logger level
- Creates console handler with colored formatting
- Removes any existing handlers to prevent duplicates
- Configures root logger for consistent behavior

**Error Handling:**
None

**Used In:**
- Called at module import time (when logger.py is imported)

## Usage Pattern

All modules follow this pattern:

```python
from utils.logger import get_logger

logger = get_logger(__name__)

# Then use throughout module:
logger.info("Message")
logger.warning("Warning message")
logger.error("Error message")
logger.debug("Debug message")
```

## Color Coding

- **DEBUG**: Green - Detailed diagnostic information
- **INFO**: White - General informational messages
- **WARNING**: Yellow - Warning messages for potential issues
- **ERROR**: Red - Error messages for problems
- **CRITICAL**: Bright Red - Critical errors

## Format

Log messages are formatted as:
```
[YYYY-MM-DD HH:MM:SS] [module_name] [LEVEL] message
```

Where LEVEL is color-coded based on the log level.


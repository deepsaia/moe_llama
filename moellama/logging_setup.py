"""
Logging configuration setup using loguru.

This module provides a centralized logging configuration that can be loaded
from a JSON configuration file. It sets up loguru with color-coded output,
file rotation, and optional interception of standard logging.
"""

import json
import sys
import logging
from pathlib import Path
from typing import Optional

from loguru import logger


# Store the default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "logging.json"


def setup_logging(
    config_path: Optional[Path] = None,
    enable_file_logging: bool = True,
    log_level: str = "INFO"
):
    """
    Setup loguru logging from configuration file.

    Args:
        config_path: Path to logging.json config file (default: config/logging.json)
        enable_file_logging: Whether to enable file logging (default: True)
        log_level: Minimum log level (default: INFO)
    """
    # Remove default handler
    logger.remove()

    # Load configuration if provided
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config = None
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load logging config from {config_path}: {e}")

    # Setup handlers from config or defaults
    if config and "handlers" in config:
        for handler_config in config["handlers"]:
            sink = handler_config["sink"]

            # Handle special sink values
            if sink == "sys.stderr":
                sink = sys.stderr
            elif sink == "sys.stdout":
                sink = sys.stdout
            elif not enable_file_logging and not sink.startswith("sys."):
                # Skip file handlers if file logging is disabled
                continue

            # Prepare handler kwargs
            handler_kwargs = {
                "sink": sink,
                "level": handler_config.get("level", log_level),
                "format": handler_config.get("format"),
                "colorize": handler_config.get("colorize", False),
                "backtrace": handler_config.get("backtrace", True),
                "diagnose": handler_config.get("diagnose", True),
            }

            # Add file-specific options
            if isinstance(sink, str) and not sink.startswith("sys."):
                # Create logs directory if needed
                log_file = Path(sink)
                log_file.parent.mkdir(parents=True, exist_ok=True)

                handler_kwargs.update({
                    "rotation": handler_config.get("rotation"),
                    "retention": handler_config.get("retention"),
                    "compression": handler_config.get("compression"),
                })

            # Add handler
            logger.add(**handler_kwargs)
    else:
        # Default configuration if no config file
        # Console handler with colors
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        # File handler (if enabled)
        if enable_file_logging:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)

            logger.add(
                "logs/moellama_{time:YYYY-MM-DD}.log",
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="00:00",
                retention="7 days",
                compression="zip",
                backtrace=True,
                diagnose=True,
            )

    # Intercept standard logging if configured
    if config and config.get("extra", {}).get("intercept_standard_logging", False):
        intercept_standard_logging()

    logger.info("Logging initialized")


def intercept_standard_logging():
    """
    Intercept standard library logging and redirect to loguru.

    This is useful when using libraries that use the standard logging module.
    """
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def get_logger(name: str):
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance bound with the name
    """
    return logger.bind(name=name)


# Convenience functions for common log levels
def trace(message: str, **kwargs):
    """Log a TRACE level message."""
    logger.trace(message, **kwargs)


def debug(message: str, **kwargs):
    """Log a DEBUG level message."""
    logger.debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log an INFO level message."""
    logger.info(message, **kwargs)


def success(message: str, **kwargs):
    """Log a SUCCESS level message."""
    logger.success(message, **kwargs)


def warning(message: str, **kwargs):
    """Log a WARNING level message."""
    logger.warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log an ERROR level message."""
    logger.error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log a CRITICAL level message."""
    logger.critical(message, **kwargs)


def exception(message: str, **kwargs):
    """Log an exception with traceback."""
    logger.exception(message, **kwargs)

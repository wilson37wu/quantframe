"""Log management utilities for quantframe.

This module provides tools for log file management including:
- Log rotation
- Compression of old logs
- Automatic cleanup of expired logs
"""
import gzip
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

class LogManager:
    """Manages log files with rotation and cleanup.
    
    Attributes:
        log_dir: Directory containing log files
        max_size_mb: Maximum size of log file before rotation
        max_age_days: Maximum age of logs before cleanup
        max_backups: Maximum number of backup files to keep
    """
    
    def __init__(self, log_dir: str, max_size_mb: int = 100,
                 max_age_days: int = 30, max_backups: int = 5):
        """Initialize log manager.
        
        Args:
            log_dir: Directory for log files
            max_size_mb: Maximum size of log file before rotation
            max_age_days: Maximum age of logs before cleanup
            max_backups: Maximum number of backup files to keep
        """
        self.log_dir = Path(log_dir)
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self.max_age = timedelta(days=max_age_days)
        self.max_backups = max_backups
        self.logger = logging.getLogger(__name__)
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def should_rotate(self, log_file: Path) -> bool:
        """Check if log file should be rotated.
        
        Args:
            log_file: Path to log file
            
        Returns:
            True if file should be rotated
        """
        return log_file.stat().st_size > self.max_size

    def rotate_log(self, log_file: Path) -> None:
        """Rotate a log file.
        
        Args:
            log_file: Path to log file to rotate
        """
        try:
            # Rotate existing backups
            for i in range(self.max_backups - 1, 0, -1):
                old = log_file.with_suffix(f'.{i}.gz')
                new = log_file.with_suffix(f'.{i + 1}.gz')
                if old.exists():
                    shutil.move(str(old), str(new))
            
            # Compress current log to .1.gz
            if log_file.exists():
                with log_file.open('rb') as f_in:
                    with gzip.open(str(log_file.with_suffix('.1.gz')), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Clear current log file
                log_file.write_text('')
                
        except Exception as e:
            self.logger.error(f"Failed to rotate log {log_file}: {e}")

    def cleanup_old_logs(self) -> None:
        """Remove log files older than max_age."""
        try:
            current_time = datetime.now()
            for log_file in self.log_dir.glob('*.log*'):
                if current_time - datetime.fromtimestamp(log_file.stat().st_mtime) > self.max_age:
                    log_file.unlink()
                    self.logger.info(f"Removed old log file: {log_file}")
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")

    def check_and_rotate(self, log_file: Optional[str] = None) -> None:
        """Check if logs need rotation and rotate if necessary.
        
        Args:
            log_file: Optional specific log file to check
        """
        try:
            if log_file:
                log_path = Path(log_file)
                if self.should_rotate(log_path):
                    self.rotate_log(log_path)
            else:
                for log_path in self.log_dir.glob('*.log'):
                    if self.should_rotate(log_path):
                        self.rotate_log(log_path)
            
            # Cleanup old logs
            self.cleanup_old_logs()
            
        except Exception as e:
            self.logger.error(f"Failed to check and rotate logs: {e}")

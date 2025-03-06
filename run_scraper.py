"""Continuous data scraping script for CoinGlass liquidation data.

This script provides a continuous scraping service for cryptocurrency liquidation data
from CoinGlass. It includes features like:
- Configurable scraping intervals
- Excel file export with append functionality
- Windows sleep prevention
- Comprehensive logging
- Graceful error handling
"""

from quantframe.scrapers.coinglass import CoinglassScraper
import os
import time
import logging
import ctypes
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def prevent_sleep() -> bool:
    """Prevent Windows from going to sleep while the script is running.
    
    Uses Windows API SetThreadExecutionState to keep the system active.
    This ensures continuous data collection even during normally idle periods.
    
    Returns:
        bool: True if sleep prevention was successfully enabled, False otherwise
    
    Note:
        This only affects system sleep, not display sleep settings.
    """
    try:
        # Define Windows constants
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        
        # Call SetThreadExecutionState
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        logger.info("Sleep prevention enabled")
        return True
    except Exception as e:
        logger.error(f"Failed to prevent sleep mode: {str(e)}")
        return False

def restore_sleep() -> None:
    """Restore default Windows sleep behavior.
    
    Resets the system's thread execution state to allow normal sleep behavior.
    Should be called when scraping is complete or interrupted.
    """
    try:
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        logger.info("Sleep prevention disabled")
    except Exception as e:
        logger.error(f"Failed to restore sleep settings: {str(e)}")

def select_excel_file(exports_dir: str) -> str:
    """Allow user to select an existing Excel file or create new one.
    
    Presents a list of existing Excel files in the exports directory and allows
    the user to either select one for appending data or create a new file.
    
    Args:
        exports_dir (str): Directory path where Excel files are stored
        
    Returns:
        str: Full path to selected Excel file, or empty string for new file
        
    Example:
        >>> file_path = select_excel_file("./exports")
        >>> print("Selected:", file_path or "New file")
    """
    # List all Excel files in exports directory
    excel_files = [f for f in os.listdir(exports_dir) if f.endswith('.xlsx')]
    
    if not excel_files:
        logger.info("No existing Excel files found. Will create new file.")
        return ""
    
    print("\nExisting Excel files:")
    for i, file in enumerate(excel_files, 1):
        print(f"{i}. {file}")
    print("0. Create new file")
    
    try:
        choice = input("\nSelect file number (or press Enter for new file): ").strip()
        if not choice:
            return ""
        
        choice = int(choice)
        if choice == 0:
            return ""
        elif 1 <= choice <= len(excel_files):
            return os.path.join(exports_dir, excel_files[choice - 1])
        else:
            logger.warning("Invalid choice. Creating new file.")
            return ""
    except ValueError:
        logger.warning("Invalid input. Creating new file.")
        return ""

def get_interval_input() -> int:
    """Get scraping interval from user input.
    
    Prompts the user for a scraping interval in seconds, with input validation
    to ensure a reasonable value.
    
    Returns:
        int: Scraping interval in seconds (minimum 1, default 300)
        
    Example:
        >>> interval = get_interval_input()
        >>> print(f"Will scrape every {interval} seconds")
    """
    while True:
        try:
            interval = input("\nEnter scraping interval in seconds (minimum 1, default 300): ").strip()
            if not interval:
                return 300
            
            interval = int(interval)
            if interval < 1:
                print("Interval must be at least 1 second.")
                continue
            
            return interval
        except ValueError:
            print("Please enter a valid number.")

def main() -> None:
    """Main execution function for the CoinGlass scraper.
    
    Sets up the scraping environment, handles user input for configuration,
    and runs the continuous scraping loop. Features include:
    - Directory creation for exports
    - Excel file selection/creation
    - Interval configuration
    - Sleep prevention
    - Continuous data collection with error handling
    - Graceful shutdown
    """
    # Create exports directory if it doesn't exist
    exports_dir = os.path.join(os.path.dirname(__file__), 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    # Get the Excel file to update
    excel_file = select_excel_file(exports_dir)
    
    # Get scraping interval
    interval = get_interval_input()
    logger.info(f"Scraping interval set to {interval} seconds")
    
    # Enable sleep prevention
    if not prevent_sleep():
        logger.error("Failed to set up sleep prevention. Script may not run during sleep mode.")
    
    try:
        logger.info("Starting continuous scraping...")
        while True:
            current_time = datetime.now()
            logger.info(f"Starting scrape at {current_time}")
            
            # Get data with a new browser session each time
            data = CoinglassScraper.get_liquidation_data()
            
            if data:
                # Export to Excel, appending to existing file if specified
                excel_file = CoinglassScraper.export_to_excel(data, exports_dir, excel_file)
                if excel_file:
                    logger.info(f"Data appended to: {excel_file}")
                else:
                    logger.error("Failed to export data to Excel")
            else:
                logger.error("Failed to retrieve data")
            
            # Wait for specified interval before next scrape
            logger.info(f"Waiting {interval} seconds before next scrape...")
            time.sleep(interval)
    
    except KeyboardInterrupt:
        logger.info("Scraping stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Restore sleep settings
        restore_sleep()
        logger.info("Scraping terminated")

if __name__ == '__main__':
    main()

from quantframe.scrapers.coinglass_main import CoinglassMainScraper
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
        logging.FileHandler('main_scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def prevent_sleep():
    """Prevent Windows from going to sleep while the script is running."""
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

def restore_sleep():
    """Restore default Windows sleep behavior."""
    try:
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        logger.info("Sleep prevention disabled")
    except Exception as e:
        logger.error(f"Failed to restore sleep settings: {str(e)}")

def get_interval_input() -> int:
    """Get scraping interval from user input."""
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

def get_mode_input() -> bool:
    """Get scraping mode from user input."""
    while True:
        try:
            mode = input("\nRun in headless mode? (y/n, default: y): ").strip().lower()
            if not mode or mode == 'y':
                return True
            elif mode == 'n':
                return False
            else:
                print("Please enter 'y' or 'n'")
        except ValueError:
            print("Invalid input. Please enter 'y' or 'n'")

def main():
    # Create exports directory if it doesn't exist
    exports_dir = os.path.join(os.path.dirname(__file__), 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    # Get scraping mode
    headless = get_mode_input()
    logger.info(f"Running in {'headless' if headless else 'visible'} mode")
    
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
            data = CoinglassMainScraper.get_main_page_data(headless=headless)
            
            if data:
                # Export to Excel
                excel_file = CoinglassMainScraper.export_to_excel(data, exports_dir)
                if excel_file:
                    logger.info(f"Data exported to: {excel_file}")
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

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_page():
    logger.info("Starting page check...")
    
    # Set up Chrome options
    logger.info("Setting up Chrome options...")
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # Comment out headless mode
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)

    # Initialize Chrome driver
    logger.info("Initializing Chrome driver...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Navigate to the page
        url = "https://www.coinglass.com/LiquidationData"
        logger.info(f"Navigating to {url}")
        driver.get(url)

        # Wait for page to load
        logger.info("Waiting for page to load...")
        time.sleep(10)

        # Print page title
        logger.info(f"Page title: {driver.title}")

        # Print page source
        logger.info("Checking page source...")
        source = driver.page_source
        logger.info(f"Page source snippet: {source[:500]}")

        # Look for tables
        logger.info("Looking for tables...")
        tables = driver.find_elements(By.TAG_NAME, "table")
        logger.info(f"Found {len(tables)} tables")

        # Look for divs with class containing 'table'
        logger.info("Looking for table-related divs...")
        table_divs = driver.find_elements(By.CSS_SELECTOR, "[class*='table']")
        logger.info(f"Found {len(table_divs)} divs with 'table' in class name")
        for div in table_divs:
            logger.info(f"Class: {div.get_attribute('class')}")

        # Try to find any visible elements
        logger.info("Looking for any visible elements...")
        elements = driver.find_elements(By.CSS_SELECTOR, "*")
        logger.info(f"Found {len(elements)} total elements")
        
        # Check for specific elements that might be visible
        visible_elements = [e for e in elements[:20] if e.is_displayed()]  # Check first 20 elements
        for elem in visible_elements:
            logger.info(f"Visible element: {elem.tag_name} - Class: {elem.get_attribute('class')} - Text: {elem.text[:100]}")

        logger.info("Press Enter to close the browser...")
        input()

    except Exception as e:
        logger.error(f"Error during page check: {str(e)}")
    finally:
        logger.info("Closing browser...")
        driver.quit()

if __name__ == "__main__":
    check_page()

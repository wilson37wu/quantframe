from typing import List, Dict, Union
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time
import pandas as pd
from datetime import datetime
import os

# Configure logger
logger = logging.getLogger(__name__)

class CoinglassMainScraper:
    """Scraper for Coinglass main page data."""
    
    BASE_URL = "https://www.coinglass.com/"
    
    @staticmethod
    def setup_chrome_options(headless: bool = False) -> webdriver.ChromeOptions:
        """
        Set up Chrome options for the scraper.
        
        Args:
            headless: If True, run Chrome in headless mode
            
        Returns:
            ChromeOptions: Configured Chrome options
        """
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        
        if headless:
            options.add_argument('--headless=new')  # new headless mode for Chrome
            options.add_argument('--disable-gpu')
        else:
            # Start window minimized if not in headless mode
            options.add_argument('--start-minimized')
        
        return options
    
    @staticmethod
    def select_customizable_options(driver: webdriver.Chrome) -> None:
        """Select all customizable options in the table."""
        try:
            # Wait for and click the customizable button
            wait = WebDriverWait(driver, 20)
            customize_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Customizable')]")))
            customize_btn.click()
            time.sleep(2)  # Allow menu to fully open
            
            # Select all available options
            options = driver.find_elements(By.CSS_SELECTOR, '.ant-checkbox-wrapper')
            for option in options:
                if not option.find_element(By.CSS_SELECTOR, 'input').is_selected():
                    option.click()
                    time.sleep(0.5)  # Brief pause between selections
            
            # Apply selections
            apply_btn = driver.find_element(By.XPATH, "//button[contains(., 'Apply')]")
            apply_btn.click()
            time.sleep(2)  # Allow table to update
            
        except Exception as e:
            logger.error(f"Error selecting customizable options: {str(e)}")

    @staticmethod
    def extract_numeric_value(cell: webdriver.remote.webelement.WebElement) -> Union[float, str]:
        """
        Extract numeric value from a cell, attempting to get the full value from hover data.
        
        Args:
            cell: WebElement representing a table cell
            
        Returns:
            Union[float, str]: Extracted numeric value or original text if not numeric
        """
        try:
            # First try to get the hover value from title attribute
            title_value = cell.get_attribute('title')
            if title_value and title_value.replace('$', '').replace(',', '').replace('.', '', 1).replace('-', '', 1).replace(' ', '').isdigit():
                return float(title_value.replace('$', '').replace(',', ''))
            
            # Try to get value from data-value attribute
            data_value = cell.get_attribute('data-value')
            if data_value and data_value.replace('.', '', 1).replace('-', '', 1).isdigit():
                return float(data_value)
            
            # Try to get the hover text using JavaScript
            hover_text = cell.get_attribute('data-original-title')
            if hover_text and hover_text.replace('$', '').replace(',', '').replace('.', '', 1).replace('-', '', 1).replace(' ', '').isdigit():
                return float(hover_text.replace('$', '').replace(',', ''))
            
            # If no hover data found, process the visible text
            content = cell.text.strip()
            
            # Handle K, M, B suffixes
            if content and content[-1] in ['K', 'M', 'B']:
                number = float(content[:-1].replace('$', '').replace(',', ''))
                multiplier = {
                    'K': 1000,
                    'M': 1000000,
                    'B': 1000000000
                }[content[-1]]
                return number * multiplier
            
            # Handle regular numbers with $ or %
            if content and ('$' in content or '%' in content):
                content = content.replace('$', '').replace('%', '').replace(',', '')
                if content.replace('.', '', 1).replace('-', '', 1).isdigit():
                    return float(content)
            
            return content
        except Exception as e:
            logger.debug(f"Error extracting numeric value: {str(e)}")
            return cell.text.strip()

    @staticmethod
    def get_main_page_data(headless: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fetch and parse data from Coinglass main page.
        
        Args:
            headless: If True, run Chrome in headless mode (default: False)
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of table names and their corresponding DataFrames
        """
        driver = None
        try:
            # Set up Chrome options
            logger.info("Setting up Chrome options...")
            options = CoinglassMainScraper.setup_chrome_options(headless)
            
            # Initialize Chrome driver
            logger.info("Initializing Chrome driver...")
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            # Fetch the page
            logger.info(f"Fetching page: {CoinglassMainScraper.BASE_URL}")
            driver.get(CoinglassMainScraper.BASE_URL)
            
            # Wait for page to load
            time.sleep(5)
            
            # Select customizable options
            logger.info("Selecting customizable options...")
            CoinglassMainScraper.select_customizable_options(driver)
            
            # Initialize results dictionary
            results = {}
            
            # Find and extract all tables
            tables = driver.find_elements(By.CSS_SELECTOR, '.ant-table')
            logger.info(f"Found {len(tables)} tables")
            
            for idx, table in enumerate(tables):
                try:
                    # Get table title if available
                    title_elem = table.find_element(By.XPATH, './preceding-sibling::div[contains(@class, "title")]')
                    table_name = title_elem.text.strip()
                except NoSuchElementException:
                    table_name = f"Table_{idx + 1}"
                
                logger.info(f"Processing table: {table_name}")
                
                # Extract headers
                headers = []
                header_cells = table.find_elements(By.CSS_SELECTOR, '.ant-table-thead th')
                for cell in header_cells:
                    text = cell.text.strip()
                    if text:
                        headers.append(text)
                
                if not headers:
                    logger.warning(f"No headers found in table: {table_name}")
                    continue
                
                # Extract rows
                rows_data = []
                rows = table.find_elements(By.CSS_SELECTOR, '.ant-table-tbody tr:not(.ant-table-measure-row)')
                
                for row in rows:
                    cells = row.find_elements(By.CSS_SELECTOR, 'td')
                    if len(cells) != len(headers):
                        continue
                    
                    row_data = {}
                    for header, cell in zip(headers, cells):
                        # Use the new extract_numeric_value method
                        value = CoinglassMainScraper.extract_numeric_value(cell)
                        row_data[header] = value
                    
                    rows_data.append(row_data)
                
                # Create DataFrame for this table
                if rows_data:
                    results[table_name] = pd.DataFrame(rows_data)
                    logger.info(f"Successfully processed table: {table_name} with {len(rows_data)} rows")
            
            return results
            
        except Exception as e:
            logger.error(f"Error scraping Coinglass main page: {str(e)}")
            return {}
        finally:
            if driver:
                driver.quit()
                logger.info("Browser session closed")

    @staticmethod
    def export_to_excel(data: Dict[str, pd.DataFrame], output_dir: str = None) -> str:
        """
        Export all tables to a single Excel file with multiple sheets.
        
        Args:
            data: Dictionary of table names and their corresponding DataFrames
            output_dir: Optional directory to save the Excel file (default: current directory)
            
        Returns:
            str: Path to the Excel file
        """
        if not data:
            logger.warning("No data to export")
            return ""
            
        try:
            # Create output directory if it doesn't exist
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = os.getcwd()
            
            # Generate filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"coinglass_main_data_{timestamp_str}.xlsx"
            filepath = os.path.join(output_dir, filename)
            
            # Export to Excel with multiple sheets
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                for table_name, df in data.items():
                    # Clean sheet name (Excel has 31 character limit for sheet names)
                    sheet_name = table_name[:31].replace('/', '_').replace('\\', '_')
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"Data exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            return ""


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('coinglass_main_scraper.log'),
            logging.StreamHandler()
        ]
    )
    
    # Test the scraper
    scraper = CoinglassMainScraper()
    data = scraper.get_main_page_data(headless=True)
    if data:
        scraper.export_to_excel(data, 'exports')

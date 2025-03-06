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

class CoinglassScraper:
    """Scraper for Coinglass liquidation data."""
    
    BASE_URL = "https://www.coinglass.com/LiquidationData"
    
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
    def select_market_categories(driver: webdriver.Chrome) -> None:
        """Select all market categories (derivative, spot, meme, SOL, etc.)."""
        try:
            # Wait for and click the market category selector
            wait = WebDriverWait(driver, 20)
            category_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'market-category')]")))
            category_btn.click()
            time.sleep(2)  # Allow menu to fully open
            
            # Select all categories
            categories = driver.find_elements(By.CSS_SELECTOR, '.ant-select-item')
            for category in categories:
                category.click()
                time.sleep(0.5)  # Brief pause between selections
            
            # Click outside to close the menu
            driver.find_element(By.TAG_NAME, 'body').click()
            time.sleep(2)  # Allow table to update
            
        except Exception as e:
            logger.error(f"Error selecting market categories: {str(e)}")

    @staticmethod
    def get_liquidation_data() -> List[Dict[str, Union[str, int, float]]]:
        """
        Fetch and parse liquidation data from Coinglass.
        
        Returns:
            List[Dict]: List of dictionaries containing liquidation data for each cryptocurrency
        """
        driver = None
        try:
            # Set up Chrome options
            logger.info("Setting up Chrome options...")
            options = webdriver.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            options.add_experimental_option('excludeSwitches', ['enable-automation'])
            options.add_experimental_option('useAutomationExtension', False)
            
            # Initialize Chrome driver
            logger.info("Initializing Chrome driver...")
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            # Fetch the page
            logger.info(f"Fetching page: {CoinglassScraper.BASE_URL}")
            driver.get(CoinglassScraper.BASE_URL)
            
            # Wait for table to be fully loaded
            try:
                wait = WebDriverWait(driver, 20)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.ant-table-tbody tr')))
                wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, '.ant-table-tbody tr')) > 1)
            except TimeoutException:
                logger.error("Timeout waiting for table data to load")
                return []
            
            # Extract headers from the fixed header
            header_cells = driver.find_elements(By.CSS_SELECTOR, '.ant-table-thead th')
            headers = []
            for cell in header_cells:
                text = cell.text.strip()
                if text:  # Only add non-empty headers
                    headers.append(text)
            logger.info(f"Found headers: {headers}")
            
            if not headers:
                logger.error("No headers found in table")
                return []
            
            # Extract rows from the table body
            results = []
            rows = driver.find_elements(By.CSS_SELECTOR, '.ant-table-tbody tr:not(.ant-table-measure-row)')
            logger.info(f"Found {len(rows)} rows")
            
            for row_idx, row in enumerate(rows):
                cells = row.find_elements(By.CSS_SELECTOR, 'td')
                logger.info(f"Row {row_idx + 1} has {len(cells)} cells")
                
                if len(cells) != len(headers):
                    logger.warning(f"Row {row_idx + 1} has {len(cells)} cells but {len(headers)} headers")
                    continue
                
                row_data = {}
                for idx, cell in enumerate(cells):
                    content = cell.text.strip()
                    logger.debug(f"Row {row_idx + 1}, Cell {idx} content: {content}")
                    
                    # Handle numerical values
                    if headers[idx] in ['24h Liquidation', '4h Max Liquidation']:
                        content = content.replace('$', '').replace(',', '')
                        if content.isdigit():
                            content = int(content)
                        elif content.replace('.', '', 1).isdigit():
                            content = float(content)
                    
                    row_data[headers[idx]] = content
                
                results.append(row_data)
                logger.info(f"Added row {row_idx + 1} data: {row_data}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error scraping Coinglass: {str(e)}")
            return []
        finally:
            if driver:
                driver.quit()
                logger.info("Browser session closed")
    
    @staticmethod
    def export_to_excel(data: List[Dict[str, Union[str, int, float]]], output_dir: str = None, output_file: str = None) -> str:
        """
        Export liquidation data to Excel file, appending with timestamp.
        
        Args:
            data: List of dictionaries containing liquidation data
            output_dir: Optional directory to save the Excel file (default: current directory)
            output_file: Optional specific Excel file to update (default: create new file)
            
        Returns:
            str: Path to the Excel file
        """
        if not data:
            logger.warning("No data to export")
            return ""
            
        try:
            # Add timestamp to the data
            timestamp = datetime.now()
            for entry in data:
                entry['Timestamp'] = timestamp
            
            # Create DataFrame from data
            df = pd.DataFrame(data)
            
            # Create output directory if it doesn't exist
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = os.getcwd()
            
            # Determine the output file
            if output_file and os.path.exists(output_file):
                filepath = output_file
                # Read existing data
                try:
                    existing_df = pd.read_excel(filepath)
                    # Append new data
                    df = pd.concat([existing_df, df], ignore_index=True)
                except Exception as e:
                    logger.error(f"Error reading existing file {filepath}: {str(e)}")
                    return ""
            else:
                # Generate new filename with start timestamp
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                filename = f"coinglass_liquidation_data_start_{timestamp_str}.xlsx"
                filepath = os.path.join(output_dir, filename)
            
            # Export to Excel
            logger.info(f"Exporting data to {filepath}")
            df.to_excel(filepath, index=False, engine='openpyxl')
            logger.info("Export completed successfully")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            return ""

    @staticmethod
    def format_results(data: List[Dict[str, Union[str, int, float]]]) -> str:
        """
        Format liquidation data results into a readable string.
        
        Args:
            data: List of dictionaries containing liquidation data
            
        Returns:
            str: Formatted string representation of the data
        """
        if not data:
            return "No liquidation data available."
            
        output = ["Coinglass Liquidation Data:", "-" * 80]
        
        for idx, entry in enumerate(data, 1):
            output.append(f"Entry #{idx}:")
            output.extend([f"  {key}: {value}" for key, value in entry.items()])
            output.append("-" * 80)
            
        return "\n".join(output)


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test the scraper
    data = CoinglassScraper.get_liquidation_data()
    print(CoinglassScraper.format_results(data))
    
    # Refresh the page and get data again
    data = CoinglassScraper.get_liquidation_data()
    print(CoinglassScraper.format_results(data))
    
    # Export to Excel
    if data:
        excel_file = CoinglassScraper.export_to_excel(data, "/tmp")
        if excel_file:
            print(f"\nData exported to: {excel_file}")

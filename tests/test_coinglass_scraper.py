import pytest
from unittest.mock import Mock, patch
from quantframe.scrapers.coinglass import CoinglassScraper

# Sample test data
MOCK_LIQUIDATION_DATA = [
    {
        'Symbol': 'BTC',
        'Price': '48,123.45',
        '24h Liquidation': 1234567,
        '4h Max Liquidation': 567890
    },
    {
        'Symbol': 'ETH',
        'Price': '2,890.12',
        '24h Liquidation': 345678,
        '4h Max Liquidation': 123456
    }
]

class MockText:
    """Mock class for text elements with clean method."""
    def __init__(self, value):
        self._value = value
        self.text = self  # Make text attribute point to self

    def clean(self, preserve_newlines=False):
        print(f"Cleaning text: {self._value}")  # Debug print
        return str(self._value)

class MockCell:
    """Mock class for table cells."""
    def __init__(self, value):
        print(f"Creating cell with value: {value}")  # Debug print
        self.text = MockText(value)

class MockRow:
    """Mock class for table rows."""
    def __init__(self, data):
        print(f"Creating row with data: {data}")  # Debug print
        self._data = data

    def css(self, selector):
        print(f"Row CSS selector: {selector}")  # Debug print
        if selector == 'td':
            cells = [MockCell(value) for value in self._data.values()]
            print(f"Created cells: {len(cells)}")  # Debug print
            return cells
        return []

class MockTable:
    """Mock class for HTML table."""
    def __init__(self):
        print("Creating mock table")  # Debug print
        self._headers = ['Symbol', 'Price', '24h Liquidation', '4h Max Liquidation']
        self._data = MOCK_LIQUIDATION_DATA

    def css(self, selector):
        print(f"Table CSS selector: {selector}")  # Debug print
        if selector == 'thead th::text':
            return [MockText(header) for header in self._headers]
        elif selector == 'tbody tr':
            rows = [MockRow(row) for row in self._data]
            print(f"Created rows: {len(rows)}")  # Debug print
            return rows
        return []

@pytest.fixture
def mock_response():
    """Create a mock PlayWright response with test data."""
    print("\nCreating mock response")  # Debug print
    mock_response = Mock()
    mock_response.status = 200
    mock_response.css_first = lambda _: MockTable()
    return mock_response

class TestCoinglassScraper:
    """Test suite for CoinglassScraper class."""

    def test_base_url_constant(self):
        """Test that the base URL is correctly defined."""
        print("\nTesting base URL")  # Debug print
        assert CoinglassScraper.BASE_URL == "https://www.coinglass.com/LiquidationData"

    def test_successful_data_fetch(self, mock_response):
        """Test successful fetching and parsing of liquidation data."""
        print("\nTesting successful data fetch")  # Debug print
        with patch('quantframe.scrapers.coinglass.PlayWrightFetcher') as mock_fetcher:
            # Configure mock
            mock_fetcher.fetch.return_value = mock_response

            # Get data
            print("Getting data...")  # Debug print
            data = CoinglassScraper.get_liquidation_data()
            print(f"Got data: {data}")  # Debug print

            # Verify PlayWrightFetcher was called with correct parameters
            mock_fetcher.fetch.assert_called_once_with(
                url=CoinglassScraper.BASE_URL,
                headless=True,
                network_idle=True,
                stealth=True,
                real_chrome=True,
                timeout=30000
            )

            # Verify returned data
            assert len(data) == 2
            assert data[0]['Symbol'] == 'BTC'
            assert data[0]['Price'] == '48,123.45'
            assert data[0]['24h Liquidation'] == 1234567
            assert data[0]['4h Max Liquidation'] == 567890

    def test_failed_connection(self):
        """Test handling of failed connection."""
        print("\nTesting failed connection")  # Debug print
        with patch('quantframe.scrapers.coinglass.PlayWrightFetcher') as mock_fetcher:
            # Configure mock for failed connection
            mock_response = Mock()
            mock_response.status = 404
            mock_fetcher.fetch.return_value = mock_response

            # Get data
            data = CoinglassScraper.get_liquidation_data()

            # Verify empty list is returned on error
            assert data == []

    def test_missing_table(self):
        """Test handling of missing table in response."""
        print("\nTesting missing table")  # Debug print
        with patch('quantframe.scrapers.coinglass.PlayWrightFetcher') as mock_fetcher:
            # Configure mock for missing table
            mock_response = Mock()
            mock_response.status = 200
            mock_response.css_first = lambda _: None
            mock_fetcher.fetch.return_value = mock_response

            # Get data
            data = CoinglassScraper.get_liquidation_data()

            # Verify empty list is returned when table is missing
            assert data == []

    def test_format_results_with_data(self):
        """Test formatting of liquidation data."""
        print("\nTesting format results with data")  # Debug print
        formatted = CoinglassScraper.format_results(MOCK_LIQUIDATION_DATA)
        
        # Verify formatting
        assert "Coinglass Liquidation Data:" in formatted
        assert "Entry #1:" in formatted
        assert "Symbol: BTC" in formatted
        assert "24h Liquidation: 1234567" in formatted
        assert "Entry #2:" in formatted
        assert "Symbol: ETH" in formatted

    def test_format_results_empty_data(self):
        """Test formatting of empty data."""
        print("\nTesting format results with empty data")  # Debug print
        formatted = CoinglassScraper.format_results([])
        assert formatted == "No liquidation data available."

    def test_live_data_fetch(self):
        """Test fetching live data from Coinglass website."""
        print("\nTesting live data fetch from Coinglass")
        
        # Get live data
        data = CoinglassScraper.get_liquidation_data()
        print("\nFetched live data:")
        print(CoinglassScraper.format_results(data))
        
        # Basic validation of the data structure
        assert len(data) > 0, "Should fetch at least one cryptocurrency data"
        
        # Verify data structure
        first_entry = data[0]
        assert 'Symbol' in first_entry, "Each entry should have a Symbol"
        assert 'Price' in first_entry, "Each entry should have a Price"
        assert '24h Liquidation' in first_entry, "Each entry should have 24h Liquidation"
        assert '4h Max Liquidation' in first_entry, "Each entry should have 4h Max Liquidation"
        
        # Verify data types
        assert isinstance(first_entry['Symbol'], str), "Symbol should be a string"
        assert isinstance(first_entry['24h Liquidation'], (int, float)), "24h Liquidation should be numeric"
        assert isinstance(first_entry['4h Max Liquidation'], (int, float)), "4h Max Liquidation should be numeric"

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

"""Script to run all tests."""
import unittest
import sys
import os
from tests.test_coinglass_scraper import TestCoinglassScraper

def run_tests():
    """Discover and run all tests."""
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(project_root, 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoinglassScraper)
    unittest.TextTestRunner(verbosity=2).run(suite)

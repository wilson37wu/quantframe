import os
from pathlib import Path

def create_directory_structure():
    # Project root
    root = Path(__file__).parent
    
    # Main package directory
    dirs = [
        # Core package structure
        'quantframe',
        'quantframe/core',
        'quantframe/data',
        'quantframe/strategy',
        'quantframe/backtesting',
        'quantframe/analytics',
        'quantframe/risk',
        'quantframe/utils',
        
        # Data subdirectories
        'quantframe/data/sources',
        'quantframe/data/processors',
        'quantframe/data/storage',
        
        # Strategy subdirectories
        'quantframe/strategy/components',
        'quantframe/strategy/signals',
        'quantframe/strategy/portfolio',
        
        # Analytics subdirectories
        'quantframe/analytics/performance',
        'quantframe/analytics/risk',
        'quantframe/analytics/reporting',
        
        # Configuration and tests
        'config',
        'tests',
        'examples',
        'docs',
        'notebooks'
    ]
    
    # Create directories
    for dir_path in dirs:
        full_path = root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        # Create __init__.py files
        if 'quantframe' in str(full_path):
            init_file = full_path / '__init__.py'
            init_file.touch()
    
    print("Project directory structure created successfully!")

if __name__ == '__main__':
    create_directory_structure()

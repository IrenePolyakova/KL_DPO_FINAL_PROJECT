#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point script for running the File Preparation app.
This script properly handles the package imports and runs the Streamlit app.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Import and run the file preparation app
if __name__ == "__main__":
    from translation_evaluator import file_preparation_app
    # The file_preparation_app.py will be executed by Streamlit

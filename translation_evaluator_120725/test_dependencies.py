#!/usr/bin/env python3
"""
Test script to verify that all dependencies are working correctly
"""
import sys
import os

# Add the translation_utils directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'translation_utils'))

def test_imports():
    """Test all critical imports"""
    try:
        import pydantic
        print(f"✓ Pydantic {pydantic.__version__} imported successfully")
        
        from translation_utils.tmx_utils import create_tmx_file, TmxEntry
        print("✓ TMX utils imported successfully")
        
        from translation_utils.translator import Translator
        print("✓ Translator imported successfully")
        
        from translation_utils.batch_optimizer import BatchOptimizer
        print("✓ BatchOptimizer imported successfully")
        
        from translation_utils.model_loader import ModelLoader
        print("✓ ModelLoader imported successfully")
        
        print("\n✓ All imports successful! Dependencies resolved.")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

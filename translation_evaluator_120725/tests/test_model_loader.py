"""
Tests for model loading functionality.
"""
import pytest
from pathlib import Path
import tempfile
import zipfile
import json

from translation_utils.model_loader import load_model_from_zip

def create_test_model_zip():
    """Creates a test ZIP file with mock model files"""
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        with zipfile.ZipFile(tmp, 'w') as zf:
            # Create mock config
            config = {
                "architectures": ["MarianMTModel"],
                "model_type": "marian"
            }
            zf.writestr('config.json', json.dumps(config))
            
            # Create mock tokenizer files
            zf.writestr('tokenizer_config.json', '{}')
            zf.writestr('vocab.json', '{}')
            zf.writestr('source.spm', 'mock spm')
            zf.writestr('target.spm', 'mock spm')
            
            # Create mock model file
            zf.writestr('pytorch_model.bin', b'mock model')
        
        return tmp.name

def test_load_model_from_zip():
    """Test loading model from ZIP file"""
    # Create test ZIP
    test_zip = create_test_model_zip()
    
    try:
        # Test loading
        with tempfile.TemporaryDirectory() as tmpdir:
            model, tokenizer = load_model_from_zip(test_zip, tmpdir)
            
            # Basic assertions
            assert model is not None
            assert tokenizer is not None
            
    finally:
        # Cleanup
        Path(test_zip).unlink()

def test_load_model_from_zip_missing_files():
    """Test loading model from incomplete ZIP file"""
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        with zipfile.ZipFile(tmp, 'w') as zf:
            zf.writestr('config.json', '{}')
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with pytest.raises(Exception):
                    load_model_from_zip(tmp.name, tmpdir)
        finally:
            Path(tmp.name).unlink() 
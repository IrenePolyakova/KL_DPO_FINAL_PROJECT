"""
Tests for translation functionality.
"""
import pytest
import torch
from transformers import MarianMTModel, MarianTokenizer

from translation_utils.translator import translate_texts, translate_chunk

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def generate(self, *args, **kwargs):
        # Mock translation - returns same token IDs
        return args[0]
    
    def to(self, device):
        return self

class MockTokenizer:
    def __init__(self):
        self.model_max_length = 512
    
    def batch_encode_plus(self, texts, **kwargs):
        # Mock tokenization - returns dummy tensor
        return {
            'input_ids': torch.tensor([[1, 2, 3]] * len(texts)),
            'attention_mask': torch.tensor([[1, 1, 1]] * len(texts))
        }
    
    def decode(self, token_ids, **kwargs):
        # Mock decoding - returns original text
        return "Translated text"

@pytest.fixture
def mock_model_and_tokenizer():
    return MockModel(), MockTokenizer()


def test_translate_chunk() -> None:
    """Test translation of a single chunk"""
    model, tokenizer = mock_model_and_tokenizer()
    texts = ["Test text 1", "Test text 2"]
    
    translations = translate_chunk(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512,
        num_beams=4
    )
    
    assert len(translations) == len(texts)
    assert all(isinstance(t, str) for t in translations)

def test_translate_texts():
    """Test translation of multiple texts"""
    model, tokenizer = mock_model_and_tokenizer()
    texts = ["Test text 1", "Test text 2", "Test text 3"]
    
    translations, _ = translate_texts(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512,
        num_beams=4,
        device='cpu',
        show_progress=False
    )
    
    assert len(translations) == len(texts)
    assert all(isinstance(t, str) for t in translations)

def test_translate_texts_empty():
    """Test translation of empty text list"""
    model, tokenizer = mock_model_and_tokenizer()
    texts = []
    
    translations, _ = translate_texts(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512,
        num_beams=4,
        device='cpu',
        show_progress=False
    )
    
    assert len(translations) == 0

def test_translate_texts_with_empty_string():
    """Test translation with empty string in input"""
    model, tokenizer = mock_model_and_tokenizer()
    texts = ["Test text", "", "Another text"]
    
    translations, _ = translate_texts(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512,
        num_beams=4,
        device='cpu',
        show_progress=False
    )
    
    assert len(translations) == len(texts)
    assert translations[1] == ""  # Empty input should result in empty output 
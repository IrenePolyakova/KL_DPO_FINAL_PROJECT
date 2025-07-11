from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
from pathlib import Path
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

class TranslationInput(BaseModel):
    text: str = Field(..., min_length=1)
    source_lang: str = Field(..., min_length=2, max_length=5)
    target_lang: str = Field(..., min_length=2, max_length=5)

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text must not be empty')
        return v.strip()

class TranslationOutput(BaseModel):
    source_text: str
    translated_text: str
    model_name: str
    metrics: Optional[Dict[str, float]] = None
    translation_time: float

class ModelConfig(BaseModel):
    model_path: Path
    batch_size: int = Field(32, ge=1)
    max_length: int = Field(512, ge=1)
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu")
    num_beams: int = Field(4, ge=1)
    
    class Config:
        arbitrary_types_allowed = True

class LoadedModel(BaseModel):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    config: ModelConfig
    
    class Config:
        arbitrary_types_allowed = True

class TranslationMetrics(BaseModel):
    bleu: float = Field(..., ge=0, le=100)
    chrf: float = Field(..., ge=0, le=100)
    ter: float = Field(..., ge=0)
    
    @validator('bleu', 'chrf', 'ter')
    def validate_metrics(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('Metric must be a number')
        return float(v)

class BatchConfig(BaseModel):
    min_batch_size: int = Field(1, ge=1)
    max_batch_size: int = Field(128, ge=1)
    optimal_batch_size: Optional[int] = None
    memory_limit: int = Field(1024 * 1024 * 1024)  # 1GB default
    
    @validator('max_batch_size')
    def validate_batch_sizes(cls, v, values):
        if 'min_batch_size' in values and v < values['min_batch_size']:
            raise ValueError('max_batch_size must be greater than min_batch_size')
        return v 
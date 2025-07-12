import pytest
import torch
from translation_utils.batch_optimizer import BatchOptimizer
from translation_utils.models import BatchConfig, ModelConfig

@pytest.fixture
def sample_configs():
    model_config = ModelConfig(
        model=Mock(),
        tokenizer=Mock(),
        device="cpu",
        max_length=512
    )
    batch_config = BatchConfig(
        min_batch_size=1,
        max_batch_size=32,
        memory_limit=1000000000
    )
    return model_config, batch_config

def test_batch_optimizer_initialization(sample_configs):
    model_config, batch_config = sample_configs
    optimizer = BatchOptimizer(model_config, batch_config)
    assert optimizer.optimal_batch_size is None
    assert len(optimizer.performance_history) == 0

def test_invalid_batch_config():
    model_config, _ = sample_configs()
    with pytest.raises(ValueError):
        BatchOptimizer(model_config, BatchConfig(
            min_batch_size=0,
            max_batch_size=32,
            memory_limit=1000000000
        ))

def test_measure_batch_performance(sample_configs):
    model_config, batch_config = sample_configs
    optimizer = BatchOptimizer(model_config, batch_config)
    
    time_taken, memory_used = optimizer.measure_batch_performance(
        batch_size=1,
        sample_text="Test text"
    )
    assert isinstance(time_taken, float)
    assert isinstance(memory_used, float)
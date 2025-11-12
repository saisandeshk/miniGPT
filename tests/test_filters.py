"""
Unit tests for dataset filtering functionality.
"""

import pytest
from datasets import Dataset as HFDataset

from dataset.filters import (
    filter_by_length,
    filter_by_quality,
    calculate_quality_score,
    apply_filters
)


def create_test_dataset():
    """Create a small test dataset."""
    data = {
        'text': [
            "Short",  # Too short
            "This is a good sentence with proper punctuation.",  # Good
            "word word word word word word word word",  # Repetitive
            "A" * 5000,  # Too long
            "Another good sentence. It has multiple sentences!",  # Good
        ]
    }
    return HFDataset.from_dict(data)


def test_filter_by_length():
    """Test length-based filtering."""
    dataset = create_test_dataset()
    
    filtered = filter_by_length(dataset, min_length=20, max_length=100)
    
    # Should keep 3 samples (indices 1, 2, and 4)
    # Index 0 ("Short", len=5) - too short
    # Index 1 ("This is a good sentence...", len=48) - OK
    # Index 2 ("word word word...", len=39) - OK  
    # Index 3 ("AAA...", len=5000) - too long
    # Index 4 ("Another good sentence...", len=49) - OK
    assert len(filtered) == 3
    assert "good sentence" in filtered[0]['text']


def test_calculate_quality_score():
    """Test quality score calculation."""
    # Good text
    score1 = calculate_quality_score("This is a good sentence. It has variety and structure!")
    assert score1 > 0.7
    
    # Repetitive text
    score2 = calculate_quality_score("word word word word word word word word word word")
    assert score2 < 0.6
    
    # No punctuation
    score3 = calculate_quality_score("this text has no punctuation marks at all")
    assert score3 < 0.8
    
    # Too short
    score4 = calculate_quality_score("Hi")
    assert score4 == 0.0


def test_filter_by_quality():
    """Test quality-based filtering."""
    dataset = create_test_dataset()
    
    filtered = filter_by_quality(dataset, min_score=0.5)
    
    # Should filter out very short and very repetitive samples
    assert len(filtered) <= len(dataset)


def test_apply_filters():
    """Test applying multiple filters in sequence."""
    dataset = create_test_dataset()
    
    filters = [
        {'type': 'length', 'min_length': 10, 'max_length': 200},
        {'type': 'quality', 'min_score': 0.3}
    ]
    
    filtered = apply_filters(dataset, filters, text_field='text')
    
    # Should apply both filters
    assert len(filtered) <= len(dataset)
    assert len(filtered) > 0  # Should keep at least some samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

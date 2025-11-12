"""
Dataset filtering and preprocessing utilities.

Provides quality filters, length filters, and other preprocessing operations.
"""

from typing import List, Dict, Any
from datasets import Dataset


def apply_filters(dataset: Dataset, filters: List[Dict[str, Any]], text_field: str = 'text') -> Dataset:
    """
    Apply a sequence of filters to a dataset.
    
    Args:
        dataset: Input dataset
        filters: List of filter configurations
        text_field: Name of the text field to filter on
        
    Returns:
        Filtered dataset
        
    Examples:
        >>> filters = [
        ...     {'type': 'length', 'min_length': 50, 'max_length': 1000},
        ...     {'type': 'quality', 'min_score': 0.5}
        ... ]
        >>> filtered = apply_filters(dataset, filters, text_field='text')
    """
    original_size = len(dataset)
    
    for filter_config in filters:
        filter_type = filter_config['type']
        
        if filter_type == 'length':
            dataset = filter_by_length(
                dataset,
                min_length=filter_config.get('min_length', 0),
                max_length=filter_config.get('max_length', float('inf')),
                text_field=text_field
            )
        
        elif filter_type == 'quality':
            dataset = filter_by_quality(
                dataset,
                min_score=filter_config.get('min_score', 0.0),
                text_field=text_field
            )
        
        elif filter_type == 'language':
            dataset = filter_by_language(
                dataset,
                languages=filter_config.get('languages', ['en']),
                text_field=text_field
            )
        
        else:
            print(f"Warning: Unknown filter type '{filter_type}', skipping...")
    
    filtered_size = len(dataset)
    removed = original_size - filtered_size
    print(f"  Filters applied: {original_size} → {filtered_size} samples ({removed} removed, {removed/original_size*100:.1f}%)")
    
    return dataset


def filter_by_length(
    dataset: Dataset,
    min_length: int = 0,
    max_length: float = float('inf'),
    text_field: str = 'text'
) -> Dataset:
    """
    Filter dataset by text length (character count).
    
    Args:
        dataset: Input dataset
        min_length: Minimum text length
        max_length: Maximum text length
        text_field: Field containing text
        
    Returns:
        Filtered dataset
    """
    def length_filter(example):
        text = example.get(text_field, '')
        text_len = len(text)
        return min_length <= text_len <= max_length
    
    return dataset.filter(length_filter)


def filter_by_quality(
    dataset: Dataset,
    min_score: float = 0.0,
    text_field: str = 'text'
) -> Dataset:
    """
    Filter dataset by quality score.
    
    Currently implements a simple heuristic-based quality scoring:
    - Checks for reasonable punctuation
    - Checks for sentence structure
    - Penalizes excessive repetition
    
    Args:
        dataset: Input dataset
        min_score: Minimum quality score (0.0 to 1.0)
        text_field: Field containing text
        
    Returns:
        Filtered dataset
    """
    def quality_filter(example):
        text = example.get(text_field, '')
        score = calculate_quality_score(text)
        return score >= min_score
    
    return dataset.filter(quality_filter)


def calculate_quality_score(text: str) -> float:
    """
    Calculate a simple quality score for text.
    
    Heuristics:
    - Has reasonable length
    - Contains punctuation
    - Not too repetitive
    - Has word diversity
    
    Args:
        text: Input text
        
    Returns:
        Quality score between 0.0 and 1.0
    """
    if not text or len(text) < 10:
        return 0.0
    
    score = 1.0
    
    # Check for punctuation (basic sentence structure)
    punctuation_count = sum(1 for c in text if c in '.!?')
    if punctuation_count == 0:
        score *= 0.7
    
    # Check for word diversity
    words = text.lower().split()
    if len(words) > 5:
        unique_words = len(set(words))
        diversity = unique_words / len(words)
        score *= (0.5 + 0.5 * diversity)  # Penalty for low diversity
    
    # Check for excessive repetition (same word repeated many times)
    if len(words) > 10:
        from collections import Counter
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]
        if most_common_count > len(words) * 0.3:  # Same word > 30% of text
            score *= 0.5
    
    # Check for very short sentences (might be low quality)
    sentences = text.split('.')
    if len(sentences) > 2:
        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_sentence_len < 3:  # Very short sentences
            score *= 0.7
    
    return min(1.0, max(0.0, score))


def filter_by_language(
    dataset: Dataset,
    languages: List[str] = ['en'],
    text_field: str = 'text'
) -> Dataset:
    """
    Filter dataset by language.
    
    Note: This is a placeholder implementation. For production use,
    consider using a proper language detection library like langdetect or fasttext.
    
    Args:
        dataset: Input dataset
        languages: List of language codes to keep (e.g., ['en', 'es'])
        text_field: Field containing text
        
    Returns:
        Filtered dataset
    """
    # For now, just return the dataset unchanged
    # TODO: Implement actual language detection when needed
    print(f"  Language filter (placeholder): keeping {languages}")
    return dataset


def remove_duplicates(
    dataset: Dataset,
    text_field: str = 'text',
    method: str = 'exact'
) -> Dataset:
    """
    Remove duplicate samples from dataset.
    
    Args:
        dataset: Input dataset
        text_field: Field to check for duplicates
        method: Deduplication method ('exact', 'fuzzy')
        
    Returns:
        Deduplicated dataset
    """
    if method == 'exact':
        # Remove exact duplicates
        seen = set()
        
        def is_unique(example):
            text = example.get(text_field, '')
            if text in seen:
                return False
            seen.add(text)
            return True
        
        return dataset.filter(is_unique)
    
    elif method == 'fuzzy':
        # TODO: Implement fuzzy deduplication (e.g., MinHash)
        print("Warning: Fuzzy deduplication not implemented yet, using exact")
        return remove_duplicates(dataset, text_field, method='exact')
    
    else:
        raise ValueError(f"Unknown deduplication method: {method}")

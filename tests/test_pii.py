#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: test_pi.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-12-02 23:44:28
# Modified: 2024-12-03 00:38:31

import pytest
from pii.SAMPLE_DATA import SAMPLE_DATA
from pii import (
    validate_patterns,
    extract_patterns,
    stitch_results,
    sanitize_results,
    get_labels,
    extract,
    PATTERNS,
)

# Test input text (empty by default, replace during testing)
text = SAMPLE_DATA 

def test_validate_patterns():
    """
    Test validate_patterns function to ensure all patterns are valid.
    """
    valid_patterns = validate_patterns(PATTERNS)
    assert isinstance(valid_patterns, list), "validate_patterns should return a list"
    assert len(valid_patterns) > 0, "validate_patterns returned no valid patterns"


def test_extract_patterns():
    """
    Test extract_patterns function to ensure matches are grouped by label.
    """
    matches = extract_patterns(text, PATTERNS)
    assert isinstance(matches, dict), "extract_patterns should return a dictionary"
    for label, values in matches.items():
        assert isinstance(values, list), f"Matches for {label} should be a list"


def test_stitch_results():
    """
    Test stitch_results function to ensure contiguous matches are combined.
    """
    matches_by_label = extract_patterns(text, PATTERNS)
    stitched = stitch_results(text, matches_by_label)
    assert isinstance(stitched, dict), "stitch_results should return a dictionary"
    for label, values in stitched.items():
        assert isinstance(values, list), f"Stitched matches for {label} should be a list"


def test_sanitize_results():
    """
    Test sanitize_results function to ensure output is sanitized and deduplicated.
    """
    matches_by_label = extract_patterns(text, PATTERNS)
    stitched = stitch_results(text, matches_by_label)
    sanitized = sanitize_results(stitched)
    assert isinstance(sanitized, dict), "sanitize_results should return a dictionary"
    for label, values in sanitized.items():
        assert isinstance(values, list), f"Sanitized matches for {label} should be a list"
        assert len(values) == len(set(values)), f"Matches for {label} should be deduplicated"


def test_get_labels():
    """
    Test get_labels function to ensure it retrieves all available labels.
    """
    labels = get_labels(PATTERNS)
    assert isinstance(labels, list), "get_labels should return a list"
    assert len(labels) > 0, "get_labels returned no labels"
    for label in labels:
        assert isinstance(label, str), f"Label {label} should be a string"


def test_extract_all():
    """
    Test extract function without labels to ensure all patterns are extracted.
    """
    results = extract(text, PATTERNS)
    assert isinstance(results, dict), "extract should return a dictionary"
    for label, values in results.items():
        assert isinstance(values, list), f"Extracted matches for {label} should be a list"


def test_extract_with_labels():
    """
    Test extract function with specific labels to ensure filtered extraction works.
    """
    specific_labels = ["ipv4", "email"]
    results = extract(text, PATTERNS, labels=specific_labels)
    assert isinstance(results, dict), "extract should return a dictionary"
    for label in specific_labels:
        assert label in results, f"{label} should be in extracted results"
        assert isinstance(results[label], list), f"Matches for {label} should be a list"


def test_extract_with_single_label():
    """
    Test extract function with a single label as a string.
    """
    single_label = "email"
    results = extract(text, PATTERNS, labels=single_label)
    assert isinstance(results, dict), "extract should return a dictionary"
    assert single_label in results, f"{single_label} should be in extracted results"
    assert isinstance(results[single_label], list), f"Matches for {single_label} should be a list"


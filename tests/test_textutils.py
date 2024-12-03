#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: test_textutils.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-12-02 15:01:25
# Modified: 2024-12-02 17:40:39

import pytest
from i.utils.textutils import (
    to_lowercase,
    remove_special_chars,
    normalize_whitespace,
    remove_numbers,
    remove_accents,
    expand_contractions,
    validate_regex,
    is_regex,
    remove,
    replace,
    tokenize_words,
    tokenize_sentences,
    word_frequency,
    generate_ngrams,
    corpus_statistics,
    flatten,
    remove_stopwords,
    lemmatize_words,
    correct_spelling,

)

def test_to_lowercase():
    text = "THIS IS A TEST"
    assert to_lowercase(text) == "this is a test"

def test_remove_special_chars():
    text = "Hello, World! @#$"
    assert remove_special_chars(text) == "Hello, World"

def test_remove_numbers():
    text = "This is test 1234."
    assert remove_numbers(text) == "This is test ."

def test_normalize_whitespace():
    text = "This  is   a test.\n\nAnother line."
    expected = "This is a test. Another line."
    assert normalize_whitespace(text) == expected

def test_remove_keyword():
    text = "Remove this keyword from the text."
    assert remove(text, "keyword") == "Remove this  from the text."

def test_remove_regex():
    text = "Remove numbers 1234 and special 5678."
    assert remove(text, r"\d+") == "Remove numbers  and special ."

def test_replace_keyword():
    text = "Replace this keyword in the text."
    assert replace(text, "keyword", "pattern") == "Replace this pattern in the text."

def test_replace_regex():
    text = "Replace numbers 1234 and special 5678."
    assert replace(text, r"\d+", "#") == "Replace numbers # and special #."

def test_tokenize_words():
    text = "This is a test."
    expected = ["This", "is", "a", "test"]
    assert tokenize_words(text) == expected

def test_tokenize_sentences():
    text = "This is a test. Another sentence!"
    expected = ["This is a test.", "Another sentence!"]
    assert tokenize_sentences(text) == expected

def test_word_frequency():
    text = "This is a test. This test is a test."
    freq = word_frequency(text)
    assert freq["test"] == 3
    assert freq["is"] == 2

def test_generate_ngrams():
    text = "This is a test"
    bigrams = generate_ngrams(text, 2)
    expected = [("This", "is"), ("is", "a"), ("a", "test")]
    assert bigrams == expected

def test_corpus_statistics():
    # Typical input
    text = "This is a test. Another test sentence!"
    stats = corpus_statistics(text)
    assert stats["word_count"] == 7
    assert stats["unique_words"] == 6
    assert stats["sentence_count"] == 2
    assert stats["avg_word_length"] > 2
    assert stats["avg_sentence_length"] > 3

    # Edge Case: Empty input
    empty_text = ""
    stats = corpus_statistics(empty_text)
    assert stats["word_count"] == 0
    assert stats["unique_words"] == 0
    assert stats["sentence_count"] == 0
    assert stats["avg_word_length"] == 0
    assert stats["avg_sentence_length"] == 0

    # Edge Case: Input with only whitespace
    whitespace_text = "   "
    stats = corpus_statistics(whitespace_text)
    assert stats["word_count"] == 0
    assert stats["unique_words"] == 0
    assert stats["sentence_count"] == 0
    assert stats["avg_word_length"] == 0
    assert stats["avg_sentence_length"] == 0

    # Edge Case: Input with no sentences
    single_word_text = "Singleword"
    stats = corpus_statistics(single_word_text)
    assert stats["word_count"] == 1
    assert stats["unique_words"] == 1
    assert stats["sentence_count"] == 1  # Even a single word can be considered a "sentence"
    assert stats["avg_word_length"] == len("Singleword")
    assert stats["avg_sentence_length"] == 1

def test_nested_structures():
    nested_data = {
        "key1": "Remove This! 123\n",
        "key2": ["   Keyword Example    ", "Another keyword here!"],
        "key3": {"nested_key": "Text with   \n\n Excessive Spaces \n"},
    }
    cleaned_nested = normalize_whitespace(nested_data)
    assert cleaned_nested["key1"] == "Remove This! 123"
    assert cleaned_nested["key2"][0] == "Keyword Example"
    assert cleaned_nested["key3"]["nested_key"] == "Text with Excessive Spaces"

def test_remove_accents():
    text = "Café naïve résumé"
    expected = "Cafe naive resume"
    assert remove_accents(text) == expected

def test_expand_contractions():
    text = "I'm going to the park. He can't join you at the park."
    expected = "I am going to the park. He cannot join you at the park."
    assert expand_contractions(text) == expected

def test_validate_regex():
    valid_regex = r"\d+"
    invalid_regex = r"\d++"

    # Valid regex
    assert validate_regex(valid_regex).pattern == valid_regex

    # Invalid regex
    try:
        validate_regex(invalid_regex)
    except ValueError as e:
        assert "Invalid regex pattern" in str(e)

def test_is_regex():
    assert is_regex(r"\d+") is True

def test_flatten():
    nested = [1, [2, 3], [[4, 5], 6], 7]
    expected = [1, 2, 3, 4, 5, 6, 7]
    assert flatten(nested) == expected

    nested_with_tuples = [1, (2, 3), [[4, (5, 6)], 7]]
    expected_with_tuples = [1, 2, 3, 4, 5, 6, 7]
    assert flatten(nested_with_tuples) == expected_with_tuples

    nested_mixed = ["a", ["b", ["c", "d"]], [["e"]], "f"]
    expected_mixed = ["a", "b", "c", "d", "e", "f"]
    assert flatten(nested_mixed) == expected_mixed

    # Edge Case: Flat input remains unchanged
    already_flat = [1, 2, 3, 4]
    assert flatten(already_flat) == already_flat

    # Edge Case: Empty input
    assert flatten([]) == []


def test_remove_stopwords():
    text = "This is a simple test with some stopwords."
    expected = "simple test stopwords."
    assert remove_stopwords(text) == expected

def test_lemmatize_words():
    text = "The leaves are falling off the trees."
    expected = "The leaf be fall off the tree ."
    result = lemmatize_words(text)
    print(f"Input: {text}")
    print(f"Output: {result}")
    print(f"Expected: {expected}")
    assert result == expected

def test_correct_spelling():
    # Basic test case
    text = "This is a imple test with a typo."
    expected = "This is an simple test with a typo."
    result = correct_spelling(text)
    print(f"Input: {text}")
    print(f"Output: {result}")
    print(f"Expected: {expected}")
    assert result == expected, f"Expected '{expected}', but got '{result}'"

    # Edge case: Perfectly correct input
    text = "This is a perfectly written sentence."
    expected = "This is a perfectly written sentence."
    result = correct_spelling(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"



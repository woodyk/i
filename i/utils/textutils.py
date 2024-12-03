#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: textutils.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-12-02 14:55:33
# Modified: 2024-12-02 17:47:44

import re
import string
import unicodedata
import language_tool_python
from collections.abc import Mapping, Iterable
from collections import Counter
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from contractions import fix

# Initialize NLP tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Registry for normalization and manipulation functions
NORMALIZATIONS = {}

def get_wordnet_pos(tag):
    """Maps NLTK POS tags to WordNet POS tags."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def normalization(func):
    """
    Decorator to register a function as a normalization step
    and make it applicable to any object type.
    """
    def wrapper(value, *args, **kwargs):
        try:
            if isinstance(value, str):
                return func(value, *args, **kwargs)  # Apply to strings directly
            elif isinstance(value, Mapping):
                # Apply to keys and values for dictionaries
                return {wrapper(k, *args, **kwargs): wrapper(v, *args, **kwargs) for k, v in value.items()}
            elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                # Apply to each element in an iterable
                return type(value)(wrapper(v, *args, **kwargs) for v in value)
            else:
                return value  # Return other types unchanged
        except Exception as e:
            raise ValueError(f"Error applying {func.__name__}: {e}")

    NORMALIZATIONS[func.__name__] = wrapper
    return wrapper

# --- Text Cleaning Functions ---
@normalization
def to_lowercase(value):
    """Converts strings to lowercase."""
    return value.lower()

@normalization
def remove_special_chars(value):
    """
    Removes special characters from strings and normalizes whitespace.
    """
    # Remove special characters
    cleaned = re.sub(r'([^\-\.,A-Za-z0-9 ]|#)+', '', value)
    # Normalize whitespace
    return ' '.join(cleaned.split())

@normalization
def remove_stopwords(value):
    """Removes common stopwords from strings."""
    return ' '.join([word for word in value.split() if word.lower() not in stop_words])


@normalization
def lemmatize_words(value):
    """
    Lemmatizes words in a string using POS tagging for accurate lemmatization.
    """
    if isinstance(value, str):
        try:
            tokens = word_tokenize(value)  # Tokenize the text
            pos_tags = pos_tag(tokens)  # Get POS tags for each token
            lemmatized = [
                lemmatizer.lemmatize(word, get_wordnet_pos(tag)) if get_wordnet_pos(tag) else lemmatizer.lemmatize(word)
                for word, tag in pos_tags
            ]
            return ' '.join(lemmatized)
        except Exception as e:
            raise ValueError(f"Error in lemmatize_words: {e}\nInput: {value}")
    return value

@normalization
def correct_spelling(value):
    """
    Corrects spelling and grammar in text using LanguageTool.
    """
    tool = language_tool_python.LanguageTool('en-US')
    if isinstance(value, str):
        try:
            corrected = tool.correct(value)
            return corrected.strip()  # Ensure no leading/trailing spaces
        except Exception as e:
            raise RuntimeError(f"Error in correct_spelling: {e}\nInput: {value}")
    return value  # Return non-string types unchanged

@normalization
def remove_numbers(value):
    """Removes numeric characters from strings."""
    return re.sub(r'\d+', '', value)

@normalization
def remove_accents(value):
    """Removes accents and diacritics from strings."""
    return ''.join(c for c in unicodedata.normalize('NFKD', value) if not unicodedata.combining(c))

@normalization
def expand_contractions(value):
    """Expands contractions in strings."""
    return fix(value)

@normalization
def remove_duplicates(value):
    """
    Removes duplicate words from strings, preserving the original word order.
    Considers words as duplicates even if they are followed by punctuation,
    and retains punctuation in the final output.
    """
    if isinstance(value, str):
        words = value.split()
        seen = set()
        result = []
        for word in words:
            # Normalize the word for comparison (strip punctuation and lowercase)
            cleaned_word = word.strip(string.punctuation).lower()
            if cleaned_word not in seen:
                seen.add(cleaned_word)
                result.append(word)  # Append the original word with punctuation intact
        return ' '.join(result)
    return value


@normalization
def normalize_whitespace(value):
    """
    Removes excessive newlines (\\n+) and spaces (\\s+) from strings.
    Consolidates multiple newlines or spaces into a single one.
    """
    if isinstance(value, str):
        value = re.sub(r'\n+', '\n', value)  # Replace multiple newlines with a single newline
        value = re.sub(r'\s+', ' ', value)  # Replace multiple spaces with a single space
        return value.strip()
    return value

# --- Text Manipulation Functions ---
@normalization
def remove(value, pattern):
    """
    Removes occurrences of a pattern (string keyword or regex) from the object.
    :param value: The object to process.
    :param pattern: The string or regex pattern to remove.
    """
    if is_regex(pattern):
        regex = validate_regex(pattern)
        return regex.sub('', value)
    else:
        return value.replace(pattern, '')

@normalization
def replace(value, pattern, replacement):
    """
    Replaces occurrences of a pattern (string keyword or regex) with a replacement in the object.
    :param value: The object to process.
    :param pattern: The string or regex pattern to replace.
    :param replacement: The replacement string.
    """
    if is_regex(pattern):
        regex = validate_regex(pattern)
        return regex.sub(replacement, value)
    else:
        return value.replace(pattern, replacement)

# --- Helper Functions ---
def validate_regex(pattern):
    """
    Validates if the given pattern is a valid regex.
    :param pattern: The pattern to validate.
    :return: Compiled regex object if valid, otherwise raises an informative exception.
    """
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

def is_regex(pattern):
    """
    Determines if the input is a valid regex pattern.
    :param pattern: The pattern to check.
    :return: True if the input is a valid regex pattern, False otherwise.
    """
    if isinstance(pattern, re.Pattern):
        return True
    try:
        re.compile(pattern)  # Try compiling the pattern
        return True
    except re.error:
        return False

# --- Analytical Tools ---
def tokenize_words(value):
    """Splits text into a list of words."""
    return apply_recursively(value, lambda v: re.findall(r'\b\w+\b', v))

def tokenize_sentences(value):
    """Splits text into a list of sentences."""
    return apply_recursively(value, lambda v: re.split(r'(?<=[.!?])\s+', v))

def word_frequency(value, n=None):
    """Computes word frequencies in the text."""
    tokens = tokenize_words(value)  # Ensure proper tokenization
    if isinstance(tokens, list):
        flat_tokens = [token.lower() for sublist in tokens for token in (sublist if isinstance(sublist, list) else [sublist])]
    else:
        flat_tokens = tokens
    freq = Counter(flat_tokens)
    return freq.most_common(n) if n else freq

def generate_ngrams(value, n=2):
    """Generates n-grams from text."""
    def ngrams(tokens, n):
        return list(zip(*[tokens[i:] for i in range(n)]))
    tokens = tokenize_words(value)
    flat_tokens = [token for sublist in tokens for token in (sublist if isinstance(sublist, list) else [sublist])]
    return ngrams(flat_tokens, n)

def corpus_statistics(value):
    """Computes basic text statistics."""
    def stats(text):
        # Remove punctuation to ensure proper word counting
        cleaned_text = re.sub(r'[^\w\s]', '', text.strip())
        words = re.findall(r'\b\w+\b', cleaned_text)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip()) if text.strip() else []
        return {
            "word_count": len(words),  # Total word count
            "unique_words": len(set(words)),  # Count of unique words
            "sentence_count": len(sentences) if sentences else 0,  # Number of sentences
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,  # Average word length
            "avg_sentence_length": sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0,  # Average sentence length
        }
    return apply_recursively(value, stats)


# --- Utility Functions ---
def flatten(nested):
    """
    Flattens a nested list or iterable into a single-level list.
    :param nested: The nested iterable to flatten.
    :return: A single-level list.
    """
    if isinstance(nested, Iterable) and not isinstance(nested, (str, bytes)):
        flattened = []
        for item in nested:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                flattened.extend(flatten(item))
            else:
                flattened.append(item)
        return flattened
    return [nested]  # Wrap non-iterable in a list for consistency

def apply_recursively(value, func):
    """
    Applies a function recursively to all string elements in a nested object.
    :param value: The object to process (string, list, dict, etc.).
    :param func: The function to apply to string elements.
    :return: The processed object with the function applied to all strings.
    """
    if isinstance(value, str):
        return func(value)  # Apply the function to strings
    elif isinstance(value, Mapping):
        # Recursively process dictionaries
        return {k: apply_recursively(v, func) for k, v in value.items()}
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        # Recursively process iterables (e.g., lists, tuples)
        return type(value)(apply_recursively(v, func) for v in value)
    else:
        # Return non-string objects as-is
        return value

# --- Macro Functions ---
def clean_text(value):
    """Applies basic text cleaning steps."""
    return normalize_whitespace(
            remove_special_chars(value)
            )

def prepare_text(value):
    """Prepares text for analysis by cleaning and normalizing it."""
    clean_text = normalize_whitespace(remove_stopwords(remove_special_chars(to_lowercase(value))))
    return lemmatize_words(expand_contractions(clean_text))

def correct_and_analyze(value):
    """Corrects grammar and provides text statistics."""
    corrected = correct_spelling(value)
    stats = corpus_statistics(corrected)
    return corrected, stats


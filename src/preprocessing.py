import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
import string
from typing import List, Tuple
import pandas as pd


def download_nltk_resources():
    """Download required NLTK resources"""
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')


def preprocess_text(text: str, 
                   remove_stopwords: bool = True,
                   lemmatize: bool = True,
                   remove_punctuation: bool = True) -> str:
    """
    Preprocess text with various NLP techniques
    
    Parameters:
    -----------
    text : str
        Input text to preprocess
    remove_stopwords : bool
        Whether to remove stopwords
    lemmatize : bool
        Whether to lemmatize words
    remove_punctuation : bool
        Whether to remove punctuation
    
    Returns:
    --------
    str
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        # Get POS tags for better lemmatization
        pos_tags = pos_tag(tokens)
        tokens = []
        for word, tag in pos_tags:
            if tag.startswith('J'):  # Adjective
                pos = 'a'
            elif tag.startswith('V'):  # Verb
                pos = 'v'
            elif tag.startswith('N'):  # Noun
                pos = 'n'
            elif tag.startswith('R'):  # Adverb
                pos = 'r'
            else:
                pos = 'n'  # Default to noun
            tokens.append(lemmatizer.lemmatize(word, pos))
    
    return ' '.join(tokens)


def extract_financial_entities(text: str) -> List[Tuple[str, str]]:
    """
    Extract financial entities from text using simple patterns
    
    Parameters:
    -----------
    text : str
        Input text
    
    Returns:
    --------
    List[Tuple[str, str]]
        List of (entity, type) tuples
    """
    entities = []
    
    # Currency patterns
    currency_patterns = [
        (r'\\$\\s*\\d+[\\d,.]*\\s*[KkMmBb]?\\b', 'CURRENCY'),
        (r'\\d+[\\d,.]*\\s*[KkMmBb]?\\s*(USD|EUR|GBP|CAD|AUD|JPY)\\b', 'CURRENCY'),
        (r'\\€\\s*\\d+[\\d,.]*\\s*[KkMmBb]?\\b', 'CURRENCY'),
        (r'\\£\\s*\\d+[\\d,.]*\\s*[KkMmBb]?\\b', 'CURRENCY')
    ]
    
    # Financial instrument patterns
    instrument_patterns = [
        (r'\\b(stock|stocks|equity|equities)\\b', 'INSTRUMENT'),
        (r'\\b(bond|bonds|treasury|municipal)\\b', 'INSTRUMENT'),
        (r'\\b(option|options|put|call|derivative)\\b', 'INSTRUMENT'),
        (r'\\b(ETF|etf|mutual fund|index fund)\\b', 'INSTRUMENT'),
        (r'\\b(IRA|ira|401k|roth|traditional)\\b', 'RETIREMENT_ACCOUNT')
    ]
    
    # Financial term patterns
    term_patterns = [
        (r'\\b(dividend|dividends)\\b', 'FINANCIAL_TERM'),
        (r'\\b(interest|rate|APR|APY)\\b', 'FINANCIAL_TERM'),
        (r'\\b(tax|taxes|deduction|credit)\\b', 'FINANCIAL_TERM'),
        (r'\\b(invest|investment|portfolio)\\b', 'FINANCIAL_TERM'),
        (r'\\b(loan|mortgage|refinance)\\b', 'FINANCIAL_TERM')
    ]
    
    # Check all patterns
    all_patterns = currency_patterns + instrument_patterns + term_patterns
    
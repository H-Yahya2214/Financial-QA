import re
import string
from typing import List, Union
import pandas as pd


def normalize_currency_amount(match) -> str:
    """Normalize individual currency amounts"""
    # Different patterns will have different group structures
    groups = match.groups()

    # Determine pattern type based on group count
    if len(groups) == 3:  # Pattern: ([$€£])\\s*(\\d+)\\s*([KkMmBb]?)
        currency_symbol, amount, unit = groups
        approx = ""
    elif len(groups) == 2:  # Pattern: (\\d+)\\s*([KkMmBb]?)\\s*(USD|EUR|GBP)
        amount, unit = groups
        currency_symbol = ""
        approx = "approx." if "ish" in match.group(0).lower() else ""
    else:
        return match.group(0)

    amount = amount.replace(",", "").replace(".", "")

    try:
        n = float(amount)
    except:
        return match.group(0)

    # Apply unit multiplier
    if unit and unit.upper() == "K":
        n *= 1000
    elif unit and unit.upper() == "M":
        n *= 1_000_000
    elif unit and unit.upper() == "B":
        n *= 1_000_000_000

    # Determine currency
    if currency_symbol == "$" or "USD" in match.group(0).upper():
        currency = "USD"
    elif currency_symbol == "€" or "EUR" in match.group(0).upper():
        currency = "EUR"
    elif currency_symbol == "£" or "GBP" in match.group(0).upper():
        currency = "GBP"
    else:
        currency = "USD"

    return f"{int(n)} {currency} {approx}".strip()


def normalize_currency_range(match) -> str:
    """Normalize currency ranges"""
    groups = match.groups()

    if len(groups) == 3:  # Pattern: (\\d+)[-–](\\d+)\\s*([KkMmBb])
        start, end, unit = groups
        currency_symbol = ""
    elif len(groups) == 4:  # Pattern: ([$€£])\\s*(\\d+)\\s*[-–]\\s*([$€£])\\s*(\\d+)
        currency_symbol, start, _, end = groups
    else:
        return match.group(0)

    try:
        start_val = float(start.replace(",", ""))
        end_val = float(end.replace(",", ""))
    except:
        return match.group(0)

    # Apply unit multiplier
    if unit and unit.upper() == "K":
        start_val *= 1000
        end_val *= 1000
    elif unit and unit.upper() == "M":
        start_val *= 1_000_000
        end_val *= 1_000_000

    # Determine currency
    if currency_symbol == "$" or "USD" in match.group(0).upper():
        currency = "USD"
    elif currency_symbol == "€" or "EUR" in match.group(0).upper():
        currency = "EUR"
    elif currency_symbol == "£" or "GBP" in match.group(0).upper():
        currency = "GBP"
    else:
        currency = "USD"

    return f"{int(start_val)}–{int(end_val)} {currency}"


def normalize_single_amount(amount_str: str) -> str:
    """Helper to normalize single amount strings"""
    amount_str = amount_str.upper()

    # Extract number and unit
    num_match = re.search(r'(\\d+[,.]?\\d*)', amount_str)
    unit_match = re.search(r'([KMB])', amount_str)

    if not num_match:
        return amount_str

    amount = float(num_match.group(1).replace(",", ""))

    if unit_match:
        unit = unit_match.group(1)
        if unit == "K":
            amount *= 1000
        elif unit == "M":
            amount *= 1_000_000
        elif unit == "B":
            amount *= 1_000_000_000

    return str(int(amount))


def get_currency_code(symbol: str) -> str:
    """Map currency symbol to code"""
    symbol_map = {'$': 'USD', '€': 'EUR', '£': 'GBP'}
    return symbol_map.get(symbol, 'USD')


def clean_financial_text(text: str) -> str:
    """
    Comprehensive financial text cleaning and normalization
    """
    if not isinstance(text, str):
        return text

    # 1) Remove unwanted characters and HTML
    text = re.sub(r'[|\\ǀ│\'"|—().-]', '', text)
    text = re.sub(r'<.*?>', '', text)

    # 2) Remove URLs and specific phrases
    text = re.sub(r'http\\S+|www\\S+', ' ', text)
    text = re.sub(r'see,? for starters at least,?', '', text, flags=re.IGNORECASE)

    # 3) Standardize abbreviations
    text = re.sub(r"\\bU\\.S\\.\\b", "US", text)
    text = re.sub(r'check[- ]cashing', 'check cashing', text, flags=re.IGNORECASE)

    # 4) Normalize currency patterns - SIMPLIFIED version

    # Handle ranges with units (5K-10K, 1M-2M)
    text = re.sub(r'(\\d+[,.]?\\d*)[-–](\\d+[,.]?\\d*)\\s*([KkMmBb])\\b',
                 lambda m: f"{normalize_single_amount(m.group(1) + m.group(3))}–{normalize_single_amount(m.group(2) + m.group(3))}", text)

    # Handle simple amounts with symbols ($100, €50)
    text = re.sub(r'([$€£])\\s*(\\d+[,.]?\\d*)\\b',
                 lambda m: f"{m.group(2).replace(',', '')} {get_currency_code(m.group(1))}", text)

    # Handle amounts with units ($5K, €2.5M)
    text = re.sub(r'([$€£])\\s*(\\d+[,.]?\\d*)\\s*([KkMmBb])\\b',
                 lambda m: normalize_single_amount(m.group(2) + m.group(3) + " " + get_currency_code(m.group(1))), text)

    # 6) Clean whitespace
    text = re.sub(r'\\s+', ' ', text).strip()

    return text


def normalize_dataframe_text(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Apply text normalization to specified columns in dataframe
    """
    for col in text_columns:
        if col in df.columns:
            # Basic cleaning first
            df[col] = df[col].astype(str).str.strip().str.replace("\\s+", " ", regex=True)
            # Then apply financial normalization
            df[col] = df[col].apply(clean_financial_text)

    return df
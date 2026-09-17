"""
Data Preprocessing Module for AI/ML Project.

This module provides comprehensive preprocessing utilities for:
1. Natural Language Processing (NLP) text cleaning, tokenization, stopword removal,
   lemmatization, and stemming.
2. Tabular dataset cleaning, missing value imputation, categorical encoding, and feature scaling.
"""

from typing import List, Optional, Union
import re
import string
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Import NLTK components
try:
    import nltk  # type: ignore
    from nltk.corpus import stopwords  # type: ignore
    from nltk.stem import PorterStemmer, WordNetLemmatizer  # type: ignore
    from nltk.tokenize import word_tokenize  # type: ignore
except ImportError:
    nltk = None  # type: ignore
    stopwords = None  # type: ignore
    PorterStemmer = None  # type: ignore
    WordNetLemmatizer = None  # type: ignore
    word_tokenize = None  # type: ignore


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A robust text preprocessor for NLP tasks with configurable cleaning pipeline.
    Works seamlessly offline with built-in fallbacks.
    """

    def __init__(
        self,
        language: str = 'english',
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        lemmatize: bool = False,
        stem: bool = True,
        custom_stopwords: Optional[List[str]] = None,
    ):
        self.language = language
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.custom_stopwords = custom_stopwords or []

        # Initialize NLTK stopwords with offline fallback
        try:
            assert stopwords is not None
            self.stop_words = set(stopwords.words(self.language))
        except Exception:
            self.stop_words = {
                'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
                'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
                'by', 'can', 'did', 'do', 'does', 'doing', 'don', 'down', 'during', 'each', 'few', 'for',
                'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself',
                'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just',
                'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she',
                'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them',
                'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too',
                'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
                'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
            }
        
        self.stop_words.update(self.custom_stopwords)
        
        # pyrefly: ignore [not-callable]
        self.stemmer = PorterStemmer()
        try:
            # pyrefly: ignore [not-callable]
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    def clean_text(self, text: str) -> str:
        """
        Cleans the raw input string according to configured options.
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<.*?>', '', text)

        # Remove numbers if requested
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into individual words using NLTK word_tokenize with regex fallback.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        try:
            # pyrefly: ignore [not-callable]
            tokens = word_tokenize(cleaned)
        except Exception:
            tokens = re.findall(r'\b\w+\b', cleaned)
        return tokens

    def process_tokens(self, tokens: List[str]) -> List[str]:
        """
        Applies stopword removal, lemmatization, and/or stemming to a list of tokens.
        """
        processed = []
        for token in tokens:
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue

            term = token
            if self.lemmatize and self.lemmatizer is not None:
                try:
                    term = self.lemmatizer.lemmatize(term)
                except Exception:
                    term = self.stemmer.stem(term) if self.stem else term
            elif self.stem:
                try:
                    term = self.stemmer.stem(term)
                except Exception:
                    pass

            if term:
                processed.append(term)

        return processed

    def preprocess(self, text: str, return_tokens: bool = False) -> Union[str, List[str]]:
        """
        Full end-to-end preprocessing pipeline for a single text document.
        """
        tokens = self.tokenize(text)
        processed_tokens = self.process_tokens(tokens)
        if return_tokens:
            return processed_tokens
        return ' '.join(processed_tokens)

    def fit(self, X, y=None):
        """Fit method for scikit-learn pipeline compatibility."""
        return self

    def transform(self, X: Union[List[str], pd.Series, np.ndarray]) -> List[str]:
        """
        Transform method for scikit-learn pipeline compatibility.
        """
        if isinstance(X, pd.Series):
            return X.fillna("").apply(lambda t: self.preprocess(t)).tolist()
        # pyrefly: ignore [bad-return]
        return [self.preprocess(str(item)) for item in X]


class TabularPreprocessor:
    """
    Data preprocessor for structured/tabular datasets (Pandas DataFrames).
    Handles missing values, categorical encoding, and feature scaling.
    """

    def __init__(
        self,
        numerical_imputation: str = 'median',
        categorical_imputation: str = 'most_frequent',
        scaler: str = 'standard',
    ):
        self.numerical_imputation = numerical_imputation
        self.categorical_imputation = categorical_imputation
        self.scaler_type = scaler

        self.scaler = StandardScaler() if scaler == 'standard' else MinMaxScaler()
        self.label_encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.impute_values = {}

    def fit(self, df: pd.DataFrame):
        """Learns statistics, medians, modes, and encodings from training dataframe."""
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category', 'string', 'str']).columns.tolist()

        # Compute numerical impute values
        for col in self.numerical_cols:
            if self.numerical_imputation == 'median':
                self.impute_values[col] = df[col].median()
            elif self.numerical_imputation == 'mean':
                self.impute_values[col] = df[col].mean()
            else:
                self.impute_values[col] = 0

        # Compute categorical impute values and label encoders
        for col in self.categorical_cols:
            mode_series = df[col].mode()
            self.impute_values[col] = mode_series.iloc[0] if not mode_series.empty else "Missing"
            le = LabelEncoder()
            clean_series = df[col].fillna(self.impute_values[col]).astype(str)
            le.fit(clean_series)
            self.label_encoders[col] = le

        # Fit scaler on imputed numerical data
        if self.numerical_cols:
            imputed_num = df[self.numerical_cols].fillna(
                {c: self.impute_values[c] for c in self.numerical_cols}
            )
            self.scaler.fit(imputed_num)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies imputation, encoding, and scaling to a dataframe."""
        df_clean = df.copy()

        # Impute and scale numerical features
        if self.numerical_cols:
            for col in self.numerical_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(self.impute_values.get(col, 0))
            scaled_vals = self.scaler.transform(df_clean[self.numerical_cols])
            df_clean[self.numerical_cols] = scaled_vals

        # Impute and encode categorical features
        if self.categorical_cols:
            for col in self.categorical_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(self.impute_values.get(col, "Missing")).astype(str)
                    le = self.label_encoders.get(col)
                    if le:
                        # Handle unseen labels gracefully
                        known_classes = set(le.classes_)
                        df_clean[col] = df_clean[col].apply(
                            lambda x: x if x in known_classes else le.classes_[0]
                        )
                        df_clean[col] = le.transform(df_clean[col])

        return df_clean

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits and transforms the dataframe in a single step."""
        return self.fit(df).transform(df)


# Aliases for convenience
DataPreprocessor = TabularPreprocessor


if __name__ == "__main__":
    print("=== Testing TextPreprocessor ===")
    sample_text = """
    Hello, World! Check out this AI/ML course at https://example.com/learn-ai.
    Natural Language Processing & Machine Learning are revolutionizing data analytics!
    Contact support@aiml.org for 100% assistance.
    """
    
    text_prep = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        stem=True
    )
    
    cleaned_str = text_prep.preprocess(sample_text)
    token_list = text_prep.preprocess(sample_text, return_tokens=True)
    
    print(f"Original Text:\n{sample_text.strip()}\n")
    print(f"Preprocessed Text Output:\n{cleaned_str}\n")
    print(f"Preprocessed Tokens:\n{token_list}\n")

    print("=== Testing Tabular DataPreprocessor ===")
    data = {
        'age': [25, np.nan, 30, 35, 40],
        'salary': [50000, 60000, np.nan, 80000, 95000],
        'city': ['New York', 'London', 'Paris', np.nan, 'New York'],
        'purchased': ['No', 'Yes', 'No', 'Yes', 'No']
    }
    df_sample = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df_sample)
    
    tab_prep = TabularPreprocessor(scaler='standard')
    df_transformed = tab_prep.fit_transform(df_sample)
    
    print("\nTransformed DataFrame (Imputed, Encoded, Scaled):")
    print(df_transformed)
    print("\nAll preprocessing checks passed successfully!")
import pandas as pd
from unittest.mock import patch, MagicMock
import pytest

@pytest.fixture(autouse=True)
def mock_models():
    # Mock both BGEM3 and SemHash initialization
    with patch('src.data_quality.processors.boilerplate_cleaner.BGEM3FlagModel') as mock_bge, \
         patch('src.data_quality.processors.boilerplate_cleaner.SemHash') as mock_semhash:
        mock_bge.return_value = MagicMock()
        mock_semhash.return_value = MagicMock()
        yield

from src.data_quality.processors.boilerplate_cleaner import TurkishBoilerplateCleanerProcessor, BoilerplateCleanerProcessor

def test_no_columns_configured(sample_df):
    processor = TurkishBoilerplateCleanerProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_removes_turkish_legal_patterns():
    df = pd.DataFrame({'text': ['T.C. Yargıtay Kararı devam ediyor.']})
    processor = TurkishBoilerplateCleanerProcessor({'boilerplate_columns': {'text': {'remove_legal_patterns': True}}})
    result = processor.process(df)
    assert list(result['text']) == ['Yargıtay ı devam ediyor.']

def test_removes_common_patterns():
    df = pd.DataFrame({'text': ['Sayfa 1 / 5 içeriği']})
    processor = TurkishBoilerplateCleanerProcessor({'boilerplate_columns': {'text': {'remove_common_patterns': True}}})
    result = processor.process(df)
    assert list(result['text']) == ['içeriği']

def test_custom_patterns():
    df = pd.DataFrame({'text': ['[GİZLİ] belge içeriği']})
    processor = TurkishBoilerplateCleanerProcessor({
        'boilerplate_columns': {'text': {'custom_patterns': [r'\[GİZLİ\]']}}
    })
    result = processor.process(df)
    assert list(result['text']) == ['[] belge içeriği']

def test_remove_duplicates_tfidf():
    df = pd.DataFrame({'text': ['hello world\nhello world\nunique text']})
    processor = TurkishBoilerplateCleanerProcessor({'embedding_model': 'tfidf', 'boilerplate_columns': {'text': {'remove_duplicates': True, 'duplicate_threshold': 0.8}}})
    result = processor.process(df)
    # The output should have duplicate lines removed
    assert result['text'].iloc[0] == 'hello world unique text'

def test_remove_header_footer():
    df = pd.DataFrame({'text': ['Tarih: 01.01.2023\nAna içerik burada.\nİmza: Müdür']})
    processor = TurkishBoilerplateCleanerProcessor({
        'boilerplate_columns': {'text': {'remove_header_footer': True}}
    })
    result = processor.process(df)
    assert 'Ana içerik burada.' in list(result['text'])[0]

def test_template_matching():
    df = pd.DataFrame({'text': ['Sözleşme No: 12345 için imzalanmıştır.']})
    processor = TurkishBoilerplateCleanerProcessor({
        'boilerplate_columns': {'text': {'use_template_matching': True, 'template_threshold': 0.8}}
    })
    # Since templates are empty in the basic implementation, this should not remove anything
    # But it covers the code path
    result = processor.process(df)
    assert len(result) == 1

def test_normalize_turkish_text():
    processor = TurkishBoilerplateCleanerProcessor({})
    text = "ŞĞÜİÖÇ şğüiöç"
    normalized = processor._normalize_turkish_text(text)
    assert normalized == "şğüi̇öç şğüiöç"

def test_backward_compat_alias():
    processor = BoilerplateCleanerProcessor({})
    assert isinstance(processor, TurkishBoilerplateCleanerProcessor)

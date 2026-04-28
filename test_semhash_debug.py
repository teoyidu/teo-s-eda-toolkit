import pandas as pd
from unittest.mock import patch, MagicMock

with patch('src.data_quality.processors.semhash_turkish_detector.SEMHASH_AVAILABLE', True):
    from src.data_quality.processors.semhash_turkish_detector import SemHashTurkishDetector

    df = pd.DataFrame({'text': ['A text to check', 'A similar text to check']})
    processor = SemHashTurkishDetector({'text_column': 'text', 'action': 'semhash'})
    
    mock_result = MagicMock()
    mock_result.selected = [{'text': 'A text to check'}]
    
    # We must patch SemHash completely
    with patch('src.data_quality.processors.semhash_turkish_detector.SemHash') as mock_semhash:
        mock_instance = mock_semhash.from_records.return_value
        mock_instance.self_deduplicate.return_value = mock_result
        
        result = processor._process_with_semhash(df)
        print("Result columns:", result.columns)
        print("is_duplicate values:", list(result['is_duplicate']))

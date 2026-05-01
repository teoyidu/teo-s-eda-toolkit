import pandas as pd
from unittest.mock import patch, MagicMock

with patch('src.data_quality.processors.semhash_turkish_detector.SEMHASH_AVAILABLE', True):
    from src.data_quality.processors.semhash_turkish_detector import SemHashTurkishDetector

    # We will subclass it and override _preprocess_turkish_text to see what goes in
    class MySemHashTurkishDetector(SemHashTurkishDetector):
        pass

    df = pd.DataFrame({'text': ['A text to check', 'A similar text to check']})
    processor = MySemHashTurkishDetector({'text_column': 'text', 'action': 'semhash'})
    
    mock_result = MagicMock()
    mock_result.selected = [{'text': 'A text to check'}]
    
    # We must patch SemHash completely
    with patch('src.data_quality.processors.semhash_turkish_detector.SemHash') as mock_semhash:
        mock_instance = mock_semhash.from_records.return_value
        mock_instance.self_deduplicate.return_value = mock_result
        
        # Monkey patch process_with_semhash to add print
        original_process = processor._process_with_semhash
        def _process_with_semhash(df):
            print("Running _process_with_semhash")
            return original_process(df)
            
        processor._process_with_semhash = _process_with_semhash
        
        result = processor.process(df)
        print("is_duplicate:", list(result['is_duplicate']))

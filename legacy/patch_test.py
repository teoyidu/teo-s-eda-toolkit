from unittest.mock import patch, MagicMock

@patch('src.data_quality.processors.semhash_turkish_detector.SemHash')
def run(mock_semhash):
    mock_instance = mock_semhash.from_records.return_value
    mock_result = MagicMock()
    mock_result.selected = [{'text': 'A text to check'}]
    mock_instance.self_deduplicate.return_value = mock_result
    
    from src.data_quality.processors.semhash_turkish_detector import SemHashTurkishDetector
    import pandas as pd
    
    df = pd.DataFrame({'text': ['A text to check', 'A similar text to check']})
    processor = SemHashTurkishDetector({'text_column': 'text', 'action': 'semhash'})
    result = processor._process_with_semhash(df)
    print("is_duplicate values:", list(result['is_duplicate']))

run()

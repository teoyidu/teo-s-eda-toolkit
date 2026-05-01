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
    
    original_preprocess = processor._preprocess_turkish_text
    
    # Let's see what result.selected is!
    # I'll monkey patch process_with_semhash
    records = df.to_dict(orient='records')
    semhash = mock_semhash.from_records(
        records=records,
        columns=[processor.text_column],
        use_ann=processor.use_ann
    )
    result = semhash.self_deduplicate()
    print("result.selected:", result.selected)

run()

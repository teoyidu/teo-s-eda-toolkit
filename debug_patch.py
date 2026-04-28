import pandas as pd
from unittest.mock import patch, MagicMock

with patch('src.data_quality.processors.semhash_turkish_detector.SEMHASH_AVAILABLE', True):
    from src.data_quality.processors.semhash_turkish_detector import SemHashTurkishDetector

    df = pd.DataFrame({'text': ['A text to check', 'A similar text to check']})
    processor = SemHashTurkishDetector({'text_column': 'text', 'action': 'semhash'})
    
    mock_result = MagicMock()
    mock_result.selected = [{'text': 'A text to check'}]
    
    with patch('src.data_quality.processors.semhash_turkish_detector.SemHash') as mock_semhash:
        mock_instance = mock_semhash.from_records.return_value
        mock_instance.self_deduplicate.return_value = mock_result
        
        records = df.to_dict(orient='records')
        print("records:", records)
        
        selected_records = mock_result.selected
        print("selected_records:", selected_records)
        
        selected_texts = []
        for record in selected_records:
            text = record.get(processor.text_column, "")
            selected_texts.append(processor._preprocess_turkish_text(text))
        print("selected_texts:", selected_texts)
        
        for idx, row in df.iterrows():
            original_text = row[processor.text_column]
            preprocessed_text = processor._preprocess_turkish_text(original_text)
            print(f"row {idx}: original='{original_text}', preprocessed='{preprocessed_text}', in_selected={preprocessed_text in selected_texts}")

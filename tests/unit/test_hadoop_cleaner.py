import pandas as pd
from src.data_quality.processors.hadoop_cleaner import HadoopCleanerProcessor

def test_no_columns_configured(sample_df):
    processor = HadoopCleanerProcessor({})
    result = processor.process(sample_df)
    assert len(result) == 4

def test_extracts_value_tags():
    df = pd.DataFrame({'log': ['<metadata>junk</metadata><value>real data</value>']})
    processor = HadoopCleanerProcessor({'hadoop_columns': {'log': {}}})
    result = processor.process(df)
    assert list(result['log']) == ['real data']

def test_falls_back_to_xml_tag_strip_when_no_value_tags():
    df = pd.DataFrame({'log': ['<metadata>real data</metadata>']})
    processor = HadoopCleanerProcessor({'hadoop_columns': {'log': {}}})
    result = processor.process(df)
    assert list(result['log']) == ['real data']

def test_removes_hadoop_metadata_patterns():
    df = pd.DataFrame({'log': ['job_123_456 task_1_2_3 log message']})
    processor = HadoopCleanerProcessor({'hadoop_columns': {'log': {'remove_metadata': True}}})
    result = processor.process(df)
    assert list(result['log']) == ['log message']

def test_custom_patterns():
    df = pd.DataFrame({'log': ['[DEBUG] actual message']})
    processor = HadoopCleanerProcessor({
        'hadoop_columns': {'log': {'custom_patterns': [r'\[DEBUG\]\s*']}}
    })
    result = processor.process(df)
    assert list(result['log']) == ['actual message']

def test_preserves_json_structured_data():
    df = pd.DataFrame({'log': ['{"key": "value", "ignore": "this"}']})
    processor = HadoopCleanerProcessor({
        'hadoop_columns': {'log': {
            'preserve_structured_data': True,
            'preserve_fields': ['key']
        }}
    })
    result = processor.process(df)
    assert list(result['log']) == ['{"key": "value"}']

def test_extract_metadata_creates_new_columns():
    df = pd.DataFrame({'log': ['application_123_456 stage_1 message']})
    processor = HadoopCleanerProcessor({
        'hadoop_columns': {'log': {
            'extract_metadata': True,
            'metadata_fields': ['application_id', 'stage_id']
        }}
    })
    result = processor.process(df)
    assert 'log_application_id' in result.columns
    assert 'log_stage_id' in result.columns
    assert list(result['log_application_id']) == ['application_123_456']
    assert list(result['log_stage_id']) == ['stage_1']

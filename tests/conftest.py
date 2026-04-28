import pytest
from pyspark.sql import SparkSession
import pandas as pd

@pytest.fixture(scope="session")
def spark():
    """Create a shared SparkSession for testing."""
    spark = (
        SparkSession.builder.appName("teo-eda-toolkit-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield spark
    spark.stop()

@pytest.fixture
def sample_df():
    """Create a basic pandas DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4],
        'text': ['hello', 'world', 'hello', 'test'],
        'value': [10.5, 20.0, None, 15.5]
    })

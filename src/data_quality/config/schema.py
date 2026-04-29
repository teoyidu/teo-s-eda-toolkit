from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class IrrelevantDataCondition(BaseModel):
    column: str
    values: List[str]

class ReferenceData(BaseModel):
    path: str
    key_column: str
    df_column: str

class UniqueConstraint(BaseModel):
    columns: List[str]
    action: Literal["drop_duplicates", "keep_first"] = "drop_duplicates"
    order_by: Optional[str] = None

class TextStandardization(BaseModel):
    trim: Optional[bool] = None
    case: Optional[Literal["upper", "lower", "title"]] = None

class TextValidationRule(BaseModel):
    pattern: Optional[str] = None
    min_length: Optional[int] = Field(None, ge=0)
    max_length: Optional[int] = Field(None, ge=0)
    remove_special_chars: Optional[bool] = None

class RelationshipRule(BaseModel):
    type: Literal["foreign_key", "conditional"]
    child_column: Optional[str] = None
    parent_table_path: Optional[str] = None
    parent_column: Optional[str] = None
    condition_column: Optional[str] = None
    condition_value: Optional[bool] = None
    dependent_column: Optional[str] = None
    dependent_required: Optional[bool] = None

class EntryRule(BaseModel):
    name: str
    type: Literal["range_check", "allowed_values", "cross_field_validation"]
    column: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[str]] = None
    field1: Optional[str] = None
    field2: Optional[str] = None
    operator: Optional[Literal[">", "<", ">=", "<=", "==", "!="]] = None

class DataQualityConfig(BaseSettings):
    """
    Main configuration schema for the Data Quality Framework using Pydantic.
    Supports environment variables (e.g. DQ_BATCH_SIZE=500000).
    """
    checkpoint_dir: str = "/tmp/dq_checkpoints"
    output_dir: str = "/tmp/dq_output"
    batch_size: int = Field(1000000, ge=1)
    
    missing_value_strategy: Literal["drop", "fill"] = "drop"
    missing_threshold: float = Field(50.0, ge=0, le=100)
    critical_columns: List[str] = Field(default_factory=list)
    fill_values: Dict[str, Any] = Field(default_factory=dict)
    
    mandatory_fields: List[str] = Field(default_factory=list)
    numerical_columns: List[str] = Field(default_factory=list)
    decimal_places: int = Field(2, ge=0)
    date_columns: List[str] = Field(default_factory=list)
    data_retention_days: int = Field(365, ge=1)
    
    irrelevant_data_conditions: List[IrrelevantDataCondition] = Field(default_factory=list)
    reference_data: List[ReferenceData] = Field(default_factory=list)
    unique_constraints: List[UniqueConstraint] = Field(default_factory=list)
    
    category_mappings: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    text_standardization: Dict[str, TextStandardization] = Field(default_factory=dict)
    text_validation_rules: Dict[str, TextValidationRule] = Field(default_factory=dict)
    
    relationship_rules: List[RelationshipRule] = Field(default_factory=list)
    entry_rules: List[EntryRule] = Field(default_factory=list)
    
    # Extra settings
    legal_domain_filtering: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = SettingsConfigDict(
        env_prefix="DQ_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )

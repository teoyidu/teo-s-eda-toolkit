"""
Configuration validation utility for the Data Quality Framework
"""

from typing import Dict, List, Tuple, Any
from pydantic import ValidationError
from ..config.schema import DataQualityConfig

class ConfigurationValidator:
    """Validates configuration for the data quality framework using Pydantic"""
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration against Pydantic schema
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            # Pydantic will validate the dictionary against the schema
            DataQualityConfig(**config)
            return True, []
        except ValidationError as e:
            errors = []
            for err in e.errors():
                loc = ".".join(str(l) for l in err["loc"])
                errors.append(f"{loc}: {err['msg']}")
            return False, errors
        except Exception as e:
            return False, [f"Unexpected error during validation: {str(e)}"] 
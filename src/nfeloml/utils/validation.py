'''
Validation utilities for data quality checks and feature validation
'''
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class DataValidator:
    '''
    Validates data quality for NFL play-by-play data
    '''
    
    @staticmethod
    def validate_required_columns(data: pd.DataFrame, required_cols: List[str]) -> bool:
        '''
        Check if all required columns are present
        
        Params:
        * data: pd.DataFrame - data to validate
        * required_cols: List[str] - list of required column names
        
        Returns:
        * bool: True if all columns present, raises ValueError otherwise
        '''
        missing = set(required_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True
    
    @staticmethod
    def validate_range(
        data: pd.DataFrame, 
        column: str, 
        min_val: Optional[float] = None, 
        max_val: Optional[float] = None
    ) -> Dict[str, Any]:
        '''
        Validate that column values are within expected range
        
        Params:
        * data: pd.DataFrame - data to validate
        * column: str - column name to check
        * min_val: Optional[float] - minimum allowed value
        * max_val: Optional[float] - maximum allowed value
        
        Returns:
        * Dict[str, Any]: validation results with out_of_range count
        '''
        if column not in data.columns:
            raise ValueError(f"Column {column} not found in data")
        col_data = data[column].dropna()
        out_of_range = 0
        if min_val is not None:
            out_of_range += (col_data < min_val).sum()
        if max_val is not None:
            out_of_range += (col_data > max_val).sum()
        return {
            'column': column,
            'out_of_range': int(out_of_range),
            'total': len(col_data),
            'percent_valid': (1 - out_of_range / len(col_data)) * 100
        }
    
    @staticmethod
    def validate_no_nulls(data: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
        '''
        Check for null values in specified columns
        
        Params:
        * data: pd.DataFrame - data to validate
        * columns: List[str] - columns to check for nulls
        
        Returns:
        * Dict[str, int]: null counts per column
        '''
        null_counts = {}
        for col in columns:
            if col in data.columns:
                null_counts[col] = int(data[col].isna().sum())
        return null_counts
    
    @staticmethod
    def validate_ep_features(features: pd.DataFrame) -> bool:
        '''
        Validate Expected Points feature data
        
        Params:
        * features: pd.DataFrame - EP features to validate
        
        Returns:
        * bool: True if valid, raises ValueError otherwise
        '''
        ##  Check yardline is between 0 and 100
        if (features['yardline_100'] < 0).any() or (features['yardline_100'] > 100).any():
            raise ValueError("yardline_100 must be between 0 and 100")
        ##  Check down is between 1 and 4
        if 'down' in features.columns:
            if (features['down'] < 1).any() or (features['down'] > 4).any():
                raise ValueError("down must be between 1 and 4")
        ##  Check timeouts are between 0 and 3
        if (features['posteam_timeouts_remaining'] < 0).any() or \
           (features['posteam_timeouts_remaining'] > 3).any():
            raise ValueError("timeouts must be between 0 and 3")
        return True
    
    @staticmethod
    def validate_wp_features(features: pd.DataFrame) -> bool:
        '''
        Validate Win Probability feature data
        
        Params:
        * features: pd.DataFrame - WP features to validate
        
        Returns:
        * bool: True if valid, raises ValueError otherwise
        '''
        ##  Check game_seconds_remaining is between 0 and 3600
        if (features['game_seconds_remaining'] < 0).any() or \
           (features['game_seconds_remaining'] > 3600).any():
            raise ValueError("game_seconds_remaining must be between 0 and 3600")
        ##  Check half_seconds_remaining is between 0 and 1800
        if (features['half_seconds_remaining'] < 0).any() or \
           (features['half_seconds_remaining'] > 1800).any():
            raise ValueError("half_seconds_remaining must be between 0 and 1800")
        ##  Check down is between 1 and 4
        if 'down' in features.columns:
            if (features['down'] < 1).any() or (features['down'] > 4).any():
                raise ValueError("down must be between 1 and 4")
        return True
    
    @staticmethod
    def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
        '''
        Run comprehensive data quality checks
        
        Params:
        * data: pd.DataFrame - data to check
        
        Returns:
        * Dict[str, Any]: data quality report
        '''
        report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'null_counts': data.isna().sum().to_dict(),
            'duplicate_rows': int(data.duplicated().sum()),
        }
        return report


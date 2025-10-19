'''
Data loader for Expected Points model
'''
import pandas as pd
import numpy as np

try:
    import nfelodcm as dcm
except ImportError:
    dcm = None

from nfeloml.core.base_data_loader import BaseDataLoader
from .utils import create_next_score_labels, add_ep_features

class EPDataLoader(BaseDataLoader):
    '''
    Data loader for Expected Points model
    '''
    
    def _fetch_from_source(self) -> pd.DataFrame:
        '''
        Fetch play-by-play data from nfelodcm
        
        Returns:
        * pd.DataFrame: raw play-by-play data
        '''
        if dcm is None:
            raise ImportError(
                "nfelodcm is required for training but not installed. "
                "Install it with: pip install nfelodcm"
            )
        db = dcm.load(['pbp'])
        return db['pbp'].copy()
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Apply EP-specific transformations and filters
        
        Params:
        * data: pd.DataFrame - raw pbp data
        
        Returns:
        * pd.DataFrame: transformed data
        '''
        df = data.copy()
        ##  Filter to relevant plays only
        df = df[
            df['defteam_timeouts_remaining'].notna() &
            df['posteam_timeouts_remaining'].notna() &
            df['yardline_100'].notna()
        ]
        ##  Create next score labels using vectorized helper function
        df = create_next_score_labels(df)
        ##  Create weighting scheme
        df = self._create_weights(df)
        return df
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Create EP-specific features
        
        Params:
        * data: pd.DataFrame - transformed data
        
        Returns:
        * pd.DataFrame: data with all features
        '''
        ##  Use shared utility function for feature engineering
        return add_ep_features(data)
    
    def _create_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Create sample weights using nflscrapR weighting scheme
        
        Weights prioritize plays closer to scoring events and in closer games.
        XGBoost requires positive weights, so we scale to (0, 1] range.
        
        Params:
        * data: pd.DataFrame - play data
        
        Returns:
        * pd.DataFrame: data with Total_W_Scaled weights
        '''
        df = data.copy()
        ##  Calculate distance to next scoring drive (will be NaN for plays with no next score)
        df['Drive_Score_Dist'] = df['Drive_Score_Half'] - df['fixed_drive']
        ##  Inverse normalize - plays closer to scoring get higher weight
        max_dist = df['Drive_Score_Dist'].max()
        min_dist = df['Drive_Score_Dist'].min()
        if pd.notna(max_dist) and pd.notna(min_dist) and max_dist != min_dist:
            df['Drive_Score_Dist_W'] = (max_dist - df['Drive_Score_Dist']) / (max_dist - min_dist)
        else:
            df['Drive_Score_Dist_W'] = 1.0
        ##  Inverse normalize score differential - closer games get higher weight
        max_abs_diff = df['score_differential'].abs().max()
        min_abs_diff = df['score_differential'].abs().min()
        if pd.notna(max_abs_diff) and pd.notna(min_abs_diff) and max_abs_diff != min_abs_diff:
            df['ScoreDiff_W'] = (max_abs_diff - df['score_differential'].abs()) / (max_abs_diff - min_abs_diff)
        else:
            df['ScoreDiff_W'] = 1.0
        ##  Combine weights
        df['Total_W'] = df['Drive_Score_Dist_W'] + df['ScoreDiff_W']
        ##  Scale to [0, 1] range (matching nflfastR exactly)
        max_w = df['Total_W'].max()
        min_w = df['Total_W'].min()
        if pd.notna(max_w) and pd.notna(min_w) and max_w != min_w:
            df['Total_W_Scaled'] = (df['Total_W'] - min_w) / (max_w - min_w)
        else:
            df['Total_W_Scaled'] = 1.0
        ##  Fill any remaining NaN weights with 0 (XGBoost can handle zero weights)
        df['Total_W_Scaled'] = df['Total_W_Scaled'].fillna(0.0)
        ##  Drop intermediate columns
        df = df.drop(columns=['Drive_Score_Dist', 'Drive_Score_Dist_W', 'ScoreDiff_W', 'Total_W'])
        return df

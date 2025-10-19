'''
Utility functions for calculating Expected Points Added (EPA)
'''

import pandas as pd
import numpy as np
from typing import Optional


def calculate_epa(df: pd.DataFrame, ep_column: str = 'expected_points') -> pd.DataFrame:
    '''
    Calculate Expected Points Added (EPA) for each play in a dataframe.
    
    EPA measures the change in expected points from the start to the end of a play,
    accounting for scoring plays and changes of possession.
    
    Logic:
    - For non-scoring plays: EPA = EP_end - EP_start
    - For scoring plays: EP_end = points scored (TD=7, FG=3, Safety=-2 for offense)
    - When possession changes: EP_end is negated (opponent's EP from their perspective)
    - Plays without valid EP (missing required features) result in null EPA
    
    Params:
    * df: pd.DataFrame - play-by-play data with EP predictions already added
    * ep_column: str - name of the column containing expected points values
    
    Returns:
    * pd.DataFrame: dataframe with 'epa' column added
    '''
    result = df.copy().sort_values(['game_id', 'play_id'])
    
    ##  First, identify which plays have valid EP (same criteria as model training)
    ##  A play has valid EP if it has non-null values for model features
    valid_ep_mask = (
        result[ep_column].notna() &
        result['posteam'].notna() &
        result['yardline_100'].notna() &
        result['down'].notna() &
        result['ydstogo'].notna() &
        result['half_seconds_remaining'].notna() &
        result['posteam_timeouts_remaining'].notna() &
        result['defteam_timeouts_remaining'].notna()
    )
    
    ##  Initialize EPA column
    result['epa'] = np.nan
    
    ##  Mark valid EP plays and use bfill to find next valid EP
    ##  Create a column that's only populated for valid EP plays
    result['_valid_ep'] = np.where(valid_ep_mask, result[ep_column], np.nan)
    result['_valid_posteam'] = np.where(valid_ep_mask, result['posteam'], np.nan)
    
    ##  Use shift to get "lead" (next row) values within each game
    ##  But we need the next VALID EP, so we'll use a different approach
    ##  Create cumulative count of valid plays to use as a key
    result['_valid_idx'] = valid_ep_mask.cumsum()
    
    ##  For each game, find the next valid EP play
    def find_next_valid(group):
        ##  For each play, find the next row with valid EP
        next_ep_vals = []
        next_posteam_vals = []
        
        valid_indices = group[valid_ep_mask.loc[group.index]].index
        
        for idx in group.index:
            ##  Find next valid index after this one
            future_valid = valid_indices[valid_indices > idx]
            if len(future_valid) > 0:
                next_idx = future_valid[0]
                next_ep_vals.append(group.loc[next_idx, ep_column])
                next_posteam_vals.append(group.loc[next_idx, 'posteam'])
            else:
                next_ep_vals.append(np.nan)
                next_posteam_vals.append(None)
        
        group['_next_ep'] = next_ep_vals
        group['_next_posteam'] = next_posteam_vals
        return group
    
    result = result.groupby('game_id', group_keys=False).apply(find_next_valid)
    
    ##  Now calculate EP_end for each play
    ##  Start with next_ep as the default
    result['ep_end'] = result['_next_ep']
    
    ##  Handle scoring plays
    ##  Touchdown (including XP) = 7 points
    result.loc[
        (result['touchdown'] == 1) & 
        (result['td_team'] == result['posteam']),
        'ep_end'
    ] = 7.0
    
    ##  Field Goal = 3 points
    result.loc[
        (result['field_goal_result'] == 'made'),
        'ep_end'
    ] = 3.0
    
    ##  Safety = -2 points for offense (2 points for defense)
    result.loc[
        (result['safety'] == 1),
        'ep_end'
    ] = -2.0
    
    ##  Opponent scores (defensive/special teams TD, or turnover leading to score)
    ##  For opponent TD, it's -7 from offense perspective
    result.loc[
        (result['touchdown'] == 1) & 
        (result['td_team'] != result['posteam']) &
        (result['td_team'].notna()),
        'ep_end'
    ] = -7.0
    
    ##  Handle possession changes (when next_posteam differs from current posteam)
    ##  In these cases, negate the EP because it's from opponent's perspective
    possession_change = (
        (result['_next_posteam'] != result['posteam']) &
        (result['_next_posteam'].notna()) &
        (result['posteam'].notna()) &
        ##  But NOT on scoring plays (already handled above)
        ~((result['touchdown'] == 1) | 
          (result['field_goal_result'] == 'made') | 
          (result['safety'] == 1))
    )
    
    result.loc[possession_change, 'ep_end'] = -result.loc[possession_change, 'ep_end']
    
    ##  Finally, calculate EPA = EP_end - EP_start
    ##  Only calculate for plays that have valid starting EP
    result.loc[valid_ep_mask, 'epa'] = result.loc[valid_ep_mask, 'ep_end'] - result.loc[valid_ep_mask, ep_column]
    
    ##  Clean up temporary columns
    temp_cols = ['_valid_ep', '_valid_posteam', '_valid_idx', '_next_ep', '_next_posteam', 'ep_end']
    result = result.drop(columns=[c for c in temp_cols if c in result.columns])
    
    return result


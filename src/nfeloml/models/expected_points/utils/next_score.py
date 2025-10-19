'''
Helper functions for creating next score labels
'''

import pandas as pd


def create_next_score_labels(df: pd.DataFrame) -> pd.DataFrame:
    '''
    For each play, find the next scoring event in the same game half
    
    Excludes extra points and 2-point conversions - only counts independent scoring plays.
    Uses vectorized transform for efficiency.
    
    Params:
    * df: pd.DataFrame - play-by-play data with required scoring columns
    
    Returns:
    * pd.DataFrame: dataframe with Next_Score_Half and Drive_Score_Half columns added
    '''
    df = df.sort_values(['game_id', 'play_id']).copy()
    
    ##  Identify plays that are NOT extra points or 2-point conversions
    not_extra_point = (df['extra_point_attempt'] != 1) | (df['extra_point_attempt'].isna())
    not_two_point = (df['two_point_attempt'] != 1) | (df['two_point_attempt'].isna())
    independent_play = not_extra_point & not_two_point
    
    ##  Initialize scoring columns
    df['score_type'] = None
    df['scoring_team_for_play'] = None
    
    ##  Mark touchdown scoring plays (excluding PATs and 2-pt conversions)
    mask_td = (df['touchdown'] == 1) & independent_play
    df.loc[mask_td, 'score_type'] = 'Touchdown'
    df.loc[mask_td, 'scoring_team_for_play'] = df.loc[mask_td, 'td_team']
    
    ##  Mark field goal scoring plays
    mask_fg = (df['field_goal_result'] == 'made') & independent_play
    df.loc[mask_fg, 'score_type'] = 'Field_Goal'
    df.loc[mask_fg, 'scoring_team_for_play'] = df.loc[mask_fg, 'posteam']
    
    ##  Mark safety scoring plays (excluding 2-pt conversion safeties)
    mask_safety = (df['safety'] == 1) & independent_play
    df.loc[mask_safety, 'score_type'] = 'Safety'
    df.loc[mask_safety, 'scoring_team_for_play'] = df.loc[mask_safety, 'defteam']
    
    ##  Use transform with bfill to propagate next score info backward within each game half
    ##  This is MUCH faster than apply()!
    df['next_score_type'] = df.groupby(['game_id', 'game_half'])['score_type'].transform('bfill')
    df['next_scoring_team'] = df.groupby(['game_id', 'game_half'])['scoring_team_for_play'].transform('bfill')
    
    ##  For Drive_Score_Half, capture the drive number where the next score occurred
    ##  Create a temporary column with drive numbers only where there's a score
    df['score_drives'] = df['fixed_drive'].where(df['score_type'].notna())
    df['Drive_Score_Half'] = df.groupby(['game_id', 'game_half'])['score_drives'].transform('bfill')
    
    ##  Create Next_Score_Half labels based on scoring team
    df['Next_Score_Half'] = 'No_Score'
    mask_has_next = df['next_score_type'].notna()
    
    ##  Own team scores
    own_scores = mask_has_next & (df['next_scoring_team'] == df['posteam'])
    df.loc[own_scores, 'Next_Score_Half'] = df.loc[own_scores, 'next_score_type']
    
    ##  Opponent scores
    opp_scores = mask_has_next & (df['next_scoring_team'] != df['posteam'])
    df.loc[opp_scores, 'Next_Score_Half'] = 'Opp_' + df.loc[opp_scores, 'next_score_type']
    
    ##  Cleanup temporary columns
    df = df.drop(columns=['score_type', 'scoring_team_for_play', 'next_score_type', 
                          'next_scoring_team', 'score_drives'])
    
    return df


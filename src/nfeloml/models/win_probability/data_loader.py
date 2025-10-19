'''
Data loader for Win Probability model
'''
import pandas as pd
import numpy as np

try:
    import nfelodcm as dcm
except ImportError:
    dcm = None

from nfeloml.core.base_data_loader import BaseDataLoader

class WPDataLoader(BaseDataLoader):
    '''
    Data loader for Win Probability model
    '''
    
    def _fetch_from_source(self) -> pd.DataFrame:
        '''
        Fetch play-by-play data from nfelodcm
        
        Returns:
        * pd.DataFrame: raw play-by-play data with winner and spreads
        '''
        if dcm is None:
            raise ImportError(
                "nfelodcm is required for training but not installed. "
                "Install it with: pip install nfelodcm"
            )
        db = dcm.load(['pbp', 'games'])
        pbp = db['pbp'].copy()
        games = db['games'].copy()
        ## get winner for each game ##
        games['winner'] = np.where(
            games['home_score'] > games['away_score'],
            games['home_team'],
            np.where(
                games['home_score'] < games['away_score'],
                games['away_team'],
                np.nan
            )
        )
        ## Keep spread_line as-is (positive = home favored, negative = away favored)
        ## Drop spread_line from pbp if it exists to avoid merge conflicts
        if 'spread_line' in pbp.columns:
            pbp = pbp.drop(columns=['spread_line'])
        pbp = pd.merge(
            pbp,
            games[['game_id', 'winner', 'spread_line']],
            on=['game_id'],
            how='left'
        )
        pbp = pbp[pbp['winner'].notna()].copy().reset_index(drop=True)
        return pbp
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Apply WP-specific transformations and filters
        
        Params:
        * data: pd.DataFrame - raw pbp data
        
        Returns:
        * pd.DataFrame: transformed data
        '''
        df = data.copy()
        ##  Filter to regulation time only
        df = df[df['qtr'] <= 4]
        ##  Filter to valid plays
        df = df[
            df['defteam_timeouts_remaining'].notna() &
            df['posteam_timeouts_remaining'].notna() &
            df['yardline_100'].notna() &
            df['score_differential'].notna()
        ]
        ##  Create winner label using lowercase 'winner'
        if 'winner' in df.columns:
            df['label'] = (df['posteam'] == df['winner']).astype(int)
        else:
            df['label'] = 0
        return df
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Create WP-specific features
        
        Params:
        * data: pd.DataFrame - transformed data
        
        Returns:
        * pd.DataFrame: data with all features
        '''
        df = data.copy()
        ##  Calculate Diff_Time_Ratio
        df['Diff_Time_Ratio'] = df['score_differential'] * np.exp(
            4 * (3600 - df['game_seconds_remaining']) / 3600
        )
        ##  Create home indicator
        df['home'] = (df['posteam'] == df['home_team']).astype(int)
        ##  Calculate posteam_spread using raw spread_line
        ##  spread_line: positive when home favored, negative when away favored
        ##  posteam_spread: positive when posteam favored, negative when underdog
        ##  Always create spread features (fill with 0 if spread_line missing)
        if 'spread_line' in df.columns:
            df['posteam_spread'] = np.where(
                df['posteam'] == df['home_team'],
                df['spread_line'],    # home team gets spread_line as-is
                -df['spread_line']    # away team gets opposite sign
            )
        else:
            ##  If no spread_line, default to 0
            df['posteam_spread'] = 0.0
        ##  Always calculate spread_time (required for spread model consistency)
        df['spread_time'] = df['posteam_spread'] * np.exp(
            -4 * (3600 - df['game_seconds_remaining']) / 3600
        )
        ##  Create receive_2h_ko properly
        ##  Team that didn't receive opening kickoff receives 2nd half kickoff
        if 'home_opening_kickoff' in df.columns:
            df['receive_2h_ko'] = np.where(
                (df['home_opening_kickoff'] == 1) & (df['posteam'] == df['home_team']),
                0,  # home received opening, so won't receive 2h
                np.where(
                    (df['home_opening_kickoff'] == 1) & (df['posteam'] == df['away_team']),
                    1,  # away will receive 2h
                    np.where(
                        (df['home_opening_kickoff'] == 0) & (df['posteam'] == df['home_team']),
                        1,  # home will receive 2h
                        0   # away won't receive 2h
                    )
                )
            ).astype(int)
        else:
            df['receive_2h_ko'] = 0
        return df

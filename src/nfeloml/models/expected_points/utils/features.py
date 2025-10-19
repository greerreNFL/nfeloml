'''
Feature engineering utilities for Expected Points model
'''

import pandas as pd


def add_ep_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Add all engineered features needed for EP model predictions
    
    Creates era indicators, down indicators, roof indicators, and home indicator
    from raw play-by-play data. Only creates features that don't already exist.
    
    Params:
    * df: pd.DataFrame - play-by-play data with season, down, roof, posteam, home_team columns
    
    Returns:
    * pd.DataFrame: dataframe with added feature columns
    '''
    result = df.copy()
    
    ##  Create era indicators from season if needed
    if 'era0' not in result.columns:
        if 'season' in result.columns:
            result['era0'] = ((result['season'] >= 1999) & (result['season'] <= 2001)).astype(int)
            result['era1'] = ((result['season'] >= 2002) & (result['season'] <= 2005)).astype(int)
            result['era2'] = ((result['season'] >= 2006) & (result['season'] <= 2013)).astype(int)
            result['era3'] = ((result['season'] >= 2014) & (result['season'] <= 2017)).astype(int)
            result['era4'] = (result['season'] >= 2018).astype(int)
        else:
            raise ValueError("'season' column required to create era indicators")
    
    ##  Create down indicators if needed
    if 'down1' not in result.columns:
        if 'down' in result.columns:
            result['down1'] = (result['down'] == 1).astype(int)
            result['down2'] = (result['down'] == 2).astype(int)
            result['down3'] = (result['down'] == 3).astype(int)
            result['down4'] = (result['down'] == 4).astype(int)
        else:
            raise ValueError("'down' column required to create down indicators")
    
    ##  Create roof indicators if needed
    if 'retractable' not in result.columns:
        if 'roof' in result.columns:
            result['retractable'] = (result['roof'] == 'retractable').astype(int)
            result['dome'] = (result['roof'] == 'dome').astype(int)
            result['outdoors'] = (result['roof'] == 'outdoors').astype(int)
        else:
            raise ValueError("'roof' column required to create roof indicators")
    
    ##  Create home indicator if needed
    if 'home' not in result.columns:
        if 'posteam' in result.columns and 'home_team' in result.columns:
            result['home'] = (result['posteam'] == result['home_team']).astype(int)
        else:
            raise ValueError("'posteam' and 'home_team' columns required to create home indicator")
    
    return result


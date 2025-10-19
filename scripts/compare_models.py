'''
Script to compare nfeloml Expected Points model with nflfastr EP model

This script:
1. Loads PBP data using nfelodcm
2. Filters to plays with existing EPA values
3. Renames nflfastr ep/epa to ep_original/epa_original
4. Runs nfeloml EP model predictions
5. Calculates correlations between models
6. Identifies plays with largest EP differences
7. Saves top 500 differences to CSV in scripts/output/
'''

import sys
from pathlib import Path

##  Add src directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import nfelodcm as dcm
import pandas as pd
import numpy as np
from nfeloml import ExpectedPointsModel

def main():
    print("Loading PBP data from nfelodcm...")
    db = dcm.load(['pbp'])
    pbp = db['pbp'].copy()
    print(f"Loaded {len(pbp):,} total plays")
    
    ##  Filter to plays with EPA (nflfastr model results)
    print("\nFiltering to plays with EPA values...")
    pbp_with_epa = pbp[pbp['epa'].notna()].copy()
    print(f"Filtered to {len(pbp_with_epa):,} plays with EPA")
    
    ##  Rename original columns
    print("\nRenaming ep and epa to ep_original and epa_original...")
    pbp_with_epa = pbp_with_epa.rename(columns={
        'ep': 'ep_original',
        'epa': 'epa_original'
    })
    
    ##  Load our EP model
    print("\nLoading nfeloml Expected Points model...")
    ep_model = ExpectedPointsModel()
    if ep_model.metadata:
        print(f"Model loaded: {ep_model.metadata.model_name}")
        print(f"Version: {ep_model.metadata.version}")
        print(f"Trained on seasons: {ep_model.metadata.training_seasons}")
    else:
        print("Model loaded successfully")
    
    ##  Run predictions on the dataset
    print("\nRunning EP model predictions...")
    print("This may take a few minutes for large datasets...")
    pbp_predicted = ep_model.predict_df(pbp_with_epa, include_epa=True)
    
    ##  Rename our predictions to distinguish them
    pbp_predicted = pbp_predicted.rename(columns={
        'expected_points': 'ep',
        'epa': 'epa'
    })
    
    print(f"Predictions complete for {len(pbp_predicted):,} plays")
    
    ##  Filter to plays where both models have valid EP
    valid_mask = pbp_predicted['ep'].notna() & pbp_predicted['ep_original'].notna()
    pbp_valid = pbp_predicted[valid_mask].copy()
    print(f"\n{len(pbp_valid):,} plays have valid EP from both models")
    
    ##  Calculate correlations
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    ep_corr = pbp_valid['ep'].corr(pbp_valid['ep_original'])
    print(f"\nExpected Points (EP) Correlation: {ep_corr:.6f}")
    
    ##  EPA correlation (only for plays with valid EPA from both)
    epa_valid_mask = pbp_valid['epa'].notna() & pbp_valid['epa_original'].notna()
    epa_corr = pbp_valid.loc[epa_valid_mask, 'epa'].corr(
        pbp_valid.loc[epa_valid_mask, 'epa_original']
    )
    print(f"Expected Points Added (EPA) Correlation: {epa_corr:.6f}")
    
    ##  Calculate EP differences
    print("\n" + "="*60)
    print("DIFFERENCE ANALYSIS")
    print("="*60)
    
    pbp_valid['ep_diff'] = abs(pbp_valid['ep'] - pbp_valid['ep_original'])
    
    print(f"\nMean Absolute EP Difference: {pbp_valid['ep_diff'].mean():.4f}")
    print(f"Median Absolute EP Difference: {pbp_valid['ep_diff'].median():.4f}")
    print(f"Max Absolute EP Difference: {pbp_valid['ep_diff'].max():.4f}")
    
    ##  Get top 500 plays with largest differences
    print("\nFinding top 500 plays with largest EP differences...")
    top_diffs = pbp_valid.nlargest(500, 'ep_diff')
    
    ##  Select columns for output
    ##  Include: desc, all feature columns, both EP values, and the difference
    feature_cols = [
        'desc',
        ##  Raw feature inputs
        'season', 'down', 'ydstogo', 'yardline_100',
        'half_seconds_remaining', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
        'posteam', 'home_team', 'roof',
        ##  Derived features
        'home', 'retractable', 'dome', 'outdoors',
        'era0', 'era1', 'era2', 'era3', 'era4',
        'down1', 'down2', 'down3', 'down4',
        ##  Model outputs
        'ep_original', 'ep', 'ep_diff',
        'epa_original', 'epa',
        ##  Context
        'game_id', 'play_id', 'game_date', 'week',
        'posteam', 'defteam', 'posteam_score', 'defteam_score',
        'qtr', 'time', 'play_type'
    ]
    
    ##  Only include columns that exist in the dataframe
    output_cols = [col for col in feature_cols if col in top_diffs.columns]
    output_df = top_diffs[output_cols]
    
    ##  Save to CSV in scripts/output/
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'ep_model_comparison_top_500_diffs.csv'
    print(f"\nSaving top 500 differences to {output_file}...")
    output_df.to_csv(output_file, index=False)
    print(f"Saved {len(output_df)} records")
    
    ##  Print some sample differences
    print("\n" + "="*60)
    print("SAMPLE OF TOP DIFFERENCES")
    print("="*60)
    
    print("\nTop 10 plays with largest EP differences:")
    for idx, row in top_diffs.head(10).iterrows():
        print(f"\n{row['desc'][:100]}...")
        print(f"  EP (nfeloml): {row['ep']:.3f}")
        print(f"  EP (nflfastr): {row['ep_original']:.3f}")
        print(f"  Difference: {row['ep_diff']:.3f}")
        print(f"  Down: {row['down']}, Yards to go: {row['ydstogo']}, Yardline: {row['yardline_100']}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTotal plays analyzed: {len(pbp_valid):,}")
    print(f"EP Correlation: {ep_corr:.6f}")
    print(f"EPA Correlation: {epa_corr:.6f}")
    print(f"Mean Absolute Difference: {pbp_valid['ep_diff'].mean():.4f}")
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()


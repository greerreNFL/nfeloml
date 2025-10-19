'''
Full training script for Win Probability model with spread
Trains on all available seasons with spread data (2006+)
'''
import sys
from pathlib import Path
from datetime import datetime
##  Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from nfeloml.models.win_probability.data_loader import WPDataLoader
from nfeloml.models.win_probability.trainer import WPTrainer
from nfeloml.models.win_probability.types import WPSpreadTrainingConfig
from nfeloml.core.types import ModelMetadata

def train_full_model():
    '''Train WP model with spread on all available seasons'''
    print("="*80)
    print("TRAINING WIN PROBABILITY MODEL (WITH SPREAD) - FULL DATASET")
    print("="*80)
    
    ##  Load data first to determine available seasons
    print("\n[1/5] Loading data to determine available seasons...")
    loader = WPDataLoader()
    data = loader.load_data()
    available_seasons = sorted(data['season'].unique())
    print(f"[OK] Found {len(available_seasons)} seasons: {available_seasons[0]}-{available_seasons[-1]}")
    print(f"[OK] Total plays: {len(data):,}")
    
    ##  Filter to seasons with spread data (typically 2006+)
    ##  Check if spread_line is available
    has_spread = data['spread_line'].notna().sum()
    print(f"[OK] Plays with spread data: {has_spread:,} ({has_spread/len(data)*100:.1f}%)")
    
    ##  Create training config with all seasons
    print("\n[2/5] Creating training configuration...")
    config = WPSpreadTrainingConfig(
        seasons=available_seasons,
        validation_strategy='loso',
        random_seed=2013,
        verbose=True
    )
    print(f"[OK] Config: {len(config.seasons)} seasons, LOSO cross-validation")
    print(f"[OK] Model uses spread: {config.use_spread}")
    print(f"[OK] XGBoost params: eta={config.eta}, nrounds={config.nrounds}, max_depth={config.max_depth}")
    print(f"[OK] Monotone constraints enabled: {config.gamma}")
    
    ##  Initialize trainer
    print("\n[3/5] Initializing trainer...")
    trainer = WPTrainer(config, loader)
    print("[OK] Trainer ready")
    
    ##  Train model with LOSO CV
    print(f"\n[4/5] Training model (this will take a while - {len(available_seasons)} LOSO folds + final model)...")
    print("This may take 30-60 minutes depending on your machine...\n")
    model = trainer.train()
    print("\n[OK] Training completed!")
    
    ##  Evaluate
    print("\n[5/5] Evaluating model...")
    metrics = trainer.evaluate()
    print("[OK] Evaluation completed!")
    print(f"\n{'='*80}")
    print("MODEL PERFORMANCE:")
    print(f"{'='*80}")
    print(f"  {'Log Loss':20s}: {metrics['log_loss']:.6f}")
    print(f"  {'Accuracy':20s}: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  {'Calibration Error':20s}: {metrics['calibration_error']['overall']:.6f}")
    print(f"\n{'Calibration by Quarter:':20s}")
    for qtr, error in sorted(metrics['calibration_error']['by_quarter'].items()):
        print(f"    {qtr:18s}: {error:.6f}")
    print(f"{'='*80}")
    
    ##  Save model
    print("\nSaving production model...")
    model_dir = Path(__file__).parent.parent / 'src' / 'nfeloml' / 'models' / 'win_probability'
    save_path = model_dir / 'trained_model' / 'model.ubj'
    metadata = ModelMetadata(
        model_name='win_probability_spread',
        version='0.1.0',
        trained_date=datetime.now(),
        training_seasons=config.seasons,
        calibration_error=metrics['calibration_error']['overall'],
        additional_metrics={
            'log_loss': metrics['log_loss'],
            'accuracy': metrics['accuracy'],
            'calibration_by_quarter': metrics['calibration_error']['by_quarter']
        }
    )
    trainer.save_model(save_path, metadata)
    print(f"[OK] Model saved: {save_path}")
    print(f"[OK] Metadata saved: {save_path.parent / 'metadata.json'}")
    
    ##  Save training run
    print("\nSaving training run record...")
    training_runs_dir = model_dir / 'training_runs'
    run_path = trainer.save_training_run(
        training_runs_dir=training_runs_dir,
        metrics=metrics,
        additional_info={
            'note': 'Production model trained on all available seasons with spread data',
            'data_rows': len(data),
            'spread_available': has_spread,
            'training_duration': 'see timestamp for approximate duration'
        }
    )
    print(f"[OK] Training run saved: {run_path}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nModel ready for deployment:")
    print(f"  - Model file: {save_path}")
    print(f"  - Seasons: {available_seasons[0]}-{available_seasons[-1]} ({len(available_seasons)} total)")
    print(f"  - Plays: {len(data):,}")
    print(f"  - Overall calibration error: {metrics['calibration_error']['overall']:.6f}")
    print(f"  - Log loss: {metrics['log_loss']:.6f}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nTo use the model:")
    print(f"  from nfeloml import WinProbabilityModel")
    print(f"  model = WinProbabilityModel(use_spread=True)")
    print(f"  # Make predictions...")
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    try:
        train_full_model()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user. Partial results may be saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



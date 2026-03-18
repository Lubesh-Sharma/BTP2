import optuna
import copy
import yaml
import torch
from train_st_te import load_train_shapes, train_model
from test_st_te import load_test_shapes, test_model
from models.asmae import ASMAE

def load_base_config():
    with open('config/train_st_te_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def objective(trial, train_shapes, test_shapes):
    config = load_base_config()
    
    # 1. Propose Hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    ema_alpha = trial.suggest_float('ema_alpha', 0.9, 0.9999)
    student_mask_ratio = trial.suggest_float('student_mask_ratio', 0.4001, 0.8)
    student_feature_ratio = trial.suggest_float('student_feature_ratio', 0.1, 0.5)
    teacher_mask_ratio = trial.suggest_float('teacher_mask_ratio', 0.05, 0.4)
    teacher_feature_ratio = trial.suggest_float('teacher_feature_ratio', 0.01, 0.1)
    consistency_weight = trial.suggest_float('consistency_weight', 0.1, 5.0)
    contrastive_weight = trial.suggest_float('contrastive_weight', 0.1, 5.0)
    contrastive_temperature = trial.suggest_float('contrastive_temperature', 0.05, 1.0)
    
    # Override in config
    config['training']['lr'] = lr
    config['training']['ema_alpha'] = ema_alpha
    config['training']['student_mask_ratio'] = student_mask_ratio
    config['training']['student_feature_ratio'] = student_feature_ratio
    config['training']['teacher_mask_ratio'] = teacher_mask_ratio
    config['training']['teacher_feature_ratio'] = teacher_feature_ratio
    config['training']['consistency_weight'] = consistency_weight
    config['training']['contrastive_weight'] = contrastive_weight
    config['training']['contrastive_temperature'] = contrastive_temperature
    
    # We will set epochs to 500 for proper trial evaluation!
    config['training']['epochs'] = 200
    
    # 2. Setup Data
    device = config['training'].get('device', 'cuda')
    
    feature_dim = train_shapes[0]['feat'].shape[1]
    
    # 3. Model Init
    student_cfg = config.get('student_model', config.get('model', {}))
    student = ASMAE(
        feature_dim=feature_dim,
        embed_dim=student_cfg['embed_dim'],
        depth=student_cfg['depth'],
        num_heads=student_cfg['num_heads'],
        decoder_embed_dim=student_cfg['decoder_embed_dim'],
        decoder_depth=student_cfg['decoder_depth'],
        decoder_num_heads=student_cfg['decoder_num_heads'],
        mlk_ratio=student_cfg['mlk_ratio'],
        num_mask_queries=student_cfg.get('num_mask_queries', 5000),
        encoder_k=student_cfg.get('encoder_k', 20),
        aamg_k=student_cfg.get('aamg_k', 10),
        aamg_emb_dim=student_cfg.get('aamg_emb_dim', 64),
        pos_embed_dim=student_cfg.get('pos_embed_dim', 64),
        temperature=student_cfg.get('temperature', 1.0)
    ).to(device)
    
    teacher_cfg = config.get('teacher_model', config.get('model', {}))
    teacher = ASMAE(
        feature_dim=feature_dim,
        embed_dim=teacher_cfg['embed_dim'],
        depth=teacher_cfg['depth'],
        num_heads=teacher_cfg['num_heads'],
        decoder_embed_dim=teacher_cfg['decoder_embed_dim'],
        decoder_depth=teacher_cfg['decoder_depth'],
        decoder_num_heads=teacher_cfg['decoder_num_heads'],
        mlk_ratio=teacher_cfg['mlk_ratio'],
        num_mask_queries=teacher_cfg.get('num_mask_queries', 5000),
        encoder_k=teacher_cfg.get('encoder_k', 20),
        aamg_k=teacher_cfg.get('aamg_k', 10),
        aamg_emb_dim=teacher_cfg.get('aamg_emb_dim', 64),
        pos_embed_dim=teacher_cfg.get('pos_embed_dim', 64),
        temperature=teacher_cfg.get('temperature', 1.0)
    ).to(device)
    
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False
        
    # 4. Train
    student = train_model(student, teacher, train_shapes, config)
    
    # Prepare test config natively matching the trained config's structure
    test_config = copy.deepcopy(config)
    test_config['testing'] = {
        'device': device,
        'mask_ratio': config['training']['student_mask_ratio'],
        'feature_ratio': config['training']['student_feature_ratio']
    }
    
    # 5. Evaluate (Return MAE)
    mae = test_model(student, test_shapes, test_config)
    
    # Return mapping for Optuna to minimize
    return mae

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING BAYESIAN HYPERPARAMETER SEARCH (OPTUNA)")
    print("Optimizing 9 Student/Teacher Parameters to minimize Test MAE")
    print("=" * 60)
    
    # 0. Load Data ONCE prior to starting tuning
    config = load_base_config()
    data_dir = config['data_dir']
    k = config['geometry']['k']
    t = config['geometry']['t']
    neigvecs = config['geometry'].get('neigvecs', 300)
    train_size = config['train_size']
    test_size = config['test_size']
    
    print("\nPre-loading shapes before tuning starts...")
    train_shapes = load_train_shapes(data_dir, k, t, neigvecs, max_shapes=train_size, output_dir=config['output_dir'])
    test_shapes = load_test_shapes(data_dir, k, t, neigvecs, start_idx=train_size, num_shapes=test_size, output_dir=config['output_dir'])
    
    # Create study
    study = optuna.create_study(direction="minimize")
    
    # Start optimizing (Pass preloaded shapes natively)
    study.optimize(lambda trial: objective(trial, train_shapes, test_shapes), n_trials=50)

    print("\n" + "=" * 60)
    print("OPTIMIZATION FINISHED!")
    print(f"Best Test MAE Achieved: {study.best_value:.6f}")
    print("=" * 60)
    print("Best Hyperparameters Discovered:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

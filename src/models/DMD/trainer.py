import numpy as np
import os
import time
from pathlib import Path
import json
import yaml
import argparse
from typing import Dict, Any, List, Callable

import sys
current_file_path = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_file_path, '..', '..', '..'))
sys.path.insert(0, project_root)

from src.models.DMD.dmd import DMD
from src.models.DMD.edmd import EDMD, create_polynomial_basis, create_rbf_basis, create_fourier_basis
from src.models.DMD.utils import DMDDataLoader


def create_dictionary_functions(dictionary_config: Dict[str, Any]) -> List[Callable[[np.ndarray], np.ndarray]]:
    """
    Create dictionary functions based on configuration
    
    Args:
        dictionary_config: Dictionary configuration from YAML
    
    Returns:
        List of dictionary functions
    """
    functions = []
    
    # Process each dictionary function type
    for func_type, params in dictionary_config.items():
        if not params.get('enabled', True):
            continue
            
        print(f"Creating {func_type} dictionary functions with params: {params}")
        
        if func_type == 'polynomial':
            degree = params.get('degree', 2)
            poly_funcs = create_polynomial_basis(degree=degree)
            functions.extend(poly_funcs)
            print(f"  Added {len(poly_funcs)} polynomial basis functions (degree {degree})")
            
        elif func_type == 'rbf':
            n_centers = params.get('n_centers', 10)
            sigma = params.get('sigma', 1.0)
            center_type = params.get('center_type', 'random')
            
            if center_type == 'random':
                # Generate random centers - we'll need feature dimension
                # For now, store parameters and create during fitting
                functions.append({
                    'type': 'rbf',
                    'n_centers': n_centers,
                    'sigma': sigma,
                    'center_type': center_type
                })
                print(f"  RBF basis registered: {n_centers} centers, sigma={sigma}")
            else:
                raise ValueError(f"Unsupported RBF center_type: {center_type}")
                
        elif func_type == 'fourier':
            frequencies = params.get('frequencies', [1.0, 2.0, 3.0])
            if isinstance(frequencies, dict):
                # Handle range specification
                if 'range' in frequencies:
                    freq_range = frequencies['range']
                    n_freq = frequencies.get('n_frequencies', 5)
                    frequencies = np.linspace(freq_range[0], freq_range[1], n_freq)
            
            fourier_funcs = create_fourier_basis(np.array(frequencies))
            functions.extend(fourier_funcs)
            print(f"  Added {len(fourier_funcs)} Fourier basis functions with frequencies: {frequencies}")
            
        elif func_type == 'custom':
            # Allow for custom function specification
            custom_type = params.get('type', 'identity')
            if custom_type == 'identity':
                def identity_func(x):
                    return x.copy()
                functions.append(identity_func)
                print(f"  Added identity function")
            else:
                print(f"  Warning: Custom function type '{custom_type}' not implemented")
                
        else:
            print(f"  Warning: Unknown dictionary function type: {func_type}")
    
    return functions


def create_rbf_centers(n_centers: int, n_features: int, center_type: str = 'random', 
                      data_range: tuple = None) -> np.ndarray:
    """
    Create RBF centers based on specifications
    
    Args:
        n_centers: Number of RBF centers
        n_features: Feature dimension
        center_type: Type of center placement
        data_range: Optional data range for center placement
    
    Returns:
        RBF centers [n_centers, n_features]
    """
    if center_type == 'random':
        if data_range is not None:
            # Place centers within data range
            low, high = data_range
            centers = np.random.uniform(low, high, size=(n_centers, n_features))
        else:
            # Standard normal distribution
            centers = np.random.randn(n_centers, n_features)
    else:
        raise ValueError(f"Unsupported center_type: {center_type}")
    
    return centers


def process_dictionary_functions(functions: List, n_features: int, 
                                data_sample: np.ndarray = None) -> List[Callable]:
    """
    Process dictionary functions, handling deferred creation for RBF
    
    Args:
        functions: List of functions or function specifications
        n_features: Feature dimension
        data_sample: Sample data for determining ranges
    
    Returns:
        List of callable functions
    """
    processed_functions = []
    
    for func in functions:
        if callable(func):
            # Already a function
            processed_functions.append(func)
        elif isinstance(func, dict) and func.get('type') == 'rbf':
            # Create RBF functions
            n_centers = func['n_centers']
            sigma = func['sigma']
            center_type = func['center_type']
            
            # Determine data range if available
            data_range = None
            if data_sample is not None:
                data_min = np.min(data_sample, axis=0)
                data_max = np.max(data_sample, axis=0)
                # Expand range slightly
                margin = 0.1 * (data_max - data_min)
                data_range = (data_min - margin, data_max + margin)
            
            centers = create_rbf_centers(n_centers, n_features, center_type, data_range)
            rbf_funcs = create_rbf_basis(centers, sigma)
            processed_functions.extend(rbf_funcs)
            print(f"  Created RBF functions with {n_centers} centers")
        else:
            print(f"  Warning: Unknown function specification: {func}")
    
    return processed_functions


def concatenate_training_data(dataset, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate all training samples into time-delay embedded data matrices
    
    Args:
        dataset: Training dataset
        dataset_name: Name of the dataset for specific handling
    
    Returns:
        tuple: (X_train, Y_train) where
               X_train: [S*(T-1), D] - input features at time t
               Y_train: [S*(T-1), D] - target features at time t+1
    """
    print(f"Processing {dataset_name} dataset...")
    print(f"Training dataset shape: {dataset.data.shape}")
    
    if dataset_name.lower() == "kolmogorov":
        # data shape: [samples, time, features] = [96, 320, 4096]
        samples, time_steps, features = dataset.data.shape
        print(f"Kolmogorov: {samples} samples, {time_steps} time steps, {features} features")
        
        # X_flat shape: [S, T, D] where D = features
        X_flat = dataset.data  # Already in correct format [S, T, D]
        
    elif dataset_name.lower() == "cylinder":
        # data shape: [samples, time, channels, spatial_features]
        samples, time_steps, channels, spatial_features = dataset.data.shape
        print(f"Cylinder: {samples} samples, {time_steps} time steps, {channels} channels, {spatial_features} spatial features")
        
        # Reshape to [samples, time, total_features]
        total_features = channels * spatial_features
        X_flat = dataset.data.reshape(samples, time_steps, total_features)  # [S, T, D]
        
    elif dataset_name.lower() == "chap":
        # data shape: [samples, time, features] (years, days, spatial_features)
        samples, time_steps, features = dataset.data.shape
        print(f"CHAP: {samples} samples, {time_steps} time steps, {features} features")
        
        # X_flat shape: [S, T, D] where D = features
        X_flat = dataset.data  # Already in correct format [S, T, D]
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get dimensions
    S, T, D = X_flat.shape
    print(f"Flattened data shape: [S={S}, T={T}, D={D}]")
    
    # Create time-delay embedded data
    X_list = []  # list of [T-1, D] arrays
    Y_list = []  # list of [T-1, D] arrays
    
    for s in range(S):
        # For this sample, take t=0..T-2 as X, t=1..T-1 as Y
        Xi = X_flat[s, :T-1]   # shape [T-1, D] - input at time t
        Yi = X_flat[s, 1:]     # shape [T-1, D] - target at time t+1
        X_list.append(Xi)
        Y_list.append(Yi)
    
    # Stack all samples
    X_train = np.vstack(X_list)  # shape [S*(T-1), D]
    Y_train = np.vstack(Y_list)  # shape [S*(T-1), D]
    
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Total training pairs: {X_train.shape[0]}")
    
    return X_train, Y_train


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from: {config_path}")
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False, indent=2))
    
    return config


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def validate_config(config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Validate and process configuration"""
    # Required fields
    required_fields = ['dataset', 'data_path', 'save_dir']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Set defaults based on model type
    if model_type.upper() == 'DMD':
        defaults = {
            'dmd_params': {
                'svd_rank': None,
                'exact': True,
                'use_streaming': False
            },
            'dataset_params': {},
            'training': {
                'normalize': True,
                'train_ratio': 0.8,
                'random_seed': 42
            }
        }
    elif model_type.upper() == 'EDMD':
        defaults = {
            'edmd_params': {
                'svd_rank': None,
                'exact': True,
                'use_streaming': False
            },
            'dictionary_functions': {
                'polynomial': {
                    'enabled': True,
                    'degree': 2
                }
            },
            'dataset_params': {},
            'training': {
                'normalize': True,
                'train_ratio': 0.8,
                'random_seed': 42
            }
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Merge with defaults
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
        elif isinstance(default_value, dict):
            for sub_key, sub_default in default_value.items():
                if sub_key not in config[key]:
                    config[key][sub_key] = sub_default
    
    # Validate dataset-specific parameters
    dataset_name = config['dataset'].lower()
    if dataset_name == 'chap':
        if 'chemical' not in config['dataset_params']:
            config['dataset_params']['chemical'] = 'Cl'
        if config['dataset_params']['chemical'] not in ['Cl', 'NH4', 'NO3', 'SO4']:
            raise ValueError(f"Invalid chemical for CHAP dataset: {config['dataset_params']['chemical']}")
    
    elif dataset_name == 'cylinder':
        if 'interpolation_mode' not in config['dataset_params']:
            config['dataset_params']['interpolation_mode'] = 'bilinear'
        if config['dataset_params']['interpolation_mode'] not in ['bilinear', 'bicubic', 'nearest']:
            raise ValueError(f"Invalid interpolation mode: {config['dataset_params']['interpolation_mode']}")
    
    return config


def train_large_model(config, model_type: str):
    """
    Train DMD or EDMD on the entire training set using configuration
    
    Args:
        config: Configuration dictionary loaded from YAML
        model_type: 'DMD' or 'EDMD'
    
    Returns:
        Training statistics
    """
    # Extract parameters from config
    dataset_name = config['dataset']
    data_path = config['data_path']
    save_dir = config['save_dir']
    dataset_params = config['dataset_params']
    training_params = config['training']
    
    # Get model-specific parameters
    if model_type.upper() == 'DMD':
        model_params = config['dmd_params']
    elif model_type.upper() == 'EDMD':
        model_params = config['edmd_params']
        dictionary_config = config['dictionary_functions']
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    svd_rank = model_params['svd_rank']
    exact = model_params['exact']
    
    print(f"\n{'='*60}")
    print(f"Training Large {model_type.upper()} Model")
    print(f"Dataset: {dataset_name}")
    print(f"SVD Rank: {svd_rank}")
    print(f"Exact: {exact}")
    if model_type.upper() == 'EDMD':
        print(f"Dictionary Functions: {list(dictionary_config.keys())}")
    print(f"{'='*60}\n")
    
    # Create data loader
    print("Loading dataset...")
    data_loader = DMDDataLoader(
        dataset_name=dataset_name,
        data_path=data_path,
        normalize=training_params['normalize'],
        train_ratio=training_params['train_ratio'],
        random_seed=training_params['random_seed'],
        **dataset_params
    )
    
    # Print dataset statistics
    stats = data_loader.get_data_stats()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get training dataset
    train_dataset = data_loader.train_dataset
    val_dataset = data_loader.val_dataset
    
    # Concatenate all training data
    print("\nConcatenating training data...")
    X_train, Y_train = concatenate_training_data(train_dataset, dataset_name)
    
    # Initialize model
    print(f"\nInitializing {model_type.upper()} model...")

    
    if model_type.upper() == 'DMD':
        model = DMD(svd_rank=svd_rank, exact=exact)
        
    elif model_type.upper() == 'EDMD':
        # Create dictionary functions
        print("Creating dictionary functions...")
        dictionary_functions = create_dictionary_functions(dictionary_config)
        
        # Process any deferred functions (like RBF)
        n_features = X_train.shape[1]
        data_sample = X_train[:min(1000, X_train.shape[0])]  # Sample for range estimation
        dictionary_functions = process_dictionary_functions(
            dictionary_functions, n_features, data_sample
        )
        
        print(f"Total dictionary functions: {len(dictionary_functions)}")
        
        model = EDMD(
            dictionary_funcs=dictionary_functions,
            svd_rank=svd_rank,
            exact=exact
        )
    
    print(f"Fitting {model_type.upper()} model...")
    start_time = time.time()
    
    # Fit the model
    model.fit(X_train, Y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Get model information
    if model._fitted:
        print(f"\nModel Information:")
        print(f"  Training samples: {model.n_samples}")
        print(f"  Feature dimensions: {model.n_features}")
        if model_type.upper() == 'EDMD':
            print(f"  Observable dimensions: {model.n_observables}")
        print(f"  DMD modes shape: {model.Phi.shape}")
        print(f"  Number of eigenvalues: {len(model.Lambda)}")
        
        # Get modes and eigenvalues
        try:
            modes, eigenvalues, amplitudes = model.get_mode_dynamics()
            print(f"  Number of DMD modes: {modes.shape[1]}")
        except AttributeError:
            modes = model.Phi
            eigenvalues = model.Lambda
            amplitudes = model.b if hasattr(model, 'b') else None
            print(f"  DMD modes (Phi): {modes.shape}")
        
        # Show eigenvalue analysis
        eigenvalue_mags = np.abs(eigenvalues)
        print(f"  Eigenvalue magnitudes (top 10): {eigenvalue_mags[:10]}")
        
        # Compute growth rates
        growth_rates = np.real(np.log(eigenvalues + 1e-12))
        stable_modes = np.sum(growth_rates <= 0)
        unstable_modes = np.sum(growth_rates > 0)
        print(f"  Stable modes: {stable_modes}")
        print(f"  Unstable modes: {unstable_modes}")
        print(f"  Growth rates range: [{growth_rates.min():.6f}, {growth_rates.max():.6f}]")
    
    # Test reconstruction on a subset
    print("\nTesting reconstruction...")
    
    # Create test data in the correct format [n_features, n_time_steps]
    test_sample_idx = 5
    if dataset_name.lower() == "kolmogorov":
        test_data_original = val_dataset.data[test_sample_idx]  # [T, D]
        test_data = test_data_original.T  # [D, T]
    elif dataset_name.lower() == "cylinder":
        test_data_original = val_dataset.data[test_sample_idx]  # [T, C, H, W]
        T, C, H_W = test_data_original.shape
        test_data_reshaped = test_data_original.reshape(T, C*H_W)  # [T, D]
        test_data = test_data_reshaped.T  # [D, T]
    elif dataset_name.lower() == "chap":
        test_data_original = val_dataset.data[test_sample_idx]  # [T, D]
        test_data = test_data_original.T  # [D, T]
    
    # Limit test steps for efficiency
    max_test_steps = min(50, test_data.shape[1])
    X_test = test_data[:, :max_test_steps]  # [D, max_test_steps]
    
    print(f"Test data shape: {X_test.shape}")
    X_recon = model.reconstruct(X_test)
    print(f"Reconstructed data shape: {X_recon.shape}")
    
    # Compute reconstruction errors
    if X_recon.shape == X_test.shape:
        errors = model.compute_error(X_test, X_recon)
        print("Reconstruction Errors:")
        for key, value in errors.items():
            print(f"  {key}: {value:.6f}")
    else:
        print("Cannot compute reconstruction errors - dimension mismatch")
        errors = {"note": "dimension_mismatch"}
    
    # Test prediction capability
    print("\nTesting prediction...")
    x0 = X_test[:, 0]  # Use first time step as initial condition
    n_pred_steps = min(20, max_test_steps)
    X_pred = model.predict(x0=x0, n_steps=n_pred_steps)
    print(f"Prediction shape: {X_pred.shape}")
    
    # Compare prediction with true data if available
    pred_errors = {"note": "not_computed"}
    if (n_pred_steps < X_test.shape[1] and 
        X_pred.shape[0] == X_test.shape[0]):
        X_true_continuation = X_test[:, 1:n_pred_steps+1]
        if X_pred.shape == X_true_continuation.shape:
            pred_errors = model.compute_error(X_true_continuation, X_pred)
            print("Prediction Errors:")
            for key, value in pred_errors.items():
                print(f"  {key}: {value:.6f}")
        else:
            print("Cannot compute prediction errors - shape mismatch")
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    
    # Create model filename
    model_name = f"{model_type.lower()}_{dataset_name}"
    if svd_rank is not None:
        model_name += f"_rank{svd_rank}"
    if exact:
        model_name += "_exact"
    else:
        model_name += "_projected"
    
    model_path = os.path.join(save_dir, f"{model_name}.npz")
    
    print(f"\nSaving model to: {model_path}")
    if model_type.upper() == 'EDMD':
        model.save(model_path)
    else:
        model.save(model_path)
    
    # Save training metadata
    metadata = {
        "model_type": model_type.upper(),
        "config": config,
        "training_time": float(training_time),
        "data_shape": {
            "X_train": [int(x) for x in X_train.shape],
            "Y_train": [int(x) for x in Y_train.shape],
            "test_data": [int(x) for x in X_test.shape]
        },
        "reconstruction_errors": convert_numpy_types(errors),
        "prediction_errors": convert_numpy_types(pred_errors),
        "dataset_stats": convert_numpy_types(stats),
        "model_info": {}
    }
    
    # Add model information if fitted successfully
    if model._fitted:
        model_info = {
            "n_features": int(model.n_features),
            "n_samples": int(model.n_samples),
            "n_modes": int(modes.shape[1]) if modes is not None else None,
            "stable_modes": int(stable_modes),
            "unstable_modes": int(unstable_modes),
            "eigenvalue_magnitudes_top10": [float(x) for x in eigenvalue_mags[:10]],
            "growth_rates_range": [float(growth_rates.min()), float(growth_rates.max())],
            "training_data_shapes": {
                "X_train_shape": getattr(model, '_X_train_shape', None),
                "Y_train_shape": getattr(model, '_Y_train_shape', None)
            }
        }
        
        if model_type.upper() == 'EDMD':
            model_info["n_observables"] = int(model.n_observables)
            model_info["n_dictionary_functions"] = len(model.dictionary_funcs)
        
        metadata["model_info"] = model_info
    
    # Save normalization parameters for later use
    if hasattr(train_dataset, 'mean') and train_dataset.mean is not None:
        metadata["normalization"] = {
            "mean": convert_numpy_types(train_dataset.mean),
            "std": convert_numpy_types(train_dataset.std)
        }
    
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    
    return metadata


def main():
    """Main function to run training with command line arguments"""
    parser = argparse.ArgumentParser(description='Train DMD or EDMD models')
    parser.add_argument('--model', type=str, choices=['DMD', 'EDMD'], required=True,
                       help='Model type to train (DMD or EDMD)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    model_type = args.model.upper()
    config_path = args.config
    
    print(f"Training {model_type} model with config: {config_path}")
    
    try:
        # Load and validate configuration
        config = load_config(config_path)
        config = validate_config(config, model_type)
        
        # Train model
        metadata = train_large_model(config, model_type)
        
        print(f"\n{'='*60}")
        print(f"{model_type} training completed successfully!")
        print(f"Model saved in: {config['save_dir']}")
        print(f"Training time: {metadata['training_time']:.2f} seconds")
        if 'model_info' in metadata and metadata['model_info']:
            print(f"Model features: {metadata['model_info']['n_features']}")
            print(f"Training samples: {metadata['model_info']['n_samples']}")
            print(f"DMD modes: {metadata['model_info']['n_modes']}")
            if model_type == 'EDMD':
                print(f"Observable features: {metadata['model_info']['n_observables']}")
                print(f"Dictionary functions: {metadata['model_info']['n_dictionary_functions']}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
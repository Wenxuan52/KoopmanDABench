# Results for ESE MSc Independent Research Project (wy524)

This directory contains experimental results, trained models, and analysis outputs from the thesis research on data assimilation methods with different dynamical models.

## Download

The dataset files can be downloaded from Google Drive:

[Download wy524_irp_results.zip](https://drive.google.com/file/d/1PP5Xl5RUVIsWT_ZCkj5Q-tbGPWcqC2h_/view?usp=drive_link)

## Extraction

***Important:*** Do not extract `wy524_irp_results.zip` directly here, otherwise this `README.md` may be overwritten.  
Instead, extract to a temporary folder and then merge:

```bash
# Start from root directory
# create a temporary folder
mkdir tmp_results

# unzip into the temporary folder
unzip wy524_irp_results.zip -d tmp_results/

# merge contents into ./data, while keeping this README.md
rsync -av --ignore-existing tmp_results/results/ results/

# remove the temporary folder
rm -r tmp_results
```

After extraction, the directory structure will look like:

```bash
code/
├── README.md
├── results/
│   ├── README.md
│   ├── CAE_DMD/
│   ├── CAE_Koopman/
│   └── ...
├── ...
```

## Directory Structure

### Baseline Model Results

#### **DMD ROM**: `CAE_DMD/`
Nonlinear Autoencoder with latent Dynamic Mode Decomposition results:
- **Cylinder/**, **Dam/**, **ERA5/**: Trained model weights (`dmd_model.pth`, `model_info.pth`)
- **DA/**: Data assimilation analysis results including performance comparisons, noise impact studies, and latent space visualizations
- **figures/**: Model comparison plots, error analysis, and rollout predictions

#### **Koopman ROM**: `CAE_Koopman/`
Nonlinear Autoencoder with Koopman operator results:
- **Cylinder/**, **Dam/**, **ERA5/**: Model weights organized by dataset with multiple training configurations
- **DA/**: Data assimilation performance analysis and comparison metrics
- **figures/**: Visualization of model performance and error analysis

#### **Linear ROM**: `CAE_Linear/`
Nonlinear Autoencoder with linear dynamics:
- **Cylinder/**, **Dam/**, **ERA5/**: Trained model weights and statistics
- **DA/**: Data assimilation analysis and performance comparisons
- **figures/**: Model evaluation visualizations

#### **MLP ROM**: `CAE_MLP/`
Nonlinear Autoencoder with Multi-Layer Perceptron dynamics:
- Similar structure to other NAE variants with MLP-based forward models

#### **Weaklinear ROM**: `CAE_Weaklinear/`
Nonlinear Autoencoder with weakly nonlinear dynamics:
- Hybrid approach combining nonlinear structure with linear constraint.

#### **DMD**: `DMD/`
Standard Dynamic Mode Decomposition baseline:
- **Cylinder/**, **Dam/**, **ERA5/**: Pure DMD model weights for different datasets
- **DA/**: Data assimilation performance baselines
- **figures/**: Comprehensive DMD performance analysis

### Comparative Analysis

#### Comparison/
Cross-model comparison and integrated analysis:
- **figures/**: Comprehensive comparison plots across all methods including:
  - 4D-Var assimilation comparisons
  - Average performance metrics
  - Background field improvements
  - Multi-model latent space analysis
  - Noise robustness comparisons

## File Types

- **.pth/.pt**: PyTorch model checkpoints and weights
- **.pkl**: Serialized Python objects (statistics, metrics, analysis data)
- **.png**: Visualization plots and figures
- **.npy/.npz**: NumPy arrays (rollout predictions, comparison data)

## Key Analysis Components

### Data Assimilation (DA) Results
Each model directory contains DA analysis including:
- Performance comparisons under different noise levels
- Observation density impact studies
- Latent space analysis with t-SNE visualizations
- Signal-to-noise ratio vs. performance relationships

### Model Performance Metrics
- **Rollout predictions**: Long-term forecasting accuracy
- **Error analysis**: Spatial and temporal error distributions
- **Per-frame metrics**: Time-series performance evolution
- **Channel-wise analysis**: Multi-variable performance (ERA5)

### Comparative Studies
- Cross-model performance benchmarking
- 4D-Var integration effectiveness
- Background field improvement quantification
- Latent dynamic visualization

# Datasets for ESE MSc Independent Research Project (wy524)

This directory contains datasets and related files for the IRP project, organized into three main categories. 

## Download

The dataset files can be downloaded from Google Drive:

[Download wy524_irp_experiment_data.zip](https://drive.google.com/file/d/1U9SXe6lyA9PXwpaP1lMRcgYsivOBfwT2/view?usp=drive_link)

## Extraction

***Important:*** Do not extract `wy524_irp_experiment_data.zip` directly here, otherwise this `README.md` may be overwritten.  
Instead, extract to a temporary folder and then merge:

```bash
# Start from root directory
# create a temporary folder
mkdir tmp_data

# unzip into the temporary folder
unzip wy524_irp_experiment_data.zip -d tmp_data/

# merge contents into ./data, while keeping this README.md
rsync -av --ignore-existing tmp_data/data/ data/

# remove the temporary folder
rm -r tmp_data
```

After extraction, the directory structure will look like:

```bash
code/
├── README.md
├── data/
│   ├── README.md
│   ├── cylinder/
│   ├── dam/
│   └── ERA5/
├── ...
```

## Data Sources

- **cylinder/** and **dam/**: Computational fluid dynamics data from [CFDBench](https://github.com/luo-yining/CFDBench)
- **ERA5/**: Weather reanalysis data from [WeatherBench](https://github.com/pangeo-data/WeatherBench)

## Directory Structure

### cylinder/
Contains cylindrical flow simulation data:
- `cylinder_data.npy` - Complete dataset
- `cylinder_train_data.npy` - Training subset
- `cylinder_val_data.npy` - Validation subset

### dam/
Contains dam break simulation data:
- `dam_data.npy` - Complete dataset  
- `dam_train_data.npy` - Training subset
- `dam_val_data.npy` - Validation subset
- `idx.txt` - Index file for data organization
- `visual.py` - Visualization script

### ERA5/
Contains ERA5 reanalysis weather data:
- `ERA5_data/` - Main data directory
  - `*_seq_obs.h5` - Observation sequences (train/val/test)
  - `*_seq_state.h5` - State sequences (train/val/test) 
  - `max_val.npy`, `min_val.npy` - Normalization parameters
  - `weight_matrix.npy` - Weighting matrix for data processing
- `info.py` - Dataset information and utilities
- `visual.py` - Visualization script
- `weight_matrix_heatmap.png` - Weight matrix visualization

## File Formats
- `.npy` - NumPy arrays (simulation data, parameters)
- `.h5` - HDF5 format (sequential weather data)
- `.py` - Python scripts for data processing and visualization
- `.txt` - Text files (indices, metadata)

## Usage Notes
This data is used for research purposes only. Please refer to the original data sources for licensing and usage terms:
- CFDBench: [https://github.com/luo-yining/CFDBench](https://github.com/luo-yining/CFDBench)
- WeatherBench: [https://github.com/pangeo-data/WeatherBench](https://github.com/pangeo-data/WeatherBench)
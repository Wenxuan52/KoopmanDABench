#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import numpy as np
import os
import h5py
import os
import xarray as xr
from pathlib import Path
# from data_generation.cfdbench import get_auto_dataset

def process_weatherbench(filename, timesteps_per_file, save_folder):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    with h5py.File(filename, 'r') as f:
        # Assuming the dataset is named 'data'. Adjust if it's named differently.
        original_data = f['data']
        
        total_timesteps = original_data.shape[0]
        print(total_timesteps, original_data.shape)
        num_files = int(np.ceil(total_timesteps / timesteps_per_file))
        
        for i in range(num_files):
            start_idx = i * timesteps_per_file
            end_idx = min((i + 1) * timesteps_per_file, total_timesteps)
            print(original_data[start_idx:end_idx].shape)
            out = original_data[start_idx:end_idx]
            out = np.transpose(out, (1, 2, 0, 3))
            print(out.shape)
            # # Create a new file for this chunk
            with h5py.File(save_folder + 'data_{}.hdf5'.format(i), 'w') as new_file:
                # Copy the chunk of data to the new file
                new_file.create_dataset('data', data=original_data[start_idx:end_idx])
                
                # If there are any attributes you want to copy, do it here
                for key, value in original_data.attrs.items():
                    new_file['data'].attrs[key] = value
                
            print(f"Created file {i+1} of {num_files}")

    print("Splitting complete!")
    return


def preprocess_weatherbench(data_path):
    """
    Preprocess the ERA5 dataset from WeatherBench

    there are 32 variables in the dataset but extract 26

    test data shape: (1460, 64, 32, 26)
    train data shape : (54056, 64, 32, 26)
    """
    # GCP_PATH = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr'

    GCP_PATH = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'

    # Load in the data
    ds = xr.open_zarr(GCP_PATH)

    train_time = slice("1979-01-01T00:00:00", "2015-12-31T18:00:00")
    test_time  = slice("2018-01-01T00:00:00", "2018-12-31T18:00:00")


    z500 = ds['geopotential'][:,7,...]
    u850 = ds['u_component_of_wind'][:,10,...]
    v850 = ds['v_component_of_wind'][:,10,...]
    t850 = ds['temperature'][:,10,...]
    # Note that SFNO paper uses relative humidity
    q700 = ds['specific_humidity'][:,8,...]

    with h5py.File(os.path.join(data_path,'weatherbench_test.h5'), 'w') as g:
        data = np.stack([
            z500.sel(time=test_time),
            u850.sel(time=test_time),
            v850.sel(time=test_time),
            t850.sel(time=test_time),
            q700.sel(time=test_time),
        ], axis=-1)
        
        g.create_dataset('data', data=data)
    g.close()

    with h5py.File(os.path.join(data_path,'weatherbench_train.h5'), 'w') as g:
        data = np.stack([
            z500.sel(time=train_time),
            u850.sel(time=train_time),
            v850.sel(time=train_time),
            t850.sel(time=train_time),
            q700.sel(time=train_time),
        ], axis=-1)
        g.create_dataset('data', data=data)
    
    g.close()

# def preprocess_weatherbench_geo(data_path):
#     """
#     Preprocess the ERA5 dataset from WeatherBench

#     there are 32 variables in the dataset but extract geopotential

#     test data shape: (1460, 64, 32, 1)
#     train data shape : (1460, 64, 32, 1)
#     """
#     GCP_PATH = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr'

#     # Load in the data
#     ds = xr.open_zarr(GCP_PATH)
#     train_start = 29216 # '1979-01-01T00:00:00.000000000'
#     train_end = 30676 # '2016-01-01T00:00:00.000000000'

#     test_start = 86196 # '2018-01-01T00:00:00.000000000'
#     test_end = 87656 # '2019-01-01T00:00:00.000000000'


#     z500 = ds['geopotential'][:,7,...]

#     with h5py.File(os.path.join(data_path,'weatherbench_geo_test.h5'), 'w') as g:
#         data = np.stack([z500[test_start:test_end,...]], axis=-1)
        
#         g.create_dataset('data', data=data)
#     g.close()

#     with h5py.File(os.path.join(data_path,'weatherbench_geo_train.h5'), 'w') as g:
#         data = np.stack([z500[train_start:train_end,...]], axis=-1)
        
#         g.create_dataset('data', data=data)
    
#     g.close()

if __name__ == '__main__':
    #### WeatherBench datasets
    data_path = 'raw_data/'
    timesteps_per_file = 20
    # preprocess_weatherbench_geo(data_path)
    preprocess_weatherbench(data_path)
    process_weatherbench(os.path.join(data_path,'weatherbench_train.h5'), timesteps_per_file, os.path.join(data_path,'train/'))
    process_weatherbench(os.path.join(data_path,'weatherbench_test.h5'), timesteps_per_file, os.path.join(data_path,'test/'))
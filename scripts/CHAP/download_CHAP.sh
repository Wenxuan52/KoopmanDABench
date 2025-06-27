#!/bin/bash

# Base URL
base_url="https://zenodo.org/record/10011898/files"

# Ion types
ions=("Cl" "NH4" "NO3" "SO4")

# Years
years=(2013 2014 2015 2016 2017 2018 2019 2020)

# Download root directory
root_dir="../data/CHAP"

# Create root directory
mkdir -p "$root_dir"

# Loop through each ion and year
for ion in "${ions[@]}"; do
    ion_dir="$root_dir/$ion"
    mkdir -p "$ion_dir"

    for year in "${years[@]}"; do
        filename="ECHAP_${ion}_D1K_${year}_V1.zip"
        url="${base_url}/${filename}?download=1"
        output_path="${ion_dir}/${filename}"

        echo "Downloading $filename into $ion_dir"
        wget -q --show-progress "$url" -O "$output_path"
    done
done

# MRI Denoising Project

This repository provides simple utilities for downloading fMRI data from the [OpenNeuro](https://openneuro.org/) repository and training a lightweight 3D U-Net Wasserstein GAN (UWGAN) for denoising.

These scripts are intended to run in environments such as Google Colab.

## Data Collection

`data_collection.py` streams `.nii.gz` files directly from the OpenNeuro S3 bucket and saves them into a single NumPy array file.

Example usage to download 20 runs from the `ds002306` dataset:

```bash
python data_collection.py --output fmri_dataset_chunk.npy --num-files 20
```

## Training the Denoising Model

The training script expects a single `.nii.gz` file. Preprocessing extracts smaller 3D patches and adds Gaussian noise for training.

```bash
python uwgan_denoiser.py /path/to/file.nii.gz --epochs 10 --batch-size 10
```

The script will save the generator and critic models (`fmri_denoiser_generator.h5` and `fmri_critic.h5`) along with visualization images of intermediate results.

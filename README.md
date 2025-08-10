# MRI Denoising Project

Utilities for streaming fMRI volumes from [OpenNeuro](https://openneuro.org/) and
training a lightweight 3D U-Net Wasserstein GAN (UWGAN) to remove noise.

## Repository structure

```
├── scripts
│   ├── collect_data.py   # CLI for downloading fMRI runs
│   └── train_model.py    # CLI for training the denoiser
└── src
    └── mri_denoising
        ├── __init__.py
        ├── data_collection.py   # Streaming utilities
        └── uwgan_denoiser.py    # Model and training code
```

## Getting started

The project requires Python 3 with TensorFlow, NumPy, Nibabel and other common
scientific libraries installed.

### Download sample data

```bash
python scripts/collect_data.py --output fmri_dataset_chunk.npy --num-files 20
```

### Train the denoising model

Provide a `.nii.gz` file containing fMRI data. The script extracts 3D patches,
adds Gaussian noise and trains the UWGAN model.

```bash
python scripts/train_model.py /path/to/file.nii.gz --epochs 10 --batch-size 10
```

The generator and critic models are saved as `fmri_denoiser_generator.h5` and
`fmri_critic.h5`, respectively, along with visualization images of intermediate
results.

---

Feel free to adapt the scripts for larger datasets or different training
configurations.

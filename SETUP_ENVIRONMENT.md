# CUDA-Enabled Environment Setup Guide

This guide will help you set up a complete CUDA-enabled Python environment for face detection with GPU acceleration.

## Prerequisites

- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (`nvidia-smi` should work)
- CUDA Toolkit 11.8 (install using `install_cuda_11.8.sh` if needed)

## Quick Setup (Recommended)

### Step 1: Install Conda/Mamba

If you don't have conda installed:

```bash
chmod +x install_conda.sh
./install_conda.sh
```

Choose **Mamba** (option 1) for faster package resolution, or **Miniconda** (option 2) for standard conda.

After installation, **restart your terminal** or run:

```bash
source ~/.bashrc
```

### Step 2: Create the Environment

```bash
conda env create -f environment.yml
```

This will create an environment named `facedet-cuda` with:
- ✅ Python 3.12
- ✅ PyTorch with CUDA 11.8 support
- ✅ OpenCV with CUDA support
- ✅ All data science packages (pandas, matplotlib, seaborn, numpy)
- ✅ Additional ML libraries (scikit-learn, scipy)

### Step 3: Activate the Environment

```bash
conda activate facedet-cuda
```

### Step 4: Verify CUDA Support

Run the test script:

```bash
python test_kuda.py
```

You should see:
```
CUDA is available. PyTorch can use the GPU.
Number of available GPUs: 1
Current GPU Name: [Your GPU Name]
```

Test OpenCV CUDA:

```python
import cv2
print(f"OpenCV Version: {cv2.__version__}")
print(f"CUDA Devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
```

## Manual Installation (Alternative)

If conda is already installed:

```bash
# Using conda
conda env create -f environment.yml
conda activate facedet-cuda

# Or using mamba (faster)
mamba env create -f environment.yml
mamba activate facedet-cuda
```

## Updating the Environment

If you need to add packages:

```bash
# Edit environment.yml, then update:
conda env update -f environment.yml --prune
```

## Removing the Environment

```bash
conda deactivate
conda env remove -n facedet-cuda
```

## Troubleshooting

### CUDA Not Detected

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```

2. **Check CUDA toolkit:**
   ```bash
   nvcc --version
   ```

3. **Verify GPU compatibility:**
   Your GPU must support CUDA Compute Capability ≥ 3.5

### Conda is Slow

Use `mamba` instead - it's a drop-in replacement for conda but much faster:

```bash
# Install mamba in base environment
conda install -c conda-forge mamba

# Use mamba instead of conda
mamba env create -f environment.yml
```

### Package Conflicts

If you encounter package conflicts:

```bash
# Use mamba for better dependency solving
mamba env create -f environment.yml

# Or create environment with specific Python version
conda create -n facedet-cuda python=3.12
conda activate facedet-cuda
conda install -c pytorch -c nvidia -c conda-forge --file environment.yml
```

## Environment Information

The environment includes:

| Package | Version | CUDA Support |
|---------|---------|--------------|
| Python | 3.12 | N/A |
| PyTorch | 2.5.x | ✅ CUDA 11.8 |
| OpenCV | 4.10.x | ✅ CUDA |
| NumPy | Latest | CPU |
| Pandas | Latest | CPU |
| Matplotlib | Latest | CPU |
| Seaborn | Latest | CPU |

## Next Steps

After setting up the environment:

1. Activate the environment: `conda activate facedet-cuda`
2. Run your preprocessing script: `python 02_praproses_optimized.py`
3. Monitor GPU usage: `watch -n 1 nvidia-smi`

Enjoy GPU-accelerated face detection! 🚀

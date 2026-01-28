"""
Test Installation
=================

Run this script to verify all components are working correctly.

Usage:
    python test_installation.py
"""

import sys
from pathlib import Path

# Add project to path - handle both script and Jupyter notebook
try:
    PROJECT_DIR = Path(__file__).parent
except NameError:
    PROJECT_DIR = Path(r"D:\Udemy_Cour\Pancharping\pansharpening_project")

sys.path.insert(0, str(PROJECT_DIR))

print("=" * 60)
print("Pansharpening Project - Installation Test")
print("=" * 60)

# Test 1: Import utilities
print("\n[1/5] Testing utilities import...")
try:
    from utils import (
        load_image, normalize_image, upsample_image, save_geotiff,
        create_patches, compute_all_metrics, print_metrics,
        plot_comparison, plot_metrics_comparison
    )
    print("  OK - All utilities imported successfully")
except ImportError as e:
    print(f"  FAILED - {e}")

# Test 2: Import models
print("\n[2/5] Testing models import...")
try:
    from models import PNN, PanNet, DRPNN, CombinedLoss, GradientLoss
    print("  OK - All models imported successfully")
except ImportError as e:
    print(f"  FAILED - {e}")

# Test 3: Import classic methods
print("\n[3/5] Testing classic methods import...")
try:
    from methods.classic import (
        brovey_fusion, ihs_fusion, sfim_fusion,
        gram_schmidt_fusion, hpf_fusion
    )
    print("  OK - All classic methods imported successfully")
except ImportError as e:
    print(f"  FAILED - {e}")

# Test 4: Import deep learning components
print("\n[4/5] Testing deep learning components import...")
try:
    from methods.deep_learning import PansharpeningTrainer, PansharpeningDataset
    print("  OK - All DL components imported successfully")
except ImportError as e:
    print(f"  FAILED - {e}")

# Test 5: Import configuration
print("\n[5/5] Testing configuration import...")
try:
    from configs.config import (
        DEFAULT_PAN_PATH, DEFAULT_MS_PATH, RESULTS_DIR,
        TRAINING_CONFIG, AVAILABLE_MODELS
    )
    print("  OK - Configuration imported successfully")
    print(f"  PAN path: {DEFAULT_PAN_PATH}")
    print(f"  MS path: {DEFAULT_MS_PATH}")
except ImportError as e:
    print(f"  FAILED - {e}")

# Test 6: Check PyTorch
print("\n[Bonus] Checking PyTorch and CUDA...")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("  PyTorch not found - please install PyTorch")

# Test 7: Create a simple model and run forward pass
print("\n[Bonus] Testing model forward pass...")
try:
    import torch
    import numpy as np
    from models import PanNet

    model = PanNet(ms_bands=4)
    ms_dummy = torch.randn(1, 4, 64, 64)
    pan_dummy = torch.randn(1, 1, 64, 64)
    output = model(ms_dummy, pan_dummy)
    print(f"  Input shapes: MS={list(ms_dummy.shape)}, PAN={list(pan_dummy.shape)}")
    print(f"  Output shape: {list(output.shape)}")
    print("  OK - Model forward pass successful")
except Exception as e:
    print(f"  FAILED - {e}")

print("\n" + "=" * 60)
print("Installation test complete!")
print("=" * 60)

# Check if data files exist
print("\nChecking data files...")
pan_exists = Path(DEFAULT_PAN_PATH).exists()
ms_exists = Path(DEFAULT_MS_PATH).exists()

if pan_exists and ms_exists:
    print(f"  PAN file: Found")
    print(f"  MS file: Found")
    print("\nYou can now run:")
    print("  python run_classic.py         # Run classic methods")
    print("  python run_deep_learning.py   # Train deep learning model")
    print("  python run_all.py             # Run all methods")
else:
    print(f"  PAN file: {'Found' if pan_exists else 'NOT FOUND'}")
    print(f"  MS file: {'Found' if ms_exists else 'NOT FOUND'}")
    print("\nPlease update paths in configs/config.py or provide paths via command line:")
    print("  python run_classic.py --pan path/to/pan.tif --ms path/to/ms.tif")

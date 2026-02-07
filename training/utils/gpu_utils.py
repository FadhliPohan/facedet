"""
GPU Configuration and Monitoring Utilities
Optimized for NVIDIA RTX GPUs
"""
import tensorflow as tf
import os


def configure_gpu(memory_limit_mb=None, allow_growth=True, use_mixed_precision=True):
    """
    Configure GPU settings for optimal training
    
    Args:
        memory_limit_mb: Limit GPU memory (None = use all available)
        allow_growth: Allow dynamic memory growth
        use_mixed_precision: Enable mixed precision (FP16) for RTX GPUs
        
    Returns:
        True if GPU configured successfully, False otherwise
    """
    print("\n" + "="*80)
    print("⚙️  GPU CONFIGURATION")
    print("="*80)
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("❌ NO GPU DETECTED!")
        print("   Training requires GPU. Please check your CUDA installation.")
        print("="*80 + "\n")
        return False
    
    try:
        # Configure each GPU
        for gpu in gpus:
            # Set memory growth
            if allow_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Memory growth enabled for {gpu.name}")
            
            # Set memory limit if specified
            if memory_limit_mb:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                )
                print(f"✅ Memory limit set to {memory_limit_mb} MB")
        
        # Configure mixed precision for RTX GPUs
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"✅ Mixed precision enabled (FP16)")
            print(f"   Compute dtype: {policy.compute_dtype}")
            print(f"   Variable dtype: {policy.variable_dtype}")
        
        print(f"\n✅ GPU configuration successful!")
        print(f"   Available GPUs: {len(gpus)}")
        
        return True
        
    except RuntimeError as e:
        print(f"❌ GPU configuration failed: {e}")
        return False
    finally:
        print("="*80 + "\n")


def check_gpu_availability(raise_on_fail=True):
    """
    Check if GPU is available
    
    Args:
        raise_on_fail: Raise exception if no GPU found
        
    Returns:
        bool: True if GPU available
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        msg = "NO GPU DETECTED! Training requires NVIDIA GPU with CUDA support."
        if raise_on_fail:
            raise RuntimeError(msg)
        else:
            print(f"⚠️  {msg}")
            return False
    
    return True


def get_gpu_info():
    """
    Get detailed GPU information
    
    Returns:
        dict: GPU information
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        return {
            'available': False,
            'count': 0,
            'devices': []
        }
    
    gpu_info = {
        'available': True,
        'count': len(gpus),
        'devices': []
    }
    
    for i, gpu in enumerate(gpus):
        device_info = {
            'id': i,
            'name': gpu.name,
            'type': gpu.device_type
        }
        
        # Try to get CUDA version and device name
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                                   '--format=csv,noheader,nounits', f'--id={i}'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                output = result.stdout.strip().split(',')
                if len(output) >= 2:
                    device_info['gpu_name'] = output[0].strip()
                    device_info['total_memory_mb'] = float(output[1].strip())
        except:
            pass
        
        gpu_info['devices'].append(device_info)
    
    return gpu_info


def print_gpu_info():
    """Print GPU information in a readable format"""
    info = get_gpu_info()
    
    print("\n" + "="*80)
    print("🖥️  GPU INFORMATION")
    print("="*80)
    
    if not info['available']:
        print("❌ No GPU available")
    else:
        print(f"✅ {info['count']} GPU(s) detected\n")
        
        for device in info['devices']:
            print(f"GPU {device['id']}:")
            print(f"  Device: {device['name']}")
            if 'gpu_name' in device:
                print(f"  Name: {device['gpu_name']}")
            if 'total_memory_mb' in device:
                print(f"  Memory: {device['total_memory_mb']:.0f} MB ({device['total_memory_mb']/1024:.1f} GB)")
            print()
    
    print("="*80 + "\n")
    
    return info


def get_memory_usage():
    """
    Get current GPU and RAM memory usage
    
    Returns:
        dict: Memory usage information
    """
    import psutil
    
    memory_info = {
        'ram': {
            'total_mb': psutil.virtual_memory().total / (1024**2),
            'used_mb': psutil.virtual_memory().used / (1024**2),
            'percent': psutil.virtual_memory().percent
        },
        'gpu': []
    }
    
    # Try to get GPU memory info
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                used, total = line.split(',')
                memory_info['gpu'].append({
                    'used_mb': float(used.strip()),
                    'total_mb': float(total.strip()),
                    'percent': (float(used.strip()) / float(total.strip())) * 100
                })
    except:
        pass
    
    return memory_info


def print_memory_usage():
    """Print current memory usage"""
    info = get_memory_usage()
    
    print("\n💾 MEMORY USAGE:")
    print(f"   RAM: {info['ram']['used_mb']:.0f} / {info['ram']['total_mb']:.0f} MB ({info['ram']['percent']:.1f}%)")
    
    for i, gpu_mem in enumerate(info['gpu']):
        print(f"   GPU {i}: {gpu_mem['used_mb']:.0f} / {gpu_mem['total_mb']:.0f} MB ({gpu_mem['percent']:.1f}%)")
    print()


def clear_gpu_memory():
    """
    Clear GPU memory and run garbage collection
    """
    import gc
    
    # Clear Keras session
    tf.keras.backend.clear_session()
    
    # Run garbage collection
    gc.collect()
    
    print("🧹 GPU memory cleared")


def setup_gpu_for_training(verbose=True):
    """
    Complete GPU setup for training
    This should be called at the start of training scripts
    
    Returns:
        bool: True if setup successful
    """
    if verbose:
        print_gpu_info()
    
    # Check availability
    if not check_gpu_availability(raise_on_fail=True):
        return False
    
    # Configure GPU
    success = configure_gpu(
        memory_limit_mb=None,  # Use all available
        allow_growth=True,      # Dynamic growth
        use_mixed_precision=True  # FP16 for RTX GPUs
    )
    
    if verbose and success:
        print_memory_usage()
    
    return success

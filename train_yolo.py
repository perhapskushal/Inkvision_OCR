from ultralytics import YOLO
import yaml
from pathlib import Path
import torch
import signal
import sys
import os

# Set PyTorch memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def create_dataset_yaml():
    """Create dataset.yaml file for YOLOv8 training"""
    dataset_config = {
        'path': str(Path('dataset').absolute()),
        'train': 'train/images',
        'val': 'test/images',
        'names': {
            0: 'nepali_text'
        }
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)

def train_yolo():
    """Train YOLOv8 model on the Nepali text dataset"""
    try:
        # Create dataset configuration
        create_dataset_yaml()
        
        # Initialize YOLOv8 model
        model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8 nano model
        
        # Configure training parameters for GPU with minimal memory usage
        training_args = {
            'data': 'dataset.yaml',
            'epochs': 100,
            'imgsz': 320,  # Further reduced image size
            'batch': 8,    # Minimal batch size
            'patience': 20,
            'save': True,
            'device': 0,   # Use first GPU
            'workers': 2,  # Minimal workers
            'cache': False, # Disable caching
            'amp': True,   # Enable automatic mixed precision
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'close_mosaic': 10,
            'verbose': True,
            'half': True,  # Enable FP16 training
            'rect': True,  # Enable rectangular training
            'multi_scale': False,  # Disable multi-scale training
            'overlap_mask': False,  # Disable mask overlap
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'save_json': False,
            'save_hybrid': False,
            'conf': 0.001,  # Lower confidence threshold
            'iou': 0.6,    # Lower IoU threshold
            'max_det': 100  # Reduce maximum detections
        }
        
        # Clear CUDA cache before training
        torch.cuda.empty_cache()
        
        # Train the model
        results = model.train(**training_args)
        
        # Save the trained model
        model.export(format='onnx')  # Export to ONNX format for faster inference
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        sys.exit(1)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\nTraining interrupted by user. Stopping...")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start training
    train_yolo() 
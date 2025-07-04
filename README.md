# retico-objectFeatures

A retico module for extracting visual features from detected objects using CLIP and other vision models from Hugging Face transformers.

## Overview

The `retico-objectFeatures` module provides functionality to extract visual features from detected objects from other modules like [YoloV11](https://github.com/retico-team/retico-yolov11.git) or [SAM](https://github.com/retico-team/retico-sam.git). 
It uses state-of-the-art vision models including CLIP, ResNet, ViT, and other transformers compatible models to generate meaningful feature representations from extracted objects.

## Installation

### Step 1: Install retico dependencies
First, ensure you have the `retico-core` and `retico-vision` modules installed.
The `retico-vision` module needs to be installed and added to your Python path:
```bash
git clone https://github.com/retico-team/retico-vision.git
```
**Important**: Make sure to add the path to the `retico-vision` library to your `PYTHONPATH` environment variable. This is required for the module to properly import the vision components.

### Step 2: Install the package

```bash
pip install git+https://github.com/retico-team/retico-objectFeatures.git
```

## Usage
For a basic example of how to use the `retico-objectFeatures` module, refer to the `example.py` file in the repository. Note that you will also need to install and add to the environment the `retico-yolov11` module to provide the object detection capabilities.

## Configuration

### ObjectFeaturesExtractor Parameters

- `model_name` (str): HuggingFace model identifier (default: "openai/clip-vit-base-patch32")
- `top_objects` (int): Number of top objects to extract features from (default: 1)
- `pool` (bool): Whether to use pooled features or raw hidden states (default: True). Note that the features are flattened to a 1D vector.

### Project Structure

```
retico-objectFeatures/
├── retico_objectFeatures/
│   ├── __init__.py
│   ├── objects_feat_extr.py    # Main feature extraction module
│   └── version.py              # Version information
├── setup.py                    # Package setup
├── example.py                  # Example usage script
├── README.md                   # This file
└── LICENSE                     # License file
```

## Related Projects

- [ReTiCo Core](https://github.com/retico-team/retico-core) - The core ReTiCo framework
- [ReTiCo Vision](https://github.com/retico-team/retico-vision) - Vision components for ReTiCo
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - The transformers library used for vision models

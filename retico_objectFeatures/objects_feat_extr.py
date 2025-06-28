"""
CLIP Module
==================

This module provides extracts features from ExtractedObjectsIU using CLIP.
"""
import itertools
import threading
import time
import torch
from collections import deque

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,            # covers ResNet, ViT, ConvNeXt, …
    CLIPVisionModel,      # vision-only half of CLIP
    pipeline,
)

import retico_core
from retico_vision.vision import ExtractedObjectsIU, ObjectFeaturesIU
from PIL import Image

class ObjectFeaturesExtractor(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "HF Vision models Object Features extractor"

    @staticmethod
    def description():
        return "Module for extracting visual features from images."

    @staticmethod
    def input_ius():
        return [ExtractedObjectsIU]

    @staticmethod
    def output_iu():
        return ObjectFeaturesIU
    
    def build_image_feature_pipe(self, ckpt: str, pool: bool = True):
        """
        Returns a `pipeline("image-feature-extraction")` that accepts a PIL image and
        gives you either pooled global embeddings (`pool=True`) or the raw
        last-layer hidden states (`pool=False`, default).
        Works for CLIP, ResNet-50, ViT, Swin, etc.
        """
        cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)

        if cfg.model_type == "clip":                 # multimodal checkpoints
            model = CLIPVisionModel.from_pretrained(ckpt)
        else:                                        # ResNet, ViT, ConvNeXt, …
            model = AutoModel.from_pretrained(ckpt)

        img_proc = AutoImageProcessor.from_pretrained(ckpt)

        return pipeline(
            task="image-feature-extraction",
            model=model,
            image_processor=img_proc,
            pool=pool,
        )
    
    def __init__(self, model_name = "openai/clip-vit-base-patch32", top_objects=1, timeout=0.5, **kwargs):
        """
        Initialize the ObjectFeaturesExtractor with a specified model name and number of top objects to extract features from.

        Args:
            model_name (str): The name of the model to use for feature extraction. Default is "openai/clip-vit-base-patch32".
            top_objects (int): The number of top objects to extract features from. Default is 1.
            timeout (float): The timeout for the extractor thread to wait for new input. Default is 0.5 seconds.
        """
        super().__init__(**kwargs)
        
        self._extractor_thread_active = False
        self.timeout = timeout
        self.model_name = model_name
        self.top_objects = top_objects
        self.queue = deque(maxlen=1)

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)
    
    def flatten_features(self, features):
        """
        Flatten the features to a 1D tensor.
        """
        if isinstance(features, list):
            # convert to torch tensor and print shape
            out = []
            for i, feature in enumerate(features):
                out += self.flatten_features(feature)
            return out
        else:
            return [features]
    
    def _extractor_thread(self):
        while self._extractor_thread_active:
            if len(self.queue) == 0:
                time.sleep(self.timeout)
                continue

            input_iu = self.queue.popleft()
            image = input_iu.image
            detected_objects = input_iu.extracted_objects
            object_features = {}

            for i, obj in enumerate(detected_objects):
                # sub_img = self.get_clip_subimage(image, obj)
                if i>=self.top_objects: break
                # if it's not a valid object, skip
                sub_img = detected_objects[obj]
                # print(f"Processing object {i} with image {sub_img}")
                
                # Extract features
                # skip if image has width or height of 0
                # if sub_img is not a PIL image, skip it
                if (not isinstance(sub_img, Image.Image) or
                        sub_img.width == 0 or sub_img.height == 0):
                    # print(f"Skipping object {i} with zero width or height or not a PIL image.")
                    object_features[i] = []
                    continue
                
                # print(f"Extracting features for object {i} with size {sub_img.size}")
                features = self.pipeline(sub_img)
                if isinstance(features, list):
                    # convert to torch tensor and print shape
                    # print(f"Extracted feat+ures for object {i} with shape {torch.tensor(features[0]).shape}")
                    features = self.flatten_features(features[0])
                    object_features[i] = features

            output_iu = self.create_iu(input_iu)
            output_iu.set_object_features(image, object_features)
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        self._extractor_thread_active = True
        self.pipeline = self.build_image_feature_pipe(self.model_name)
        threading.Thread(target=self._extractor_thread).start()
    
    def shutdown(self):
        self._extractor_thread_active = False
import sys
import time
import torch

from torch import nn

from habitat import Config
from habitat.tasks.nav.object_nav_task import task_cat2hm3dcat40, mapping_mpcat40_to_goal21
from habitat_baselines.il.env_based.policy.rednet import load_rednet

from mmdet.apis import init_detector, inference_detector # needs MMDetection library


class InstanceDetector(nn.Module):
    r"""A wrapper over object detector network.
    """

    def __init__(self, model_config: Config, device):
        super().__init__()
        self.model_config = model_config
        self.detector = None
        self.device = device

        if model_config.DETECTOR.name == "mask_rcnn":
            # Default to Mask RCNN predictor
            self.detector = init_detector(model_config.DETECTOR.config_path, model_config.DETECTOR.checkpoint_path, device)

        self.eval()

    def convert_to_semantic_mask(self, preds):
        pass

    def forward(self, observations):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        rgb_obs = observations["rgb"]

        if self.model_config.DETECTOR.name == "mask_rcnn":
            x = inference_detector(self.detector, rgb_obs)
        return x

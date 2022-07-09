import sys
import time
import torch
import numpy as np

from torch import nn
import torchvision.transforms.functional as TF
from typing import Optional

from habitat import Config
from habitat.tasks.nav.object_nav_task import task_cat2hm3dcat40, mapping_mpcat40_to_goal21
from habitat_baselines.il.env_based.policy.rednet import load_rednet
from habitat_baselines.il.env_based.policy.coco_segmentation.coco_segmentation_model import COCOSegmentationModel

# import detectron2.data.transforms as T

# from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog
# from detectron2.modeling import build_model
# from detectron2.checkpoint import DetectionCheckpointer

# sys.path.insert(0, '/srv/flash1/rramrakhya6/spring_2022/Detic/')
# sys.path.insert(0, '/srv/flash1/rramrakhya6/spring_2022/Detic/third_party/CenterNet2/')
# from centernet.config import add_centernet_config
# from detic.config import add_detic_config
# from detic.modeling.utils import reset_cls_test
# from detic.modeling.text.text_encoder import build_text_encoder


class Transform:
    is_random: bool = False

    def apply(self, x: torch.Tensor):
        raise NotImplementedError

    def __call__(
        self,
        x: torch.Tensor,
        T: Optional[int] = None,
        N: Optional[int] = None,
        V: Optional[int] = None,
        skip_one_T: bool = True,
    ):
        if not self.is_random:
            return self.apply(x)

        if None in (T, N, V):
            return self.apply(x)

        if T == 1 and skip_one_T:
            return self.apply(x)

        # put environment (n) first
        _, A, B, C = x.shape
        x = torch.einsum("tnvabc->ntvabc", x.view(T, N, V, A, B, C)).flatten(1, 2)

        # apply the same transform within each environment
        x = torch.cat([self.apply(imgs) for imgs in x])

        # put timestep (t) first
        _, A, B, C = x.shape
        x = torch.einsum("ntvabc->tnvabc", x.view(N, T, V, A, B, C)).flatten(0, 2)

        return x


class ResizeTransform(Transform):
    def __init__(self, size):
        self.size = size

    def apply(self, x):
        x = x.permute(0, 3, 1, 2)
        return x


def get_transform(name, size):
    if name == "resize":
        return ResizeTransform(size)


class SemanticPredictor(nn.Module):
    r"""A wrapper over semantic predictor network.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, model_config: Config, device):
        super().__init__()
        self.model_config = model_config
        self.semantic_predictor = None
        self.device = device

        if model_config.SEMANTIC_PREDICTOR.name == "detic":
            # cfg = get_cfg()
            # add_centernet_config(cfg)
            # add_detic_config(cfg)
            # cfg.merge_from_file(model_config.SEMANTIC_PREDICTOR.DETIC.config)
            # cfg.MODEL.WEIGHTS = model_config.SEMANTIC_PREDICTOR.DETIC.pretrained_weights
            # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model_config.SEMANTIC_PREDICTOR.DETIC.score_threshold  # set threshold for this model
            # cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
            # cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
            # # self.semantic_predictor = DefaultPredictor(cfg)
            # self.semantic_predictor = build_model(cfg)
            # checkpointer = DetectionCheckpointer(
            #     self.semantic_predictor
            # ).load(cfg.MODEL.WEIGHTS)
            # self.text_encoder = build_text_encoder(pretrain=True)
            # self.text_encoder.eval()

            # self.aug = T.ResizeShortestEdge(
            #     [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            # )
            # self.transforms = get_transform("resize", cfg.INPUT.MIN_SIZE_TEST)
            # # Reset visualization threshold
            # output_score_threshold = 0.3
            # for cascade_stages in range(len(self.semantic_predictor.roi_heads.box_predictor)):
            #     self.semantic_predictor.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold

            # self.metadata = MetadataCatalog.get(str(time.time()))

            # self.metadata.thing_classes = "chair,bed,plant,toilet,tv_monitor,sofa".split(',')
            # classifier = self.get_clip_embeddings(self.metadata.thing_classes)
            # num_classes = len(self.metadata.thing_classes)
            # reset_cls_test(self.semantic_predictor, classifier, num_classes)
            pass

        elif model_config.SEMANTIC_PREDICTOR.name == "coco_maskrcnn":
            self.semantic_predictor = COCOSegmentationModel(
                sem_pred_prob_thr=0.9,
                sem_gpu_id=(-1 if self.device == torch.device("cpu") else self.device.index),
                visualize=False
            )

        else:
            # Default to RedNet predictor
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt=model_config.SEMANTIC_PREDICTOR.REDNET.pretrained_weights,
                resize=True,  # since we train on half-vision
                num_classes=model_config.SEMANTIC_PREDICTOR.REDNET.num_classes
            )
            self.semantic_predictor.eval()

        self.task_cat2hm3dcat40_t = torch.tensor(task_cat2hm3dcat40).to(self.device)
        self.mapping_mpcat40_to_goal = np.zeros(
            max(
                max(mapping_mpcat40_to_goal21.keys()) + 1,
                50,
            ),
            dtype=np.int8,
        )

        for key, value in mapping_mpcat40_to_goal21.items():
            self.mapping_mpcat40_to_goal[key] = value
        self.mapping_mpcat40_to_goal = torch.tensor(self.mapping_mpcat40_to_goal, device=self.device)

        self.eval()

    def get_clip_embeddings(self, vocabulary, prompt='a '):
        texts = [prompt + x for x in vocabulary]
        emb = self.text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

    def convert_to_semantic_mask(self, preds):
        pass

    def forward(self, observations):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        rgb_obs = observations["rgb"]
        depth_obs = observations["depth"]

        if self.model_config.SEMANTIC_PREDICTOR.name == "detic":
            # rgb_obs_tfms = self.transforms(rgb_obs)

            # inputs = [{"image": img.squeeze(0)} for img in rgb_obs_tfms]
            # preds = self.semantic_predictor(inputs)
            # masks = []
            # for pred in preds:
            #     if len(pred["instances"]) > 0:
            #         cat_to_hm3d_cat_ids = self.mapping_mpcat40_to_goal[self.task_cat2hm3dcat40_t[pred["instances"].pred_classes].to(self.device)]
            #         semantic_mask = torch.max(pred["instances"].pred_masks.long() * cat_to_hm3d_cat_ids.unsqueeze(-1).unsqueeze(-1), 0)[0].unsqueeze(0)
            #         masks.append(semantic_mask)
            #     else:
            #         zeros_sem_mask = torch.zeros((rgb_obs_tfms[0].shape[1], rgb_obs_tfms[0].shape[2])).unsqueeze(0).to(self.device)
            #         masks.append(zeros_sem_mask)
            # x = torch.cat(masks, dim=0)
            pass

        elif self.model_config.SEMANTIC_PREDICTOR.name == "coco_maskrcnn":
            semantic, _ = self.semantic_predictor.get_prediction(
                rgb_obs.cpu().numpy(),
                depth_obs.cpu().numpy()
            )
            x = torch.from_numpy(semantic).to(rgb_obs.device).long()

        else:
            x = self.semantic_predictor(rgb_obs, depth_obs)

            # Subtract 1 from class labels for THDA YCB categories
            if self.model_config.SEMANTIC_ENCODER.is_thda:
                x = x - 1

        return x

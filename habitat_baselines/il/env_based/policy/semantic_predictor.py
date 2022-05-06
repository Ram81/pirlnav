import sys
import time
import torch

from torch import nn
from habitat import Config

from habitat_baselines.il.env_based.policy.rednet import load_rednet

# import detectron2.data.transforms as T

# from detectron2.engine import DefaultPredictor
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

            # self.metadata = MetadataCatalog.get(str(time.time()))
            pass
        else:
            # Default to RedNet predictor
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt=model_config.SEMANTIC_PREDICTOR.REDNET.pretrained_weights,
                resize=True, # since we train on half-vision
                num_classes=model_config.SEMANTIC_PREDICTOR.REDNET.num_classes
            )
        self.semantic_predictor.eval()

        self.eval()
    
    def get_clip_embeddings(self, vocabulary, prompt='a '):
        texts = [prompt + x for x in vocabulary]
        emb = self.text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

    def forward(self, observations):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        rgb_obs = observations["rgb"]
        depth_obs = observations["depth"]

        # custom_vocabulary = observations["objectgoal_prompt"]

        if self.model_config.SEMANTIC_PREDICTOR.name == "detic":
            # self.metadata.thing_classes = "table".split(',')
            # classifier = self.get_clip_embeddings(self.metadata.thing_classes)
            # num_classes = len(self.metadata.thing_classes)
            # reset_cls_test(self.semantic_predictor, classifier, num_classes)
            # # Reset visualization threshold
            # output_score_threshold = 0.3
            # for cascade_stages in range(len(self.semantic_predictor.roi_heads.box_predictor)):
            #     self.semantic_predictor.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold
            
            # rgb_obs_tfms = self.aug.get_transform(rgb_obs).apply_image(rgb_obs)
            # rgb_obs_tfms = torch.as_tensor(rgb_obs_tfms.astype("float32").transpose(2, 0, 1))

            # inputs = {"image": rgb_obs_tfms, "height": rgb_obs.shape[1], "width": rgb_obs.shape[2]}
            # x = self.semantic_predictor([inputs])
            pass
        else:
            x = self.semantic_predictor(rgb_obs, depth_obs)
        return x

import argparse
import sys
import cv2
import glob
import numpy as np
import tempfile
from pathlib import Path
# import cog
import time
import torch

from PIL import Image

import detectron2.data.transforms as T
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Detic libraries
sys.path.insert(0, '/srv/flash1/rramrakhya6/spring_2022/Detic/')
sys.path.insert(0, '/srv/flash1/rramrakhya6/spring_2022/Detic/third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder

import habitat
from habitat.utils.visualizations.utils import make_rgb_palette
from habitat.tasks.nav.object_nav_task import task_cat2hm3dcat40, mapping_mpcat40_to_goal21
from habitat_baselines.il.env_based.policy.semantic_predictor import SemanticPredictor


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save("demos/" + file_name)

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class Predictor:
    def setup(self):
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file("Detic/configs/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml")
        cfg.MODEL.WEIGHTS = 'Detic/models/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        self.predictor = DefaultPredictor(cfg)
        self.BUILDIN_CLASSIFIER = {
            'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
            'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
            'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
            'coco': 'datasets/metadata/coco_clip_a+cname.npy',
        }
        self.BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
            'objects365': 'objects365_v2_val',
            'openimages': 'oid_val_expanded',
            'coco': 'coco_2017_val',
        }
        config = habitat.get_config("habitat_baselines/config/objectnav/il_ddp_objectnav.yaml")
        config.defrost()
        config.MODEL.SEMANTIC_PREDICTOR.name = "detic"
        config.freeze()
        self.device = torch.device("cuda", 0)
        self.semantic_predictor = SemanticPredictor(config.MODEL, self.device)
        self.metadata = MetadataCatalog.get(str(time.time()))
        self.metadata.thing_classes = "chair,bed,plant,toilet,tv_monitor,sofa".split(',')

        self.task_cat2hm3dcat40_t = torch.tensor(task_cat2hm3dcat40)
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

    def predict(self, paths, vocabulary, custom_vocabulary, output_path):
        images = []
        for path in paths:
            img = cv2.imread(str(path))
            print(img.dtype)
            images.append(cv2.imread(str(path)))
        image = torch.tensor(np.stack(images, axis=0)).to(self.device)
        observations = {
            "rgb": image,
            "depth": image
        }
        preds, outputs = self.semantic_predictor(observations)

        for i, _ in enumerate(preds):
            input_image = paths[i].split("/")[-1]
            colors = make_rgb_palette(20)
            mask = (colors[preds[i].long().cpu().numpy() % 20] * 255).astype(np.uint8)
            save_image(mask, "{}".format(input_image.replace("demo", "mask")))

            v = Visualizer(image[i].cpu().numpy()[:, :, ::-1], self.metadata)
            out = v.draw_instance_predictions(outputs[i]["instances"].to("cpu"))
            output_p = output_path.replace("out_1.png", input_image)
            cv2.imwrite(str(output_p), out.get_image()[:, :, ::-1])
        return output_path


def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


if __name__ == "__main__":
    args = get_parser().parse_args()
    predictor = Predictor()
    predictor.setup()

    images = glob.glob("demos/trajectory_1/*.png")
    predictor.predict(images, args.vocabulary, args.custom_vocabulary, args.output)

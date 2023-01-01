import warnings
from typing import List, Optional, Union

from habitat.config.default import _C as _HABITAT_CONFIG
from habitat.config.default import Config as CN
from habitat_baselines.config.default import _C as _BASE_CONFIG

CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# TASK CONFIG
# -----------------------------------------------------------------------------

# fmt:off
_TASK_CONFIG = _HABITAT_CONFIG.clone()
_TASK_CONFIG.defrost()

_TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 1000
_TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = 10000

_TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25
_TASK_CONFIG.SIMULATOR.TURN_ANGLE = 30
_TASK_CONFIG.SIMULATOR.TURN_ANGLE = 30
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = 128
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = 128
_TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]

_TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

_TASK_CONFIG.TASK.SIMPLE_REWARD = CN()
_TASK_CONFIG.TASK.SIMPLE_REWARD.TYPE = "SimpleReward"
_TASK_CONFIG.TASK.SIMPLE_REWARD.SUCCESS_REWARD = 2.5
_TASK_CONFIG.TASK.SIMPLE_REWARD.ANGLE_SUCCESS_REWARD = 2.5
_TASK_CONFIG.TASK.SIMPLE_REWARD.USE_DTG_REWARD = True
_TASK_CONFIG.TASK.SIMPLE_REWARD.USE_ATG_REWARD = True
_TASK_CONFIG.TASK.SIMPLE_REWARD.ATG_REWARD_DISTANCE = 1.0
_TASK_CONFIG.TASK.SIMPLE_REWARD.USE_ATG_FIX = True
_TASK_CONFIG.TASK.SIMPLE_REWARD.SLACK_PENALTY = -0.01

_TASK_CONFIG.TASK.SPARSE_REWARD = CN()
_TASK_CONFIG.TASK.SPARSE_REWARD.TYPE = "SparseReward"
_TASK_CONFIG.TASK.SPARSE_REWARD.SUCCESS_REWARD = 2.5

_TASK_CONFIG.TASK.ANGLE_TO_GOAL = CN()
_TASK_CONFIG.TASK.ANGLE_TO_GOAL.TYPE = "AngleToGoal"

_TASK_CONFIG.TASK.ANGLE_SUCCESS = CN()
_TASK_CONFIG.TASK.ANGLE_SUCCESS.TYPE = "AngleSuccess"
_TASK_CONFIG.TASK.ANGLE_SUCCESS.SUCCESS_ANGLE = 25.0
_TASK_CONFIG.TASK.ANGLE_SUCCESS.USE_TRAIN_SUCCESS = True

_TASK_CONFIG.TASK.IMAGEGOAL_ROTATION_SENSOR = CN()
_TASK_CONFIG.TASK.IMAGEGOAL_ROTATION_SENSOR.TYPE = "ImageGoalRotationSensor"
_TASK_CONFIG.TASK.IMAGEGOAL_ROTATION_SENSOR.SAMPLE_ANGLE = True

_TASK_CONFIG.TASK.TYPE = "Nav-v0"
_TASK_CONFIG.TASK.SUCCESS_DISTANCE = 0.1
_TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE = 0.1

_TASK_CONFIG.TASK.TRAIN_SUCCESS = CN()
_TASK_CONFIG.TASK.TRAIN_SUCCESS.TYPE = "TrainSuccess"
_TASK_CONFIG.TASK.TRAIN_SUCCESS.SUCCESS_DISTANCE = 0.1



# -----------------------------------------------------------------------------
# Behavior Metrics MEASUREMENT
# -----------------------------------------------------------------------------
_TASK_CONFIG.TASK.BEHAVIOR_METRICS = CN()
_TASK_CONFIG.TASK.BEHAVIOR_METRICS.TYPE = "BehaviorMetrics"

def get_task_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    config = _TASK_CONFIG.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

_CONFIG = _BASE_CONFIG.clone()
_CONFIG.defrost()

_CONFIG.VERBOSE = True

_CONFIG.BASE_TASK_CONFIG_PATH = "configs/tasks/imagenav.yaml"

_CONFIG.TRAINER_NAME = "mppo"
_CONFIG.ENV_NAME = "SimpleRLEnv"
_CONFIG.WANDB_PROJECT_NAME = "pirlnav"
_CONFIG.WANDB_NAME = "pirlnav"
_CONFIG.WANDB_MODE = "online"
_CONFIG.SENSORS = ["RGB_SENSOR"]

_CONFIG.VIDEO_OPTION = []
_CONFIG.VIDEO_DIR = "data/video"
_CONFIG.TENSORBOARD_DIR = "data/tensorboard"
_CONFIG.EVAL_CKPT_PATH_DIR = "data/checkpoints"
_CONFIG.CHECKPOINT_FOLDER = "data/checkpoints"
_CONFIG.LOG_FILE = "data/train.log"

_CONFIG.NUM_ENVIRONMENTS = 10
_CONFIG.LOG_INTERVAL = 10
_CONFIG.NUM_CHECKPOINTS = 100
_CONFIG.NUM_UPDATES = 20000
_CONFIG.TOTAL_NUM_STEPS = -1.0

_CONFIG.FORCE_TORCH_SINGLE_THREADED = False

_CONFIG.RUN_TYPE = None

_CONFIG.EVAL.SPLIT = "val"
_CONFIG.EVAL.USE_CKPT_CONFIG = True
_CONFIG.EVAL.EVAL_FREQ = 5

_CONFIG.MODEL = CN()
_CONFIG.MODEL.RGB_ENCODER = CN()
_CONFIG.MODEL.RGB_ENCODER.image_size = 256
_CONFIG.MODEL.RGB_ENCODER.backbone = "resnet50"
_CONFIG.MODEL.RGB_ENCODER.resnet_baseplanes = 32
_CONFIG.MODEL.RGB_ENCODER.vit_use_fc_norm = False
_CONFIG.MODEL.RGB_ENCODER.vit_global_pool = False
_CONFIG.MODEL.RGB_ENCODER.vit_use_cls = False
_CONFIG.MODEL.RGB_ENCODER.vit_mask_ratio = None
_CONFIG.MODEL.RGB_ENCODER.hidden_size = 512
_CONFIG.MODEL.RGB_ENCODER.use_augmentations = True
_CONFIG.MODEL.RGB_ENCODER.use_augmentations_test_time = True
_CONFIG.MODEL.RGB_ENCODER.randomize_augmentations_over_envs = False
_CONFIG.MODEL.RGB_ENCODER.pretrained_encoder = None
_CONFIG.MODEL.RGB_ENCODER.freeze_backbone = False
_CONFIG.MODEL.RGB_ENCODER.avgpooled_image = False
_CONFIG.MODEL.RGB_ENCODER.augmentations_name = "jitter+shift"
_CONFIG.MODEL.RGB_ENCODER.drop_path_rate = 0.0
_CONFIG.MODEL.RGB_ENCODER.normalize_visual_inputs = False

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    config = _CONFIG.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        for k, v in zip(opts[0::2], opts[1::2]):
            if k == "BASE_TASK_CONFIG_PATH":
                config.BASE_TASK_CONFIG_PATH = v

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)

    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    if config.NUM_PROCESSES != -1:
        warnings.warn(
            "NUM_PROCESSES is deprecated and will be removed in a future version."
            "  Use NUM_ENVIRONMENTS instead."
            "  Overwriting NUM_ENVIRONMENTS with NUM_PROCESSES for backwards compatibility."
        )

        config.NUM_ENVIRONMENTS = config.NUM_PROCESSES

    config.freeze()
    return config

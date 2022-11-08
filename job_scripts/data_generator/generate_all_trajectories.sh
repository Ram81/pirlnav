#!/bin/bash

EPISODE_ID="ziup5kvtCCR_26_[5.47169, 0.02122, 2.32604]_plant"

python examples/generate_top_down_maps.py \
--path data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/val/val.json.gz \
--evaluation-meta-path tb/objectnav_il/objectnav_hm3d/objectnav_hm3d_77k/rgb_ovrl/seed_1/hm3d_v0_1_0_evals/ckpt_best_val_replays/evaluation_meta.json \
--baseline il_hd --specific-episode-id "$EPISODE_ID"

python examples/generate_top_down_maps.py \
--path data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/val/val.json.gz \
--evaluation-meta-path tb/objectnav_il/objectnav_hm3d/objectnav_hm3d_s_path_240k/rgb_ovrl/seed_1/hm3d_v0_1_0_evals/ckpt_114_val_replays/evaluation_meta.json \
--baseline il_sp --specific-episode-id "$EPISODE_ID"

python examples/generate_top_down_maps.py \
--path data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/val/val.json.gz \
--evaluation-meta-path tb/objectnav_il/objectnav_hm3d/objectnav_hm3d_fe_70k_balanced/rgb_ovrl/seed_1/hm3d_v0_1_0_evals/ckpt_118_val_replays/evaluation_meta.json \
--baseline il_fe --specific-episode-id "$EPISODE_ID"

python examples/generate_top_down_maps.py \
--path data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/val/val.json.gz \
--evaluation-meta-path tb/objectnav_il_rl_ft/ddppo_hm3d_pt_77k/rgb_ovrl_with_augs/sparse_reward/hm3d_v0_1_0/seed_2/hm3d_v0_1_0_evals/ckpt_14_val_replays/evaluation_meta.json \
--baseline rl_ft_hd --specific-episode-id "$EPISODE_ID"

python examples/generate_top_down_maps.py \
--path data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/val/val.json.gz \
--evaluation-meta-path tb/objectnav_il_rl_ft/ddppo_hm3d_pt_s_path_240k/rgb_ovrl_with_augs/sparse_reward_ckpt_114/hm3d_v0_1_0/seed_1/v0_1_0_evals/ckpt_78_val_replays/evaluation_meta.json \
--baseline rl_ft_sp --specific-episode-id "$EPISODE_ID"

python examples/generate_top_down_maps.py \
--path data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/val/val.json.gz \
--evaluation-meta-path tb/objectnav_il_rl_ft/ddppo_hm3d_pt_fe_70k/rgb_ovrl_with_augs/sparse_reward_ckpt_118/hm3d_v0_1_0/seed_1/hm3d_v0_1_0_evals/ckpt_66_val_replays/evaluation_meta.json \
--baseline rl_ft_fe --specific-episode-id "$EPISODE_ID"

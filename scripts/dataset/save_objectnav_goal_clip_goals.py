import argparse
import clip
import torch
import numpy as np

from scripts.utils.utils import load_dataset


def save_clip_goal_embedding(backbone, dataset_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(backbone, device=device)

    dataset = load_dataset(dataset_path)
    objectnav_goals = dataset["category_to_task_category_id"]
    
    goal_embedding = [[]] * len(objectnav_goals.keys())
    print("ObjectGoals: {}".format(objectnav_goals))
    for object_name, object_goal_id in objectnav_goals.items():
        text = clip.tokenize([object_name], context_length=77).to(device)
        text_features = model.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.mean(0)

        goal_embedding[object_goal_id] = text_features.detach().cpu().numpy()
        print("ObjectGoal: {}, Embedding dim: {}".format(object_name, text_features.shape))
    
    goal_embedding = np.array(goal_embedding)
    np.save(output_path, goal_embedding)
    print("ObjectGoals cache: {}".format(goal_embedding.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", type=str, default="ViT-B/32"
    )
    parser.add_argument(
        "--path", type=str, default="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/train/train.json.gz"
    )
    parser.add_argument(
        "--output-path", type=str, default=""
    )
    args = parser.parse_args()
    save_clip_goal_embedding(args.backbone, args.path, args.output_path)

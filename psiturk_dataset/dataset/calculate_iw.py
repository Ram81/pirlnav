import argparse
import gzip
import glob
import json

from tqdm import tqdm
from habitat.datasets.utils import VocabFromText
from psiturk_dataset.utils.utils import load_dataset, load_json_dataset, write_json, write_gzip, load_vocab


VISITED_POINT_DICT = {}
assignment_dict = {}
episode_ids = ['A1L3937MY09J3I:3Z7EFSHGNBH1CG7U84ECI5ABGYYCX5','A1ZE52NWZPN85P:3C6FJU71TSWMYFE4ZRLEVP3QQU9YUY','A2CWA5VQZ6IWMQ:3YGXWBAF72KAEEJKOTC7LUDDNIP4C8','APGX2WZ59OWDN:358010RM5GWXBPDUZL9H8XY015IVXR']


def caclulate_inflections(episode):
    inflections = 1
    reference_replay = episode["reference_replay"]
    for i in range(len(reference_replay) - 1):
        if reference_replay[i]["action"] != reference_replay[i - 1]["action"]:
            inflections += 1
    return inflections, len(reference_replay)


def save_meta_for_analysis(meta, path):
    write_json(meta, path)


def get_action_time(action, task="objectnav"):
    tt = 0
    if task == "rearrangement":
        if "TURN" in action or "LOOK" in action:
            tt = 0.000358 * (5**2) + 0.108 * 5 + 2.23
        elif "MOVE" in action or "GRAB" in action:
            tt = 4.2 * 0.15 + 0.362
    else:
        if "TURN" in action or "LOOK" in action:
            tt = 0.000358 * (30**2) + 0.108 * 30 + 2.23
        elif "MOVE" in action:
            tt = 4.2 * 0.25 + 0.362
    return tt


def calculate_inflection_weight(path, stats_path):
    data = load_dataset(path)

    episodes = data["episodes"]
    inflections = 0
    total_actions = 0
    total_episodes = len(episodes)
    data = {
        "episode_length": [],
        "action_frequency": {},
    }
    ep_success_count = 0
    actions_lt_2k = 0
    ep_lt_than_2k = 0
    wall_clock_time = 0
    for episode in tqdm(episodes):
        num_inflections, num_actions = caclulate_inflections(episode)
        data["episode_length"].append(num_actions)
        reference_replay = episode["reference_replay"]
        for i in range(len(reference_replay)):
            action = reference_replay[i]["action"]
            if action not in data["action_frequency"]:
                data["action_frequency"][action] = 0
            data["action_frequency"][action] += 1
            wall_clock_time += get_action_time(action, "rearrangement")
        
        if len(reference_replay) <= 1500:
            ep_success_count += 1
        if len(reference_replay) <= 1800:
            actions_lt_2k += num_actions
            ep_lt_than_2k += 1

        inflections += num_inflections
        total_actions += num_actions
    
    save_meta_for_analysis(data, stats_path)

    print("Total episodes: {}".format(len(episodes)))
    print("Episodes less than 1.5k: {}".format(ep_lt_than_2k))
    print("Inflection weight: {}".format(total_actions / inflections))
    print("Average episode length: {}".format(total_actions / total_episodes))
    print("Total actions: {}".format(total_actions))
    print("Total success: {}".format(ep_success_count / len(episodes)))
    print("Total actions less than 2k: {}".format(actions_lt_2k))
    print("Wall clock time in hours: {}".format(wall_clock_time / 3600))

    instructions = convert_instruction_tokens(episodes)
    print("Num of distinct instructions: {}".format(len(set(instructions))))
    write_json(list(set(instructions)), "data/hit_approvals/instructions.json")


def calculate_inflection_weight_objectnav(path, stats_path):
    files = glob.glob(path + "*.json.gz")
    inflections = 0
    total_actions = 0
    total_episodes = 0
    ep_lt_than_1k = 0
    ep_lt_than_500 = 0

    data_stats = {
        "episode_length": [],
        "action_frequency": {},
        "object_frequency": {}
    }
    mx = 0
    actions_lt_2k = 0
    wall_clock_time = 0
    for file_path in tqdm(files):
        data = load_dataset(file_path)

        episodes = data["episodes"]
        for episode in episodes:
            num_inflections, num_actions = caclulate_inflections(episode)

            data_stats["episode_length"].append(num_actions)
            reference_replay = episode["reference_replay"]
            mx = max(mx, len(reference_replay))
            for i in range(len(reference_replay)):
                action = reference_replay[i]["action"]
                if action not in data_stats["action_frequency"]:
                    data_stats["action_frequency"][action] = 0
                data_stats["action_frequency"][action] += 1
                wall_clock_time += get_action_time(action, "objectnav")

            object_category = episode['object_category']
            if object_category not in data_stats['object_frequency']:
                data_stats['object_frequency'][object_category] = 0
            data_stats['object_frequency'][object_category] += 1

            inflections += num_inflections
            total_episodes += 1
            total_actions += num_actions
            if len(reference_replay) > 2500:
                ep_lt_than_1k += 1
            
            if len(reference_replay) < 2000:
                actions_lt_2k += num_actions
            
            if len(reference_replay) <= 500:
                ep_lt_than_500 += 1

    save_meta_for_analysis(data_stats, stats_path)

    print("Total episodes: {} - {}".format(total_episodes, mx))
    print("Total episodes less than 1k and 0.5k: {} -- {}".format(ep_lt_than_1k, ep_lt_than_500))
    print("Inflection weight: {}".format(total_actions / inflections))
    print("Average episode length: {}".format(total_actions / total_episodes))
    print("Total actions: {}".format(total_actions))
    print("Total actions less than 2k: {}".format(actions_lt_2k))
    print("Wall clock time in hours: {}".format(wall_clock_time / 3600))


def convert_instruction_tokens(episodes):
    vocab = load_vocab()
    instruction_vocab = VocabFromText(
        sentences=vocab["sentences"]
    )

    instructions = []
    for episode in episodes:
        instruction = instruction_vocab.token_idx_2_string(episode["instruction"]["instruction_tokens"])
        instructions.append(instruction)
    return instructions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/hit_approvals/dataset/backup/train.json.gz"
    )
    parser.add_argument(
        "--task", type=str, default="rearrangement"
    )
    parser.add_argument(
        "--stats", type=str, default="data/episodes/objectnav_sample/stats.json"
    )
    args = parser.parse_args()

    if args.task == "rearrangement":
        calculate_inflection_weight(args.path, args.stats)
    else:
        calculate_inflection_weight_objectnav(args.path, args.stats)


if __name__ == "__main__":
    main()

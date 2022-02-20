import json
import glob


def read_episodes(path):
    f = open(path, "r")
    return json.loads(f.read())


def read_files(path, data):
    file_paths = glob.glob(replay_path + "/*.mp4")
    data = []
    for ep in data["episodes"]:
        episode_id = ep["episode_id"]
        video_available = False
        for file_path in file_paths:
            if episode_id in file_path:
                video_available = True
                data.append({
                    "episodeId": episode_id,
                    "sceneId": ep["scene_id"],
                    "video": "{}".format(file_path),
                    "task": ep["instruction"]["instruction_text"],
                    "episodeLength": len(ep["reference_replay"])
                })
                break
    
    return data


if __name__ == "__main__":
    episodes = read_episodes("data/")
    
    

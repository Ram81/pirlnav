import argparse
import habitat

from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image

config = habitat.get_config("configs/tasks/objectnav_mp3d.yaml")


def make_videos(observations_list, output_prefix, ep_id):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix)


def run_reference_replay(
    cfg, num_episodes=None, output_prefix=None
):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS  
    with habitat.Env(cfg) as env:
        total_success = 0
        spl = 0

        num_episodes = min(num_episodes, len(env.episodes))
        print("Replaying {}/{} episodes".format(num_episodes, len(env.episodes)))
        for ep_id in range(num_episodes):
            observation_list = []
            env.reset()

            step_index = 1
            total_reward = 0.0
            episode = env.current_episode

            for data in env.current_episode.reference_replay[step_index:]:
                action = possible_actions.index(data.action)
                action_name = env.task.get_action_name(
                    action
                )

                observations = env.step(action=action)

                info = env.get_metrics()
                frame = observations_to_image({"rgb": observations["rgb"]}, info)
                frame = append_text_to_image(frame, "Find and go to {}".format(episode.object_category))

                observation_list.append(frame)
                if action_name == "STOP":
                    break
            make_videos([observation_list], output_prefix, ep_id)
            print("Total reward for trajectory: {}".format(total_reward))

            if len(episode.reference_replay) <= 500:
                total_success += info["success"]
                spl += info["spl"]

        print("SPL: {}, {}, {}".format(spl/num_episodes, spl, num_episodes))
        print("Success: {}, {}, {}".format(total_success/num_episodes, total_success, num_episodes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="demo"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.path
    cfg.freeze()

    run_reference_replay(
        cfg,
        num_episodes=args.num_episodes,
        output_prefix=args.output_prefix
    )


if __name__ == "__main__":
    main()

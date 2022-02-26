import argparse
import habitat

from habitat.utils.visualizations.utils import observations_to_image, images_to_video

config = habitat.get_config("configs/tasks/pickplace_mp3d.yaml")


def make_videos(observations_list, output_prefix, ep_id):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix)


def run_reference_replay(cfg, restore_state=False, step_env=False, num_episodes=None, output_prefix=None):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        total_success = 0
        total_spl = 0

        num_episodes = min(num_episodes, len(env.episodes))

        print("Replaying {}/{} episodes".format(num_episodes, len(env.episodes)))
        for ep_id in range(num_episodes):
            observation_list = []
            env.reset()
            i = 0
            success = 0
            step_index = 1

            for data in env.current_episode.reference_replay[step_index:]:
                action = possible_actions.index(data.action)

                if step_env:
                    observations = env.step(action=action)
                elif not restore_state:
                    observations = env.step(action=action, replay_data=data)
                else:
                    agent_state = data.agent_state
                    sensor_states = data.agent_state.sensor_data
                    object_states = data.object_states
                    observations = env._sim.get_observations_at(agent_state.position, agent_state.rotation, sensor_states, object_states)

                info = env.get_metrics()
                frame = observations_to_image({"rgb": observations["rgb"]}, {})

                observation_list.append(frame)
                i+=1
            
            total_success += info["success"]
            total_spl += info["spl"]

            make_videos([observation_list], output_prefix, ep_id)

        print("Average success: {} - {} - {}".format(total_success / num_episodes, total_success, num_episodes))
        print("Average SPL: {} - {} - {}".format(total_spl / num_episodes, total_spl, num_episodes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="demo"
    )
    parser.add_argument(
        "--restore-state", dest='restore_state', action='store_true'
    )
    parser.add_argument(
        "--step-env", dest='step_env', action='store_true'
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
        args.restore_state,
        args.step_env,
        num_episodes=args.num_episodes,
        output_prefix=args.output_prefix
    )

if __name__ == "__main__":
    main()

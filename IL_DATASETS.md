## Imitation Learning Datasets


### Packaging IL Datasets

`habitat-web-baseline` provides a generic implementation for replaying demonstrations from a replay buffer. Here's an example of packaging an dataset of shortest path demonstrations for ObjectNav for imitation learning:

1. Create a shortest path follower agentthat solves the ObjectNav task by following shortest path to the goal. 

    ```
    follower = ShortestPathFollower(
        env._sim, goal_radius, False
    )
    
    episode = env.current_episode
    goal_position = get_closest_goal(episode, env.sim)

    while not env.episode_over:
        next_action = follower.get_next_action(
            goal_position
        )
        observations = env.step(next_action)
    ```

2. For each episode, maintain a replay buffer with action executed and environment state metadata for each step. Store the step metadata in following schema

    ```
    {
        "action": "STOP",
        "agent_state": {
            "position": [],
            "rotation": [],
            "sensor_state": {
                "rgb": {
                    "position": [],
                    "rotation": [],
                }
            }
        }
    }
    ```

3. Save the replay buffer for each episode in key `reference_replay`.

4. Save the generated demonstrations dataset.


For a full example of a shortest path demonstration dataset generator see the example script in `examples/objectnav_shortest_path_generator.py`.

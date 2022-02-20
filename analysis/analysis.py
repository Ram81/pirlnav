import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

episode_file = open("../data/hit_data/visualization/hits.json")
data = json.loads(episode_file.read())

print("Number of episodes {}".format(len(data["episodes"])))

episode_data = []
columns = []
columns.append("episode_id")
columns.append("total_actions")

for episode in data["episodes"]:
#     if len(episode["reference_replay"]) > 4500:
#         print(episode["episode_id"], episode["episode_id"].split(":")[0])
#         continue
    entry = {}
    entry["episode_id"] = episode["episode_id"]
    entry["total_actions"] = len(episode["reference_replay"])


    for replay_data in episode["reference_replay"]:
        action = replay_data["action"]
        if not action in entry:
            entry[action] = 0
        entry[action] += 1

        if action not in columns:
            columns.append(action)
    episode_data.append(entry)


df = pd.DataFrame(episode_data, columns=columns)
df.fillna(0, inplace=True)

df["agent_actions"] = df["total_actions"] - df["stepPhysics"]

sns.histplot(df["total_actions"].values, bins=13)
plt.gca().set(title='Episode length histogram', ylabel='Frequency', xlabel='Episode length')
plt.savefig("action_distribution.jpg")

sns.histplot(data=df[["stepPhysics", "agent_actions"]], element="step")
plt.gca().set(ylabel='Frequency', xlabel='Num actions')
plt.savefig("env_and_agent_action_distribution.jpg")

plt.hist(df["agent_actions"].values, bins=13)
plt.gca().set(title='Agent action histogram', ylabel='Frequency', xlabel='Num actions')
plt.savefig("agent_action_dist.jpg")

sns.lineplot(data=df[["stepPhysics", "agent_actions"]])
plt.gca().set(title='Action per episode', ylabel='Frequency', xlabel='Episode id')
plt.savefig("action_per_episode.jpg")

sns.lineplot(data=df[["turnLeft", "turnRight", "lookDown", "lookUp", "moveForward", "moveBackward", "grabReleaseObject"]])
plt.gca().set(title='Agent action per episode', ylabel='Frequency', xlabel='Episode id')
plt.savefig("agent_action_per_episode.jpg")


from tqdm import tqdm
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import tensorflow as tf

def evaluate(model, num_episodes):
    """
    Evaluates a reinforcement learning agent.

    Args:
        model: The trained RL model.
        env: The environment to evaluate the model on.
        num_episodes: The number of episodes to run for evaluation.

    Returns:
        A tuple containing the mean reward and the mean elapsed time per episode.
    """

    env_id = "highway-fast-v0"
    env = make_vec_env(env_id)
    episode_rewards = []
    episode_times = []
    print(f"evaluating Model on {num_episodes} episodes ...")
    for _ in tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        total_reward = 0
        start_time = 0
        current_time = 0

        while not done:
          action, _states = model.predict(obs, deterministic=True)
          obs, reward, done, info = env.step(action)
          total_reward += reward
          current_time += 1

        episode_rewards.append(total_reward)
        episode_times.append(current_time - start_time)

    mean_reward = np.mean(episode_rewards)
    mean_time = np.mean(episode_times)
    std_reward = np.std(episode_rewards)
    std_time = np.std(episode_times)
    print(f"\n{'-'*50}\nResults :\n\t- Mean Reward: {mean_reward:.3f} ± {std_reward:.2f} \n\t- Mean elapsed Time per episode: {mean_time:.3f} ± {std_time:.2f}\n{'-'*50}")
    return mean_reward, mean_time
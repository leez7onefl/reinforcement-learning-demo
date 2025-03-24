import streamlit as st
import torch
from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import TRPO, QRDQN, ARS
from utils.train import train_model
from stable_baselines3.common.env_util import make_vec_env
import highway_env  # noqa: F401

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_hyperparameters(algo):
    common_params = {
        "policy_kwargs": dict(net_arch=[256, 256]),
        "learning_rate": 3e-4,
        "device": device,
    }

    if algo == "DQN":
        return {
            **common_params,
            "buffer_size": 15000,
            "learning_starts": 200,
            "batch_size": 32,
            "gamma": 0.8,
            "train_freq": 1,
            "gradient_steps": 1,
            "target_update_interval": 50,
        }
    elif algo == "PPO":
        return {
            **common_params,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    elif algo == "A2C":
        return {
            **common_params,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    elif algo == "QRDQN":
        return {
            **common_params,
            "learning_rate": 1e-3,
            "buffer_size": 1000000,
            "learning_starts": 500,
            "batch_size": 256,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
        }
    elif algo == "TRPO":
        return {
            **common_params,
            "learning_rate": 1e-3,
            "gamma": 0.99,
        }
    elif algo == "ARS":
        return {
            **common_params,
        }


st.title("Reinforcement Learning Training")

env_id = "highway-fast-v0"
env = make_vec_env(env_id, n_envs=4)

algo = st.selectbox("Choose an algorithm", ["DQN", "PPO", "A2C", "QRDQN", "TRPO", "ARS"])
total_steps = st.slider("Total Training Steps", min_value=10, max_value=10000, value=1000, step=10)

hyperparams = get_hyperparameters(algo)
if st.checkbox("Use default hyperparameters", value=True):
    st.write(f"#### {algo} Hyperparameters")
    st.write(hyperparams)

else:
    st.write("#### Custom Hyperparameters")
    st.text_area("Hyperparameters", value=hyperparams)

if st.button("Start Training"):

    with st.spinner(f"Training {algo} model..."):
        if algo == "DQN":
            model_class = DQN
        elif algo == "PPO":
            model_class = PPO
        elif algo == "A2C":
            model_class = A2C
        elif algo == "QRDQN":
            model_class = QRDQN
        elif algo == "TRPO":
            model_class = TRPO
        elif algo == "ARS":
            model_class = ARS

        model = train_model(
            model_class=model_class,
            env=env,
            model_name=f"{algo.lower()}_highway",
            total_timesteps=total_steps,
            **hyperparams
        )
        model.save(f"models/highway_{algo.lower()}_final")
    st.success(f"{algo} model trained and saved!")

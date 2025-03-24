import streamlit as st
import os
from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import TRPO, QRDQN, ARS
from utils.evaluate import evaluate

st.title("Reinforcement Learning Eval")

models_folder = "models/"

while not os.path.exists(models_folder):
    st.error("The 'models/' folder does not exist yet. Please train some models first.")
    st.stop()

model_files = [f for f in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, f))]

if not model_files:
    st.warning("No models found")

selected_models = st.multiselect("Choose model(s)", model_files)

num_episodes = st.number_input("Number of episodes : ", step=1, value=30)

if st.button("start evaluation"):
    st.write("You've selected the following models:")
    for model in selected_models:
        st.write(model)

    for model_name in selected_models:
        model_path = "models/"+model_name
        st.write(f"\nEvaluating model {model_name}...")
        clean_name = model_name.replace("highway_", "").replace("_final.zip", "").upper()
        
        if clean_name == "DQN":
            model = DQN.load(model_path)
            model_reward = evaluate(model, num_episodes)
        
        elif clean_name == "PPO":
            model = PPO.load(model_path)
            model_reward = evaluate(model, num_episodes)

        elif clean_name == "A2C":
            model = A2C.load(model_path)
            model_reward = evaluate(model, num_episodes)

        elif clean_name == "QRDQN":
            model = QRDQN.load(model_path)
            model_reward = evaluate(model, num_episodes)

        elif clean_name == "TRPO":
            model = TRPO.load(model_path)
            model_reward = evaluate(model, num_episodes)

        elif clean_name == "ARS":
            model = ARS.load(model_path)
            model_reward = evaluate(model, num_episodes)

        st.write(f"Total reward for model {model_name}: {model_reward}")

    # st.write("\nModel Comparison:")
    # for model_name in selected_models:
        
    #     odel_path = "models/"+model_name
    #     model_reward = evaluate(model_path)
    #     st.write(f"{model_name}: {model_reward}")



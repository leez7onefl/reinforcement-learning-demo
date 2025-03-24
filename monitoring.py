import os
import subprocess
import streamlit as st
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_tensorboard(logdir, port=6006):
    if not is_port_in_use(port):
        command = f'tensorboard --logdir="{logdir}" --port={port} --host=localhost'
        subprocess.Popen(command, shell=True)

logdir = "logs/dqn_highway/DQN_7"
tensorboard_port = 6006

if "tensorboard_started" not in st.session_state:
    start_tensorboard(logdir, tensorboard_port)
    st.session_state.tensorboard_started = True

tensorboard_url = f"http://localhost:{tensorboard_port}"

st.title("TensorBoard Monitoring")
st.markdown(
    f'''
    <div style="display: flex; justify-content: center;">
        <iframe src="{tensorboard_url}" width="1200" height="800"></iframe>
    </div>
    ''',
    unsafe_allow_html=True
)

st.write("TensorBoard is running from logs in the 'highway' directory.")
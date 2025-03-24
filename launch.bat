@echo off
call rl_project_venv\Scripts\activate.bat

REM Start the Streamlit app
start cmd /k "streamlit run main.py"

REM Start TensorBoard in a new window
start cmd /k "tensorboard --logdir=highway-fast-v0"

call deactivate
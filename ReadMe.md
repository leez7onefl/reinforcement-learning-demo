# Deep Reinforcement Learning Algorithms

This repository contains the final project for the "*Reinforcement Learning*" course at EFREI, 2025.

## Project Overview

## Project Structure

```
rl_final/
│
├── notebook/                # Notebook final
    ├── TH_Final_Project_student.ipynb
│
├── utils/                 
    ├── evaluate.py
    ├── train.py
    ├── video_utils.py
│
├── models/
    ├── highway_dqn_final.zip
    ├── highway_ppo_final.zip
    ├── highway_a2c_final.zip
    ├── highway_qrdqn_final.zip
    ├── highway_trpo_final.zip
    ├── highway_ars_final.zip
│
├── main.py                  # Main streamlit, organize the pages
├── training.py              # training part 
├── evaluation.py            # evaluation part
├── monitoring.py            # tensorboard embedded
│
├── build.bat                # Batch file for building project
├── launch.bat               # Batch file to launch Streamlit 
├── README.md                # Project documentation file
├── requirements.txt         # Python dependencies
├── toucan.jpg               # A chill toucan named Barry, he is there for test purposes (please say hi)
```
---
## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit

### Installation

1. Clone the repository :
    ```bash
    git clone https://github.com/leez7onefl/GenAIComputerVisionLab4
    ```

2. Install dependencies into a virtual environment :
    ```bash
    built.bat
    ```

### Run Streamlit App

```bash
launch.bat
```

---

## UI 

#### Monitoring

Embedded Tensorboard in streamlit
![monitoring](https://github.com/user-attachments/assets/824570d3-20e7-4fdf-8a7f-7399df5a913b)

#### Training

![train1](https://github.com/user-attachments/assets/8b05d053-3db3-4d88-a4cc-620c3f286195)
<ins>6 algorithms choices available</ins>
![train2](https://github.com/user-attachments/assets/d0789a73-c365-4158-aeb3-e9454977d387)
<ins>All hyperparameters are editable</ins>
![train3](https://github.com/user-attachments/assets/c77803e3-7d64-42dc-b396-44adb09e2b51)
<ins>training seen in console</ins>
![train4](https://github.com/user-attachments/assets/761be9e5-8908-4d98-9b03-458d3d4e4bc3)

#### Eval

![eval1](https://github.com/user-attachments/assets/2abf2534-acf4-4fae-91ea-29f9754305c9)
<ins>All trained models are displayed</ins>
![eval2](https://github.com/user-attachments/assets/78324f6e-cd61-478e-b346-ab453a7a26da)
<ins>Multiple choice for comparison</ins>
![eval3](https://github.com/user-attachments/assets/87453b11-5adb-4b3f-b52f-7f46dd7c1771)
<ins>Eval seen from console</ins>
![eval4](https://github.com/user-attachments/assets/cd8839c8-dfec-47c4-bfbc-266225b0e0dd)

![eval5](https://github.com/user-attachments/assets/be82dfaf-100c-43c8-80bf-04775ec1c89c)

---
## Contribution

Feel free to contribute by creating issues or submitting pull requests.

## Contact

For any queries, please contact me at leonard.gonzalez@outlook.fr.

---

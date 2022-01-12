import subprocess



if __name__ == '__main__':
    python = "C:/Users/sukhoon.jung/PycharmProjects/stable-baselines3/venv/Scripts/python.exe"
    models = ["DDPG", "SAC", "TD#"]
    procs = []
    for m in models:
        procs.append(subprocess.Popen([python,  "RUN.py", m], stdout=subprocess.PIPE))





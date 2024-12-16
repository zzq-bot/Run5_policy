# Run5_policy

---

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)

This repo contains the policy training code of our team (Run5).

---



## 1. Installation

Please install the environment in the following way:

```
conda create -n ubi_rl python=3.11.10
conda activate ubi_rl   
pip install -r requirements.txt
cd RL/grid_env
pip install -e .
```

## 2. Training

Please conduct training in the following way (you can adjust the hyperparameters as needed).

```
bash train.sh
```

One case log is placed in `./case_log` (has been trained for about 300M sampling steps, ~6 hours, reach score 78, will be updated later)


## 3. Evaluation

You can change checkpoint as you need (see `./policy_ckpt`).

```
bash eval.sh
```
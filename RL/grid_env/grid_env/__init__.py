from gymnasium.envs.registration import register

register(
    "grid_env/EvalGridWorld-v0",
    entry_point="grid_env.envs:EvalGridWorldEnvV0",
)

register(
    "grid_env/GoGridWorld-v0",
    entry_point="grid_env.envs:GoGridWorldEnvV0",
)


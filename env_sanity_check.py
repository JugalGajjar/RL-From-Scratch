import torch, gymnasium as gym, numpy as np
print("Torch:", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
env = gym.make("CartPole-v1")
obs, info = env.reset()
print("Env OK, obs shape:", np.array(obs).shape)
env.close()
print("Environment sanity check passed.")
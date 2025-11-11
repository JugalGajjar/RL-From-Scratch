import numpy as np
from sklearn import datasets
import torch
import gymnasium as gym

# Check numpy
print("NumPy:", np.__version__)
a = np.array([1, 2, 3])
print("NumPy array:", a)

# Check scikit-learn
iris = datasets.load_iris()
print("Iris dataset shape:", iris.data.shape)

# Check PyTorch
print("Torch:", torch.__version__)
if torch.backends.mps.is_available():
    x = torch.tensor([1.0, 2.0, 3.0], device='mps')
    print("Tensor on MPS device:", x)
elif torch.cuda.is_available():
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    print("Tensor on CUDA device:", x)
else:
    x = torch.tensor([1.0, 2.0, 3.0])
    print("Tensor on CPU device:", x)

# Check Gymnasium
print("Gymnasium:", gym.__version__)
env = gym.make("CartPole-v1")
obs, info = env.reset()
print("Env OK, obs shape:", np.array(obs).shape)
env.close()

print("Environment sanity check passed.")
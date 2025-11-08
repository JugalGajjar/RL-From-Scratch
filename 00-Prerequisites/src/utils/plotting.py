import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    sns.set_context("talk")
    sns.set_style("whitegrid")

def line(x, y, xlabel="", ylabel="", title=""):
    set_style()
    plt.figure(figsize=(7,4))
    plt.plot(x, y)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.show()

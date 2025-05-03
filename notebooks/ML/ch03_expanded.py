from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load synthetic neuron dataset
def load_synthetic_neurons_3class(n_samples_per_class=50, overlap=1.0, as_frame=False, random_state=42):
    rng = np.random.default_rng(random_state)
    params = [
        {"label": "Interneuron", "mu": [0.25, 40, 1.2, 0.8], "sigma": [0.05, 5, 0.2, 0.1]},
        {"label": "Pyramidal",   "mu": [0.6,  80, 0.6, 0.4], "sigma": [0.1,  8, 0.1, 0.1]},
        {"label": "Bursting",    "mu": [0.45, 65, 0.8, 0.6], "sigma": [0.08, 6, 0.15, 0.1]}
    ]
    X, y, labels = [], [], []
    for i, p in enumerate(params):
        features = rng.normal(loc=p["mu"], scale=np.array(p["sigma"]) * overlap, size=(n_samples_per_class, 4))
        X.append(features)
        y.extend([i] * n_samples_per_class)
        labels.extend([p["label"]] * n_samples_per_class)
    X = np.vstack(X)
    y = np.array(y)
    if as_frame:
        df = pd.DataFrame(X, columns=['spike_width', 'amplitude', 'upstroke_downstroke', 'symmetry_index'])
        df['neuron_type'] = labels
        return X, y, df
    return X, y


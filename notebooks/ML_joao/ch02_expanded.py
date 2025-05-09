import numpy as np
import pandas as pd

# Define a function to load the synthetic neuron dataset
def load_synthetic_neuron(as_frame=False):
    """
    Returns:
        X: np.ndarray of shape (100, 4), synthetic spike waveform features
        y: np.ndarray of shape (100,), 0 for interneurons, 1 for pyramidal
        df: pd.DataFrame (if as_frame=True), full labeled dataset
    """
    rng = np.random.default_rng(42)
    labels = np.array([0] * 50 + [1] * 50)

    interneurons = np.column_stack([
        rng.normal(0.25, 0.05, 50),   # spike width
        rng.normal(40, 5, 50),        # amplitude
        rng.normal(1.2, 0.2, 50),     # upstroke/downstroke
        rng.normal(0.8, 0.1, 50)      # symmetry index
    ])

    pyramidal = np.column_stack([
        rng.normal(0.6, 0.1, 50),     # spike width
        rng.normal(80, 8, 50),        # amplitude
        rng.normal(0.6, 0.1, 50),     # upstroke/downstroke
        rng.normal(0.4, 0.1, 50)      # symmetry index
    ])

    X = np.vstack([interneurons, pyramidal])
    y = labels

    columns = ['spike_width', 'amplitude', 'upstroke_downstroke', 'symmetry_index']
    df = pd.DataFrame(X, columns=columns)
    df['neuron_type'] = np.where(y == 0, 'Interneuron', 'Pyramidal')

    if as_frame:
        return X, y, df
    return X, y


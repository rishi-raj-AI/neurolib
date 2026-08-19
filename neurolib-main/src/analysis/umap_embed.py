import importlib

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.analysis.utils import assign_conditions_to_bins

SAMPLE_BINS = 500
# Default number of PCA components to use before UMAP
DEFAULT_PCA_COMPONENTS = 12


class _LazyUMAPModule:
    """Load umap.umap_ only when UMAP functionality is first accessed."""

    def __init__(self):
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module("umap.umap_")
        return self._module

    def __getattr__(self, name):
        return getattr(self._load(), name)


# Keep a module-level object so existing tests and callers can patch
# src.analysis.umap_embed.umap.UMAP while still avoiding the heavy UMAP
# import during normal module import.
umap = _LazyUMAPModule()


def compute_umap_embedding(df: pd.DataFrame, n_components: int = 3,
                          bin_size_ms: int = 10, random_state: int = 42, sample_mode: bool = False) -> dict:
    """
    Compute UMAP embedding combined with PCA and low-memory configurations
    for usage on low-memory machines.

    1) Check if DataFrame is empty
    2) Convert time data to datetime format
    3) Calculate time range and create bins
    4) Extract unique channels and create neural activity DataFrame
    5) Bin neural activity by channel
    6) Assign conditions to bins
    7) Apply sample mode if enabled
    8) Scale neural activity data
    9) Apply PCA for dimensionality reduction
    10) Configure and apply UMAP for low memory usage
    11) Return embedding results or error information

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter n_components: int
                    :parameter bin_size_ms: int
                    :parameter random_state: int
                    :parameter sample_mode: bool

                Returns:
                    :return embedding: dict
    """
    # Check if DataFrame is empty
    if df.empty:
        return {
            'error': 'Empty data',
            'n_components': n_components,
            'pca_components': DEFAULT_PCA_COMPONENTS,
            'pca_explained_variance_ratio': None,
            'bin_size_ms': bin_size_ms,
            'binned_data': pd.DataFrame(),
            'conditions': np.array([])
        }

    data = df.copy()

    data['Time'] = pd.to_datetime(data['Time'], utc=True).dt.tz_localize(None)

    start_time = data['Time'].min()
    end_time = data['Time'].max()
    total_ms = (end_time - start_time).total_seconds() * 1000
    num_bins = int(np.ceil(total_ms / bin_size_ms))

    bin_edges = np.linspace(0, total_ms, num_bins + 1)

    channels = data['channel'].unique()
    neural_activity = pd.DataFrame(index=range(num_bins), columns=channels)

    for channel in channels:
        times_ms = (data.loc[data['channel'] == channel, 'Time'] - start_time).dt.total_seconds() * 1000

        counts, _ = np.histogram(times_ms, bins=bin_edges)

        neural_activity[channel] = counts

    # Pass timestamps to Util func.
    bin_edges_dt = [start_time + pd.Timedelta(ms, unit='ms') for ms in bin_edges]
    binned_conditions = assign_conditions_to_bins(data, start_time, bin_edges_dt)

    # Sample mode
    if sample_mode:
        neural_activity = neural_activity.iloc[:SAMPLE_BINS]
        binned_conditions = binned_conditions[:SAMPLE_BINS]

    neural_activity_array = neural_activity.fillna(0).values
    neural_activity_scaled = StandardScaler().fit_transform(neural_activity_array)

    try:
        # Apply PCA <-- Needed for low memory usage
        pca_n_components = min(DEFAULT_PCA_COMPONENTS, neural_activity_scaled.shape[1])
        pca = PCA(n_components=pca_n_components, svd_solver='randomized', random_state=random_state)
        X_reduced = pca.fit_transform(neural_activity_scaled)

        # Configure UMAP for low memory usage. The module is loaded lazily
        # when UMAP is accessed for the first time.
        model = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_epochs=50,
            metric='euclidean',
            low_memory=True
        )

        embedding = model.fit_transform(X_reduced)

        return {
            'embedding': embedding,
            'n_components': n_components,
            'pca_components': pca_n_components,
            'pca_explained_variance_ratio': pca.explained_variance_ratio_,
            'bin_size_ms': bin_size_ms,
            'binned_data': neural_activity,
            'conditions': binned_conditions
        }
    except Exception as e:
        error_response = {
            'error': str(e),
            'n_components': n_components,
            'pca_components': DEFAULT_PCA_COMPONENTS,
            'pca_explained_variance_ratio': None,
            'bin_size_ms': bin_size_ms,
            'binned_data': neural_activity,
            'conditions': binned_conditions
        }

        return error_response

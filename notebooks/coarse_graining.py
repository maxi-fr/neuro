import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    from neuro.connectome import load_connectome

    return PCA, load_connectome, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load simulation data
    """)
    return


@app.cell
def _(load_connectome, np):
    sim_path = r"results\simulation_2026-06-21_15-28-47\log.npz"
    connectome = load_connectome()
    with np.load(sim_path) as data:
        data["universal_y_mea"].T  # (n_sensors, n_samples)
        x_data = data["universal_x"]
        data["universal_u"]

    node_output = x_data[:, 1] - x_data[:, 2]
    return connectome, node_output


@app.cell
def _(PCA, node_output):
    n_components = 20

    # Initialize PCA.
    # We transpose X so the shape is (n_time_steps, n_original_nodes).
    # This finds the dominant spatial patterns across the 74 nodes over time.
    pca = PCA(n_components=n_components)

    # X_reduced is the time series of your new 10 macro-nodes
    # Shape: (5000, 10). We transpose it back to (10, 5000) for standard conventions.
    pca.fit_transform(node_output).T
    return n_components, pca


@app.cell
def _(n_components, np, pca, plt):
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Variance explained by {n_components} virtual nodes: {explained_variance * 100:.2f}%")

    # Optionally, plot the explained variance to visualize the drop-off
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o", linestyle="--")
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Variance Explained")
    plt.grid(visible=True)
    plt.show()
    return


@app.cell
def _(connectome, n_components, np, pca):
    # pca.components_ contains the spatial weights mapping the 10 nodes to the 74 nodes.
    # Shape is (10, 74). To map back from 10 to 74, we use its transpose (74, 10).
    w_inverse = pca.components_.T

    gain_original = connectome.gain

    # Multiply the original 62x74 projection matrix by the 74x10 transformation matrix
    # The result is your new 62x10 projection matrix for the reduced model.
    gain_coarse = gain_original @ w_inverse
    np.savez(f"scripts\\configs\\reduced_eeg_gain_{n_components}.npz", gain_coarse)

    # Verify shapes
    print(f"Original Projection Matrix shape: {gain_original.shape}")
    print(f"Reduced Projection Matrix shape:  {gain_coarse.shape}")
    return (gain_coarse,)


@app.cell
def _(connectome, np, plt):
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _vmax = float(np.abs(connectome.gain).max())
    _im = _ax.imshow(connectome.gain, cmap="RdBu_r", aspect="auto", vmin=-_vmax, vmax=_vmax)
    _ax.set_title("EEG leadfield matrix L (62 channels x 76 regions)")
    _ax.set_xlabel("region")
    _ax.set_ylabel("channel")
    _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(gain_coarse, np, plt):
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _vmax = float(np.abs(gain_coarse).max())
    _im = _ax.imshow(gain_coarse, cmap="RdBu_r", aspect="auto", vmin=-_vmax, vmax=_vmax)
    _ax.set_title("EEG leadfield matrix L (62 channels x 76 regions)")
    _ax.set_xlabel("region")
    _ax.set_ylabel("channel")
    _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()

from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from anna_cookie import PROJECT_DIR


def plot_dendrogram(linkage, threshold, figsize=(12, 20), p=5, truncate_mode="level"):
    path = PROJECT_DIR / "outputs" / "figures"

    """the function plots dendrogram from the linkage array.

    Args:
        linkage (np.array): linkage array
        threshold (list of float): [0.1, 0.2, 0.3], list of float number indicating the threshold where additional lines are plotted
        figsize (tuple, optional): figsize. Defaults to (12, 20).
        p (int, optional): show only the last p merged cluster, used for 'truncate mode' in dendrogram. Defaults to 5.
        truncate_mode (str, optional): truncate the tree when the dendrogram is large. Defaults to 'level'.
    """

    fig, axes = plt.subplots(figsize=figsize)

    dendrogram(
        linkage,
        truncate_mode=truncate_mode,  # show only the last p merged clusters
        p=p,
        # leaf_label_func=llf,
        leaf_font_size=12.0,
        show_contracted=True,
        orientation="right",  # to get a distribution impression in truncated branches
    )
    top, bottom = plt.ylim()

    for i in range(len(threshold)):
        plt.plot([threshold[i], threshold[i]], [top, bottom], "k--", alpha=0.6)

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Hamming Distance")
    plt.ylabel("Cluster Index or (cluster node size)")
    plt.savefig(path / "dendro")


def plot_TSNE_level(
    distance, df, cluster="isco_label", level=4, perplexity=20, random_state=4
):
    path = PROJECT_DIR / "outputs" / "figures"
    """[summary]

    Args:
        X ([type]): [description]
        df ([type]): [description]
        cluster (str, optional): [description]. Defaults to 'isco_label'.
        level (int, optional): [description]. Defaults to 4.
        perplexity (int, optional): [description]. Defaults to 20.
    """

    X_embedded = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state
    ).fit_transform(distance)
    cat = df[cluster].apply(lambda x: x[:level])
    print(cat)
    groups = (
        pd.DataFrame(X_embedded, columns=["x", "y"])
        .assign(category=np.array(cat))
        .groupby(by="category")
    )
    fig, ax = plt.subplots(figsize=(20, 12))
    for name, points in groups:
        ax.scatter(points.x, points.y, label=name)
    plt.savefig(path / "tsne")

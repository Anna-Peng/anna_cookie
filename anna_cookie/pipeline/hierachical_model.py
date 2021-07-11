import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anna_cookie import PROJECT_DIR

from scipy.cluster.hierarchy import dendrogram, fcluster, single, complete, linkage
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score


def Agg_model():
    """Summary:
    The function read preprocessed pickled input
    and create an Agglomerative Clustering Model

    Parameters: for the AgglomerativeClustering(*arg) were:
    distance_threshold = 0
    n_clusters = None
    affinity = 'precomputed' (using precomputed distance matrix)
    linkage = 'complete'

    Returns:
        [model]: Agglomerative Clustering Model
        [dist_mat]: distance matrix using hamming distance
    """

    filepath = PROJECT_DIR / "outputs" / "data"
    occu_with_ESCO_input = filepath / "occu_with_ESCO_processed.pkl"
    skill_input = filepath / "skills_en_processed.pkl"

    df_oc = pd.read_pickle(occu_with_ESCO_input)
    df_sk = pd.read_pickle(skill_input)

    vect = CountVectorizer(
        vocabulary=df_sk.index, analyzer=lambda x: x, binary=True
    )  # disable built-in analyzer
    X = df_oc.hasEssentialSkill
    count_mat = vect.fit_transform(
        X
    ).toarray()  # m(occupation) x n(skill) get the count vector

    dist_v = pdist(
        count_mat, "hamming"
    )  # this returns a vector of distance, use hamming which focuses on the difference between highy similar vectors
    dist_mat = squareform(dist_v)  # this turns distance vector into symmetrical matrix

    model = AgglomerativeClustering(
        distance_threshold=0,
        n_clusters=None,
        affinity="precomputed",
        linkage="complete",
    ).fit(dist_mat)
    return model, dist_mat


def get_linkage(model):
    """The function generate the linkage distance matrix to children nodes

    Args:
        model (AgglomerativeClustering model): An aggglomerative Clustering Model

    Returns:
        [linkage matrix]: n x 3 matrix containing
        children node, distance matrix, number of leaf nodes
    """
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix


def silhouette_k(distance_matrix, linkage_matrix, max_k=20):

    """[summary]

    Returns:
        [type]: [description]
    """
    scores = []
    for i in range(2, max_k + 1):
        clusters = fcluster(linkage_matrix, i, criterion="maxclust")
        score = silhouette_score(distance_matrix, clusters, metric="precomputed")
        print("Silhouette score with {} clusters:".format(i), score)
        scores.append(score)
    plt.title("Silhouette score vs. number of clusters")
    plt.xlabel("# of clusters")
    plt.ylabel("Score (higher is better)")
    plt.plot(np.arange(2, max_k + 1), scores)
    plt.show()
    return scores


def get_labels(linkage, thresholds):
    """[summary]

    Args:
        linkage ([type]): [description]
        thresholds ([type]): [description]

    Returns:
        [type]: [description]
    """
    labels = []
    for threshold in thresholds:
        label = fcluster(linkage, threshold, criterion="distance")
        labels.append(label)
    labels = np.stack(labels, axis=0)

    class_size = []
    for i in range(len(labels)):
        class_size.append(len(np.unique(labels[i])))

    return labels, class_size


def assign_label(labels, colname="cluster_label"):
    filepath = PROJECT_DIR / "outputs" / "data"
    occu_with_ESCO_input = filepath / "occu_with_ESCO_processed.pkl"

    df_oc = pd.read_pickle(occu_with_ESCO_input)
    isco_label = df_oc.iscoGroup.astype("str").apply(lambda x: list(map(int, str(x))))

    for ii, idx in enumerate(
        isco_label
    ):  # it turns out some of the ISCO has leading 0 missing
        if len(idx) != 4:
            isco_label[ii].insert(0, 0)
    df_oc["isco_label"] = tuple(isco_label)

    label_tuple = tuple(map(tuple, labels.T))
    df_oc[colname] = label_tuple

    return df_oc

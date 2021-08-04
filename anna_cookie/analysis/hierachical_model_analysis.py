#%%
from anna_cookie.pipeline import plot_dendro
from anna_cookie.pipeline.hierachical_model import HierModel

# Preprocess data and save into pickle files
#%%
# Construct Agglomerative Model and Distrance Matrix
model = HierModel()
Mod_, distance_ = model.agg_model()

#%%
linkage_ = model.get_linkage(Mod_)
# %%
thresholds_ = [0.01, 0.0082, 0.005, 0.003]
labels_, class_size_ = model.get_labels(linkage_, thresholds_)
#%%
# Assign cluster labels to the pickle files
df = model.assign_label(labels_, colname="cluster_label")
# %%
# Visualise Result
plot_dendro.plot_dendrogram(
    linkage_, thresholds_, figsize=(12, 20), p=5, truncate_mode="level"
)
# %%
plot_dendro.plot_TSNE_level(
    distance=distance_, df=df, cluster="isco_label", level=3, perplexity=10
)

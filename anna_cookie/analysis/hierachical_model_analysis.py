#%%
from anna_cookie.pipeline import preprocess, hierachical_model, plot_dendro

# Preprocess data and save into pickle files
#%%
# Script that preprocess Data into Pickle Files
preprocess.preprocess()

#%%
# Construct Agglomerative Model and Distrance Matrix
Mod_, distance_ = hierachical_model.Agg_model()

#%%
linkage_ = hierachical_model.get_linkage(model=Mod_)
# %%
thresholds_ = [0.01, 0.0082, 0.005, 0.003]
labels_, class_size_ = hierachical_model.get_labels(linkage_, thresholds_)
#%%
# Assign cluster labels to the pickle files
df = hierachical_model.assign_label(labels_, colname="cluster_label")
# %%
# Visualise Result
plot_dendro.plot_dendrogram(
    linkage_, thresholds_, figsize=(12, 20), p=5, truncate_mode="level"
)
# %%
plot_dendro.plot_TSNE_level(
    distance=distance_, df=df, cluster="isco_label", level=3, perplexity=10
)


4  # %%

# %%

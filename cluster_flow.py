"""Tokenises and ngrams documents.
- Run documents through a spacy pipeline
  (large english model with entity merging)
- Convert to bag of words
  (using either lemmatisation or by remapping certain entities to a common token
   - configured by `entity_mappings` param)
- Filter low frequency words
  (note: purely done for computational efficiency to avoid explosion of n-grams)
- Generate n-grams using statistical co-occurrence (gensim)
- Filter low and high frequency n-grams
- Filter very short n-grams or bi-grams that are purely stop words
"""

#%%
# import toolz.curried as t
from metaflow import FlowSpec, step, Parameter, IncludeFile, JSONType, conda_base
from anna_cookie.pipeline import plot_dendro
from anna_cookie.pipeline.hierachical_model import HierModel

#%%
@conda_base(
    libraries={
        "scikit-learn": ">=0.24",
        "scipy": ">=1.6",
        "pandas": ">=1.2",
        "numpy": ">=1.20",
        "matplotlib": ">=3.4",
        "pyyaml": ">=5.4",
        "python-dotenv": ">=0.17",
    },
    python="3.8",
)
class HierClusterFlow(FlowSpec):
    # for key,val in config['flows']['hierachical_model_analysis'].items():
    #     exec(key + '=val')

    # thresholds_ = Parameter(
    #     "thresholds_",
    #     help="Thresholds setting for each level of hierachical cluster",
    #     default=[0.01, 0.0082, 0.005, 0.003],
    # )
    # p = Parameter(
    #     "p",
    #     help="the depth of merged dendrogram",
    #     type=int,
    #     default=5,
    # )

    # figsize = Parameter(
    #     "figsize",
    #     help="figsize for plotting",
    #     default=(12, 20),
    # )

    # cluster = Parameter(
    #     "cluster",
    #     help="cluster labels used to denote 2D tsne",
    #     default="isco_label",  # or 'cluster_label'
    # )

    @step
    def start(self):
        """Load data and run the NLP pipeline, returning tokenised documents."""
        self.model = HierModel()
        self.Mod_, self.distance_ = self.model.agg_model()
        print("hierichical modelling")
        self.next(self.assign_label)

    @step
    def assign_label(self):
        self.thresholds_ = [0.01, 0.0082, 0.005, 0.003]
        linkage_ = self.model.get_linkage(self.Mod_)
        labels_ = self.model.get_labels(linkage_, self.thresholds_)[0]
        self.df = self.model.assign_label(labels_, colname="cluster_label")
        print("assign cluster labels to dataframe!")
        self.next(self.plots)

    @step
    def plots(self):
        plot_dendro.plot_dendrogram(
            self.linkage_,
            threshold=self.thresholds,
            figsize=(12, 20),  # self.figsize,
            p=5,  # self.p,
            truncate_mode="level",
        )
        plot_dendro.plot_TSNE_level(
            distance=self.distance_,
            df=self.df,
            cluster="isco_label",  # self.cluster,
            level=3,
            perplexity=10,
        )
        print("Plots!")
        self.next(self.end)

    @step
    def end(self):
        print("Phew!")


#%%

if __name__ == "__main__":
    HierClusterFlow(FlowSpec)
# %%

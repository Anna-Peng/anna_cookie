import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anna_cookie.getters import get_data as get
from anna_cookie.utils import process_string as ps
from anna_cookie import PROJECT_DIR


def preprocess():
    filepath = PROJECT_DIR / "outputs" / "data"
    df_sk = get.get_skill()
    df_oc = get.get_occupation()
    df_esco = pd.DataFrame.from_dict(get.get_ESCO(), orient="index")

    df_sk.set_index("preferredLabel", inplace=True)
    df_oc.set_index("preferredLabel", inplace=True)

    # Take away the /n in the altLabels column
    df_sk["altLabels"] = df_sk["altLabels"].apply(ps.split_string)
    df_oc["altLabels"] = df_oc["altLabels"].apply(ps.split_string)

    df_esco["preferredTerm_role"] = df_esco["preferredTerm"].apply(
        lambda x: x.get("roles")
    )
    df_esco["preferredTerm_label"] = df_esco["preferredTerm"].apply(
        lambda x: x.get("label")
    )

    def get_values(x):
        """
        This function gets the values under the key title
        """
        ls = []
        try:
            for i in range(len(x)):
                try:
                    ls.append(x[i].get("title"))
                except TypeError:
                    pass
        except TypeError:
            pass

        return ls

    # these two keys have values not saved as list so cannot apply the get_values fucntion directly (next cell)
    df_esco = (
        df_esco["_links"]
        .apply(pd.Series)
        .drop(["self", "regulatedProfessionNote"], axis=1)
    )
    for idx in df_esco.columns:
        df_esco[idx] = df_esco[idx].apply(get_values)

    df_final = df_oc.join(df_esco, how="left")

    df_final.to_pickle(filepath / "occu_with_ESCO_processed.pkl")
    df_sk.to_pickle(filepath / "skills_en_processed.pkl")

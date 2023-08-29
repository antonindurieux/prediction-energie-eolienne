import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_facet_scatter(
    production_df: pd.DataFrame, variable: str, n_samples: int, random_state: int
) -> None:
    """_summary_

    Args:
        production_df (pd.DataFrame): _description_
        variable (str): _description_
        n_samples (int): _description_
        random_state (int): _description_
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Annoying Seaborn warning

        g = sns.FacetGrid(
            data=production_df.sample(n_samples, random_state=random_state),
            col="Région",
            hue="Région",
            col_order=production_df["Région"].unique(),
            hue_order=production_df["Région"].unique(),
            palette="tab20",
            col_wrap=4,
        )
        g.map_dataframe(
            sns.scatterplot,
            x=variable,
            y="Eolien normalisé (%)",
            s=5,
            alpha=0.5,
        )
        plt.show()

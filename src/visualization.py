import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.pipeline


def plot_average_production_per_period(
    production_df: pd.DataFrame, resample_period: str, prod_variable: str
) -> None:
    """
    Plot a time-series of power production averaged by the defined period, for each region.

    Args:
        production_df (pd.DataFrame): Power production DataFrame.
        resample_period (str): Resample period ("D", "W" or "M").
        prod_variable (str): Production variable to represent.
    """
    average_prod_df = (
        production_df.set_index("Date")
        .groupby("Région")[prod_variable]
        .resample(resample_period)
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=average_prod_df,
        x="Date",
        y=prod_variable,
        hue="Région",
        palette="tab20",
    )
    plt.legend(bbox_to_anchor=(1.25, 1))
    period_title = {"M": "mensuelle", "W": "hebdomadaire", "D": "journalière"}
    plt.title(
        f"Evolution de la production éolienne {period_title[resample_period]} moyenne par région (variable '{prod_variable}')"
    )
    plt.show()


def plot_facet_scatter(
    production_meteo_df: pd.DataFrame,
    variable: str,
    target: str,
) -> None:
    """
    Multi-scatterplots grid by region of target vs. variable.

    Args:
        production_meteo_df (pd.DataFrame):  Merged DataFrames of power production and weather reports.
        variable (str): X-axis variable of scatterplots.
        target(str): Y-axis variable of scatterplots.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Annoying Seaborn warning

        g = sns.FacetGrid(
            data=production_meteo_df,
            col="Région",
            hue="Région",
            col_order=production_meteo_df["Région"].unique(),
            hue_order=production_meteo_df["Région"].unique(),
            palette="tab20",
            col_wrap=4,
        )
        g.map_dataframe(
            sns.scatterplot,
            x=variable,
            y=target,
            s=5,
            alpha=0.1,
        )
        [
            plt.setp(ax.set_xlabel(xlabel=f"{variable}"), fontsize=9)
            for ax in g.axes.flat
        ]
        [plt.setp(ax.set_ylabel(ylabel=target), fontsize=9) for ax in g.axes.flat]
        plt.show()


def plot_correlation_matrix(df: pd.DataFrame, numerical_variables: list[str]) -> None:
    """
    Plot a correlation matrix of numerical variables.

    Args:
        df (pd.DataFrame): DataFrames on which to compute correlations.
        numerical_features (list[str]): List of numerical variables.
    """
    corr = df[numerical_variables].corr()

    size = len(numerical_variables) // 3
    plt.figure(figsize=(size, size))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,  # type: ignore
        cbar_kws={"shrink": 0.5},
    )
    plt.title("Correlations des variables numériques")
    plt.show()


def plot_pred_evaluations(
    ml_pipelines: dict[str, sklearn.pipeline.Pipeline], y_pred: pd.DataFrame
) -> None:
    """
    Plot residue distributions of the predictions, and scatterplots of y_true vs. y_pred for each model.

    Args:
        ml_pipelines (dict[str, sklearn.pipeline.Pipeline]): Dictionary of machine learning pipelines.
        y_pred (pd.DataFrame): DataFrame of predictions.
    """
    _, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6), sharey="row", sharex="row")

    for i, model in enumerate(ml_pipelines.keys()):
        sns.histplot(y_pred, x=f"residus_eolien_{model}", bins=50, ax=axs[0, i])  # type: ignore
        sns.scatterplot(
            data=y_pred,
            x="Eolien (MW)",
            y=f"pred_eolien_{model}",
            s=5,
            alpha=0.1,
            ax=axs[1, i],
        )
        axs[0, i].set_xlabel("Résidus")
        axs[0, i].set_title(model)
        axs[1, i].set_xlim(-100, 5000)
        axs[1, i].set_ylim(-100, 5000)
        axs[1, i].plot((0, 5000), (0, 5000), c="r", linestyle="--")
        axs[1, i].set_xlabel("Eolien (MW): valeurs réelles")
        axs[1, i].set_ylabel("Valeurs prédites")

    plt.tight_layout()


def plot_pred_timeseries(
    y_pred: pd.DataFrame, resample_period: str, model_name: str
) -> None:
    """
    Multi-lineplots grid by region of prediction and true value time-series, averaged by resample_period.

    Args:
        y_pred (pd.DataFrame): DataFrame of predictions.
        resample_period (str): Resample period ("D", "W" or "M").
        model_name (str): Model name for which to plot predictions.
    """
    agg_y_pred = (
        y_pred.groupby("Région")[["Eolien (MW)", f"pred_eolien_{model_name}"]]
        .resample(resample_period)
        .mean()  # type: ignore
        .reset_index()
    )

    period_title = {"M": "mensuelle", "W": "hebdomadaire", "D": "journalière"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Annoying Seaborn warning

        g = sns.relplot(
            data=agg_y_pred,
            kind="line",
            col="Région",
            hue="Région",
            palette="tab20",
            col_wrap=4,
            x="Date",
            y="Eolien (MW)",
            legend=False,  # type: ignore
        )
        g.data = agg_y_pred.reset_index()
        g.map(
            sns.lineplot,
            "Date",
            f"pred_eolien_{model_name}",
            color="k",
            label="Prédictions",
            alpha=0.5,
        )
        g.add_legend()
        [plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
        plt.suptitle(
            f"Prédiction de la production éolienne {period_title[resample_period]} moyenne par région, sur la période test\n(Modèle {model_name})",
            y=1.03,
        )
        plt.show()

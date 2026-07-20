"""Analyse exploratoire des données Mpox.

Produit les statistiques descriptives, les fréquences des variables
qualitatives et la matrice de corrélation, et enregistre ces tableaux
dans results/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import RESULTS_DIR, TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def descriptive_statistics(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Calcule les statistiques descriptives des variables numériques.

    Inclut moyenne, médiane, écart-type, variance, min, max, quartiles
    et coefficient de variation (CV = écart-type / moyenne).
    """
    cols = NUMERIC_FEATURES + [TARGET]
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    desc = num.describe().T
    desc["mediane"] = num.median()
    desc["variance"] = num.var()
    desc["cv"] = (num.std() / num.mean()).replace([np.inf, -np.inf], np.nan)
    desc = desc.round(3)
    if save:
        desc.to_csv(RESULTS_DIR / "descriptive_statistics.csv")
    return desc


def categorical_frequencies(df: pd.DataFrame) -> dict:
    """Renvoie les fréquences (effectifs et pourcentages) des variables qualitatives."""
    out = {}
    for col in CATEGORICAL_FEATURES:
        if col in df:
            counts = df[col].value_counts()
            pct = (counts / len(df) * 100).round(2)
            out[col] = pd.DataFrame({"effectif": counts, "pourcentage": pct})
    return out


def correlation_matrix(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Calcule la matrice de corrélation de Pearson des variables numériques."""
    cols = NUMERIC_FEATURES + [TARGET]
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr().round(3)
    if save:
        corr.to_csv(RESULTS_DIR / "correlation_matrix.csv")
    return corr


def target_correlations(df: pd.DataFrame) -> pd.Series:
    """Corrélations des variables explicatives avec la cible, triées."""
    corr = correlation_matrix(df, save=False)[TARGET]
    return corr.drop(TARGET).sort_values(ascending=False)


if __name__ == "__main__":
    from .data_preprocessing import preprocess

    data = preprocess(save=False)
    print(descriptive_statistics(data, save=False))
    print("\nCorrélations avec la cible :\n", target_correlations(data))

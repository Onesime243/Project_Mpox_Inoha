"""Importation, nettoyage et préparation des données Mpox.

Ce module gère la lecture robuste du CSV, la vérification des types,
la détection des doublons et des valeurs manquantes, le contrôle de
l'identité de fuite (Taux_Positivite = Cas/Tests*100) et la sauvegarde
du jeu de données nettoyé.
"""

from __future__ import annotations

import pandas as pd

from . import (
    DATA_RAW,
    DATA_PROCESSED,
    TARGET,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def load_raw(path=DATA_RAW) -> pd.DataFrame:
    """Charge le fichier CSV brut de façon robuste.

    Parameters
    ----------
    path : Path or str
        Chemin vers le fichier CSV brut.

    Returns
    -------
    pandas.DataFrame
        Données brutes.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. "
            "Placez 'mpox_original.csv' dans data/raw/."
        ) from exc
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les noms de colonnes (espaces superflus)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def check_duplicates(df: pd.DataFrame) -> dict:
    """Compte les lignes entièrement dupliquées et les ID dupliqués."""
    return {
        "lignes_dupliquees": int(df.duplicated().sum()),
        "id_dupliques": int(df["ID"].duplicated().sum()) if "ID" in df else 0,
    }


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Renvoie un tableau du nombre et du pourcentage de valeurs manquantes."""
    n = len(df)
    miss = df.isna().sum()
    rep = pd.DataFrame(
        {"n_manquant": miss, "pct_manquant": (miss / n * 100).round(2)}
    )
    return rep.sort_values("n_manquant", ascending=False)


def verify_leakage_identity(df: pd.DataFrame) -> float:
    """Vérifie l'identité Taux_Positivite = Cas_Confirmes / Tests * 100.

    Returns
    -------
    float
        Écart absolu maximal entre la colonne observée et la valeur
        recalculée. Un écart proche de 0 confirme la relation déterministe
        (donc la nécessité d'exclure Taux_Positivite_pct des prédicteurs).
    """
    if not {"Cas_Confirmes", "Tests_Realises", "Taux_Positivite_pct"} <= set(df.columns):
        return float("nan")
    recompute = (df["Cas_Confirmes"] / df["Tests_Realises"] * 100).round(2)
    return float((recompute - df["Taux_Positivite_pct"]).abs().max())


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les variables dans les types attendus."""
    df = df.copy()
    for col in NUMERIC_FEATURES + [TARGET]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col in df:
            df[col] = df[col].astype("category")
    return df


def preprocess(save: bool = True) -> pd.DataFrame:
    """Pipeline complet de prétraitement.

    Charge, nettoie, vérifie et (optionnellement) sauvegarde le jeu
    de données prêt pour l'analyse.

    Parameters
    ----------
    save : bool
        Si True, enregistre le CSV nettoyé dans data/processed/.

    Returns
    -------
    pandas.DataFrame
        Jeu de données nettoyé.
    """
    df = load_raw()
    df = clean_column_names(df)
    df = coerce_types(df)

    # Traitement des valeurs manquantes : aucune n'est présente dans ce
    # jeu de données (vérifié), donc aucune imputation n'est nécessaire.
    df = df.dropna(subset=[TARGET])  # garde-fou : cible toujours définie

    if save:
        DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PROCESSED, index=False)
    return df


if __name__ == "__main__":
    data = load_raw()
    data = clean_column_names(data)
    print("Dimensions :", data.shape)
    print("\nDoublons :", check_duplicates(data))
    print("\nValeurs manquantes :\n", missing_report(data).head())
    print("\nÉcart max identité de fuite :", verify_leakage_identity(data))

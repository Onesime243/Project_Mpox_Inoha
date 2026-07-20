"""Construction des modèles de régression sur les données Mpox.

Contient :
  - la préparation de la matrice de design (encodage one-hot) ;
  - la séparation apprentissage / test ;
  - l'ajustement du modèle OLS complet (statsmodels) ;
  - une sélection backward fondée sur l'AIC ;
  - des modèles complémentaires : Poisson et binomial négatif (comptage) ;
  - Ridge, LASSO et Elastic Net via pipelines scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET,
)


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Construit la matrice X (one-hot pour les qualitatives) et la cible y.

    On applique un encodage one-hot avec suppression de la première
    modalité (drop_first=True) pour éviter le piège des variables
    indicatrices redondantes (colinéarité parfaite avec la constante).

    Returns
    -------
    pandas.DataFrame
        DataFrame contenant les prédicteurs encodés + la colonne cible.
    """
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    data = df[cols].copy()
    data = pd.get_dummies(
        data, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=float
    )
    return data


def split_data(data: pd.DataFrame, test_size: float = 0.2):
    """Sépare les données en apprentissage / test (aléatoire, reproductible).

    La séparation aléatoire est justifiée : la variable 'Semaine' est un
    simple index (1..N) et ne code pas une dynamique temporelle réelle ;
    les données sont donc traitées comme transversales.
    """
    y = data[TARGET]
    X = data.drop(columns=[TARGET])
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )


def _clean_names(X: pd.DataFrame) -> pd.DataFrame:
    """Rend les noms de colonnes compatibles avec les formules statsmodels."""
    X = X.copy()
    X.columns = (
        X.columns.str.replace(r"[ \-/]", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)
    )
    return X


def fit_ols(X_train: pd.DataFrame, y_train: pd.Series):
    """Ajuste le modèle OLS complet avec statsmodels (constante incluse).

    Returns
    -------
    statsmodels RegressionResults
    """
    X = _clean_names(X_train)
    X = sm.add_constant(X)
    X = X.astype(float)
    model = sm.OLS(y_train.astype(float), X).fit()
    return model


def backward_selection_aic(X_train: pd.DataFrame, y_train: pd.Series):
    """Sélection backward des variables fondée sur l'AIC.

    Retire itérativement la variable dont la suppression fait le plus
    baisser l'AIC, jusqu'à ce qu'aucune suppression ne l'améliore.

    Returns
    -------
    tuple(list, statsmodels result)
        Liste des variables retenues et modèle final ajusté.
    """
    X = _clean_names(X_train).astype(float)
    included = list(X.columns)

    def aic_of(cols):
        Xc = sm.add_constant(X[cols])
        return sm.OLS(y_train.astype(float), Xc).fit().aic

    current_aic = aic_of(included)
    improved = True
    while improved and len(included) > 1:
        improved = False
        scores = []
        for col in included:
            trial = [c for c in included if c != col]
            scores.append((aic_of(trial), col))
        best_aic, best_col = min(scores, key=lambda t: t[0])
        if best_aic < current_aic - 1e-6:
            included.remove(best_col)
            current_aic = best_aic
            improved = True

    Xc = sm.add_constant(X[included])
    final = sm.OLS(y_train.astype(float), Xc).fit()
    return included, final


def fit_count_models(df: pd.DataFrame):
    """Ajuste des modèles de comptage (Poisson et binomial négatif).

    Ces modèles sont fournis en complément car la cible est un comptage.
    On utilise la formule statsmodels sur l'ensemble des données.

    Returns
    -------
    dict
        {'poisson': result, 'negbin': result} avec AIC et BIC.
    """
    num = " + ".join(NUMERIC_FEATURES)
    cat = " + ".join(f"C({c})" for c in CATEGORICAL_FEATURES)
    formula = f"{TARGET} ~ {num} + {cat}"

    poisson = smf.glm(
        formula, data=df, family=sm.families.Poisson()
    ).fit()
    negbin = smf.glm(
        formula, data=df, family=sm.families.NegativeBinomial()
    ).fit()
    return {"poisson": poisson, "negbin": negbin}


def fit_regularized(X_train, y_train, X_test, y_test):
    """Ajuste Ridge, LASSO et Elastic Net via des pipelines scikit-learn.

    La standardisation est réalisée DANS le pipeline pour éviter toute
    fuite de données ; les hyperparamètres sont optimisés par validation
    croisée interne.

    Returns
    -------
    pandas.DataFrame
        R² test pour chaque modèle régularisé.
    """
    Xtr = _clean_names(X_train).astype(float)
    Xte = _clean_names(X_test).astype(float)

    models = {
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 50)),
        "LASSO": LassoCV(n_alphas=100, random_state=RANDOM_STATE, max_iter=10000),
        "ElasticNet": ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9], n_alphas=100,
            random_state=RANDOM_STATE, max_iter=10000,
        ),
    }
    rows = []
    for name, est in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
        pipe.fit(Xtr, y_train)
        r2 = pipe.score(Xte, y_test)
        rows.append({"modele": name, "R2_test": round(r2, 4)})
    return pd.DataFrame(rows)


def save_coefficients(model, path=RESULTS_DIR / "model_coefficients.csv"):
    """Enregistre coefficients, p-values et IC 95 % du modèle statsmodels."""
    summary = pd.DataFrame(
        {
            "coefficient": model.params,
            "erreur_std": model.bse,
            "t": model.tvalues,
            "p_value": model.pvalues,
            "IC_2.5%": model.conf_int()[0],
            "IC_97.5%": model.conf_int()[1],
        }
    ).round(4)
    summary.to_csv(path)
    return summary

"""Évaluation prédictive des modèles et validation croisée.

Calcule MSE, RMSE, MAE, R² sur apprentissage et test, applique une
validation croisée k-fold et enregistre les métriques et prédictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from . import RANDOM_STATE, RESULTS_DIR, TARGET


def _metrics(y_true, y_pred) -> dict:
    """Calcule MSE, RMSE, MAE et R² pour un jeu de prédictions."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": round(mse, 3),
        "RMSE": round(np.sqrt(mse), 3),
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def evaluate_ols(model, X_train, y_train, X_test, y_test,
                 save: bool = True) -> pd.DataFrame:
    """Évalue un modèle statsmodels sur apprentissage et test.

    Le modèle a été ajusté avec une constante ; on l'ajoute donc aux deux
    jeux avant la prédiction, en alignant les colonnes.
    """
    from .regression_model import _clean_names

    Xtr = sm.add_constant(_clean_names(X_train).astype(float))
    Xte = sm.add_constant(_clean_names(X_test).astype(float))
    # Aligner les colonnes du test sur celles du modèle.
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)

    pred_train = model.predict(Xtr)
    pred_test = model.predict(Xte)

    rows = [
        {"jeu": "apprentissage", **_metrics(y_train, pred_train)},
        {"jeu": "test", **_metrics(y_test, pred_test)},
    ]
    table = pd.DataFrame(rows)
    if save:
        table.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
        pd.DataFrame(
            {"y_reel": y_test.values, "y_predit": np.asarray(pred_test)}
        ).to_csv(RESULTS_DIR / "predictions.csv", index=False)
    return table


def cross_validation(X, y, cv: int = 10) -> dict:
    """Validation croisée k-fold (R²) avec une régression linéaire scikit-learn.

    KFold classique justifié : données transversales (pas de dépendance
    temporelle réelle).
    """
    from .regression_model import _clean_names

    Xc = _clean_names(X).astype(float)
    scores = cross_val_score(
        LinearRegression(), Xc, y, cv=cv, scoring="r2"
    )
    return {
        "cv_folds": cv,
        "r2_moyen": round(float(scores.mean()), 4),
        "r2_ecart_type": round(float(scores.std()), 4),
        "r2_par_pli": [round(float(s), 4) for s in scores],
    }

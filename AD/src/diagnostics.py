"""Diagnostics des hypothèses de la régression linéaire multiple.

Regroupe les tests de normalité des résidus, d'homoscédasticité,
d'indépendance des erreurs, de multicolinéarité (VIF) et la détection
des observations influentes (distance de Cook, leviers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import (
    OLSInfluence,
    variance_inflation_factor,
)
from statsmodels.stats.stattools import durbin_watson

from . import RESULTS_DIR


def normality_tests(model) -> dict:
    """Tests de normalité des résidus (Shapiro-Wilk et Jarque-Bera).

    Shapiro-Wilk est limité à un sous-échantillon de 5000 observations
    (contrainte de l'implémentation) ; sur de grands échantillons, il est
    très sensible et doit être lu avec les graphiques (QQ-plot).
    """
    resid = model.resid
    sample = resid.sample(min(5000, len(resid)), random_state=0)
    sw_stat, sw_p = stats.shapiro(sample)
    jb_stat, jb_p, skew, kurt = sm.stats.stattools.jarque_bera(resid)
    return {
        "shapiro_stat": round(sw_stat, 4),
        "shapiro_p": round(sw_p, 6),
        "jarque_bera_stat": round(jb_stat, 4),
        "jarque_bera_p": round(jb_p, 6),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
    }


def homoscedasticity_test(model, X_with_const) -> dict:
    """Test de Breusch-Pagan de l'homoscédasticité des résidus."""
    lm, lm_p, f, f_p = het_breuschpagan(model.resid, X_with_const)
    return {
        "breusch_pagan_LM": round(lm, 4),
        "breusch_pagan_p": round(lm_p, 6),
        "f_stat": round(f, 4),
        "f_p": round(f_p, 6),
    }


def independence_test(model) -> dict:
    """Statistique de Durbin-Watson (indépendance des erreurs).

    Valeur proche de 2 = pas d'autocorrélation de premier ordre.
    """
    return {"durbin_watson": round(durbin_watson(model.resid), 4)}


def linearity_reset(model) -> dict:
    """Test RESET de Ramsey (spécification / linéarité)."""
    try:
        reset = linear_reset(model, power=2, use_f=True)
        return {"reset_F": round(float(reset.fvalue), 4),
                "reset_p": round(float(reset.pvalue), 6)}
    except Exception as exc:  # noqa: BLE001
        return {"reset_F": np.nan, "reset_p": np.nan, "erreur": str(exc)}


def vif_table(X: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Facteur d'inflation de la variance (VIF) et tolérance par variable.

    On calcule le VIF sur les variables explicatives numériques (avec
    constante ajoutée pour un calcul correct). Un VIF > 5 (ou 10) signale
    une multicolinéarité à examiner.
    """
    Xc = sm.add_constant(X.astype(float))
    rows = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        v = variance_inflation_factor(Xc.values, i)
        rows.append({"variable": col, "VIF": round(v, 3),
                     "tolerance": round(1 / v, 4) if v else np.nan})
    table = pd.DataFrame(rows).sort_values("VIF", ascending=False)
    if save:
        table.to_csv(RESULTS_DIR / "vif_results.csv", index=False)
    return table


def influence_measures(model) -> pd.DataFrame:
    """Mesures d'influence : résidus studentisés, levier, distance de Cook."""
    infl = OLSInfluence(model)
    cooks = infl.cooks_distance[0]
    leverage = infl.hat_matrix_diag
    student = infl.resid_studentized_external
    df = pd.DataFrame(
        {
            "cook_distance": cooks,
            "leverage": leverage,
            "resid_studentise": student,
        }
    )
    return df


def influence_summary(model, n_obs: int) -> dict:
    """Résume la détection d'observations influentes.

    Seuil usuel de la distance de Cook : 4 / n.
    """
    infl = influence_measures(model)
    seuil = 4 / n_obs
    n_influentes = int((infl["cook_distance"] > seuil).sum())
    return {
        "seuil_cook": round(seuil, 6),
        "n_obs_influentes": n_influentes,
        "cook_max": round(float(infl["cook_distance"].max()), 4),
    }

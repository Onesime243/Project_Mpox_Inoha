"""Pipeline principal de l'analyse de régression linéaire multiple Mpox (RDC).

Exécute l'ensemble de l'analyse de bout en bout :
    1. prétraitement des données ;
    2. analyse exploratoire ;
    3. modèle OLS complet + sélection backward (AIC) ;
    4. diagnostics des hypothèses ;
    5. évaluation prédictive + validation croisée ;
    6. modèles complémentaires (Poisson, NB, Ridge, LASSO, Elastic Net) ;
    7. génération des figures et sauvegarde des résultats.

Usage :
    python main.py
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from src import RANDOM_STATE, RESULTS_DIR, TARGET, ensure_dirs
from src import data_preprocessing as prep
from src import diagnostics as diag
from src import exploratory_analysis as eda
from src import model_evaluation as evaluate
from src import regression_model as reg
from src import visualizations as viz

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)


def main() -> None:
    """Point d'entrée : exécute toute la chaîne d'analyse."""
    ensure_dirs()
    resume = {}

    # --- 1. Prétraitement ------------------------------------------------
    print(">> 1. Prétraitement")
    df = prep.preprocess(save=True)
    resume["dimensions"] = df.shape
    resume["doublons"] = prep.check_duplicates(df)
    resume["ecart_max_identite_fuite"] = prep.verify_leakage_identity(df)
    print("   Dimensions :", df.shape)
    print("   Écart identité de fuite :", resume["ecart_max_identite_fuite"])

    # --- 2. Analyse exploratoire ----------------------------------------
    print(">> 2. Analyse exploratoire")
    eda.descriptive_statistics(df, save=True)
    corr = eda.correlation_matrix(df, save=True)
    target_corr = eda.target_correlations(df)
    resume["top_correlations"] = target_corr.head(5).round(3).to_dict()
    top_vars = target_corr.abs().sort_values(ascending=False).head(3).index.tolist()

    # --- 3. Matrice de design + split -----------------------------------
    print(">> 3. Modélisation OLS")
    data = reg.build_design_matrix(df)
    X_train, X_test, y_train, y_test = reg.split_data(data, test_size=0.2)
    resume["n_train"], resume["n_test"] = len(X_train), len(X_test)

    ols = reg.fit_ols(X_train, y_train)
    resume["ols_complet"] = {
        "R2": round(ols.rsquared, 4),
        "R2_ajuste": round(ols.rsquared_adj, 4),
        "AIC": round(ols.aic, 1),
        "BIC": round(ols.bic, 1),
        "F_pvalue": float(ols.f_pvalue),
        "n_predicteurs": int(ols.df_model),
    }
    print("   OLS complet R2 =", resume["ols_complet"]["R2"])

    # Sélection backward (AIC) -> modèle final
    retained, final = reg.backward_selection_aic(X_train, y_train)
    resume["modele_final"] = {
        "n_variables": len(retained),
        "R2": round(final.rsquared, 4),
        "R2_ajuste": round(final.rsquared_adj, 4),
        "AIC": round(final.aic, 1),
        "BIC": round(final.bic, 1),
    }
    coefs = reg.save_coefficients(final)
    print("   Modèle final :", len(retained), "variables, R2 =",
          resume["modele_final"]["R2"])

    # --- 4. Diagnostics --------------------------------------------------
    print(">> 4. Diagnostics")
    from src.regression_model import _clean_names
    import statsmodels.api as sm

    X_final = _clean_names(X_train)[retained].astype(float)
    Xc = sm.add_constant(X_final)
    resume["normalite"] = diag.normality_tests(final)
    resume["homoscedasticite"] = diag.homoscedasticity_test(final, Xc)
    resume["independance"] = diag.independence_test(final)
    resume["linearite_reset"] = diag.linearity_reset(final)
    # VIF sur les variables numériques d'origine
    from src import NUMERIC_FEATURES
    vif = diag.vif_table(df[NUMERIC_FEATURES], save=True)
    resume["vif_max"] = float(vif["VIF"].max())
    infl = diag.influence_measures(final)
    resume["influence"] = diag.influence_summary(final, len(X_train))
    print("   Durbin-Watson :", resume["independance"]["durbin_watson"],
          "| BP p :", resume["homoscedasticite"]["breusch_pagan_p"],
          "| VIF max :", round(resume["vif_max"], 2))

    # --- 5. Évaluation + validation croisée -----------------------------
    print(">> 5. Évaluation")
    Xtr_final = _clean_names(X_train)[retained]
    Xte_final = _clean_names(X_test)[retained]
    metrics = evaluate.evaluate_ols(final, Xtr_final, y_train,
                                    Xte_final, y_test, save=True)
    resume["metriques"] = metrics.to_dict("records")
    resume["validation_croisee"] = evaluate.cross_validation(
        _clean_names(data.drop(columns=[TARGET])), data[TARGET], cv=10
    )
    print(metrics.to_string(index=False))

    # --- 6. Modèles complémentaires -------------------------------------
    print(">> 6. Modèles complémentaires")
    counts = reg.fit_count_models(df)
    resume["poisson_AIC"] = round(counts["poisson"].aic, 1)
    resume["negbin_AIC"] = round(counts["negbin"].aic, 1)
    reg_metrics = reg.fit_regularized(X_train, y_train, X_test, y_test)
    # Ajouter la ligne OLS pour la comparaison
    import pandas as pd
    ols_r2_test = float(metrics.loc[metrics["jeu"] == "test", "R2"].iloc[0])
    comp = pd.concat([
        pd.DataFrame([{"modele": "OLS (final)", "R2_test": ols_r2_test}]),
        reg_metrics,
    ], ignore_index=True)
    resume["comparaison_modeles"] = comp.to_dict("records")
    print(comp.to_string(index=False))

    # --- 7. Visualisations ----------------------------------------------
    print(">> 7. Figures")
    viz.plot_target_distribution(df)
    viz.plot_correlation_matrix(corr)
    viz.plot_scatter_target(df, top_vars)
    viz.plot_residuals_vs_fitted(final)
    viz.plot_qq(final)
    pred_test = final.predict(sm.add_constant(Xte_final.astype(float)))
    viz.plot_pred_vs_actual(y_test, pred_test)
    viz.plot_coefficients(coefs)
    viz.plot_cook(infl, resume["influence"]["seuil_cook"])
    viz.plot_model_comparison(comp)

    # --- Sauvegarde du résumé JSON --------------------------------------
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, tuple):
            return list(o)
        return str(o)

    with open(RESULTS_DIR / "resume_analyse.json", "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False, default=_default)

    print("\n>> Terminé. Résultats dans results/, figures dans figures/.")


if __name__ == "__main__":
    main()

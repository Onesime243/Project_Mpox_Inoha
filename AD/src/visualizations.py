"""Visualisations professionnelles pour l'analyse Mpox.

Chaque fonction enregistre une figure haute résolution dans figures/.
Toutes les figures ont des titres, axes nommés et unités lisibles.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend non interactif (exécution en script)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import FIGURES_DIR, TARGET, NUMERIC_FEATURES

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _save(fig, name: str) -> None:
    """Enregistre et ferme une figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Histogramme + boxplot de la distribution de la variable cible."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df[TARGET], bins=40, color="#2c7fb8", edgecolor="white")
    axes[0].set_title("Distribution des cas confirmés de Mpox")
    axes[0].set_xlabel("Cas confirmés (nombre)")
    axes[0].set_ylabel("Fréquence")
    axes[1].boxplot(df[TARGET], vert=True)
    axes[1].set_title("Boxplot des cas confirmés")
    axes[1].set_ylabel("Cas confirmés (nombre)")
    _save(fig, "01_distribution_cible.png")


def plot_correlation_matrix(corr: pd.DataFrame) -> None:
    """Carte thermique de la matrice de corrélation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Corrélation de Pearson")
    ax.set_title("Matrice de corrélation des variables numériques")
    _save(fig, "02_matrice_correlation.png")


def plot_scatter_target(df: pd.DataFrame, top_vars) -> None:
    """Nuages de points entre la cible et les variables les plus corrélées."""
    n = len(top_vars)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, var in zip(axes, top_vars):
        ax.scatter(df[var], df[TARGET], s=8, alpha=0.3, color="#238b45")
        ax.set_xlabel(var)
        ax.set_ylabel(TARGET)
        ax.set_title(f"{TARGET} vs {var}")
    _save(fig, "03_nuages_points.png")


def plot_residuals_vs_fitted(model) -> None:
    """Graphique des résidus contre les valeurs ajustées (homoscédasticité)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(model.fittedvalues, model.resid, s=8, alpha=0.3)
    ax.axhline(0, color="red", lw=1)
    ax.set_xlabel("Valeurs ajustées")
    ax.set_ylabel("Résidus")
    ax.set_title("Résidus contre valeurs ajustées")
    _save(fig, "04_residus_vs_ajustes.png")


def plot_qq(model) -> None:
    """QQ-plot des résidus (normalité)."""
    fig = sm.qqplot(model.resid, line="45", fit=True)
    fig.set_size_inches(6, 6)
    fig.suptitle("QQ-plot des résidus")
    _save(fig, "05_qqplot_residus.png")


def plot_pred_vs_actual(y_true, y_pred) -> None:
    """Observations réelles contre valeurs prédites (jeu de test)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.4)
    lims = [min(np.min(y_true), np.min(y_pred)),
            max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Cas confirmés observés")
    ax.set_ylabel("Cas confirmés prédits")
    ax.set_title("Valeurs prédites vs observées (jeu de test)")
    _save(fig, "06_predit_vs_reel.png")


def plot_coefficients(coef_table: pd.DataFrame) -> None:
    """Diagramme des coefficients avec intervalles de confiance à 95 %."""
    tab = coef_table.drop(index="const", errors="ignore").copy()
    tab = tab.reindex(tab["coefficient"].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8, max(5, 0.3 * len(tab))))
    err = [tab["coefficient"] - tab["IC_2.5%"], tab["IC_97.5%"] - tab["coefficient"]]
    ax.errorbar(tab["coefficient"], range(len(tab)), xerr=err,
                fmt="o", color="#c51b8a", ecolor="gray", capsize=3, ms=4)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels(tab.index, fontsize=7)
    ax.set_xlabel("Coefficient estimé")
    ax.set_title("Coefficients du modèle final (IC 95 %)")
    _save(fig, "07_coefficients.png")


def plot_cook(influence_df: pd.DataFrame, seuil: float) -> None:
    """Distance de Cook par observation."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stem(range(len(influence_df)), influence_df["cook_distance"],
            markerfmt=",", basefmt=" ")
    ax.axhline(seuil, color="red", lw=1, ls="--",
               label=f"Seuil 4/n = {seuil:.4f}")
    ax.set_xlabel("Indice d'observation")
    ax.set_ylabel("Distance de Cook")
    ax.set_title("Distance de Cook par observation")
    ax.legend()
    _save(fig, "08_distance_cook.png")


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    """Comparaison du R² test entre modèles."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(metrics["modele"], metrics["R2_test"], color="#41b6c4")
    ax.set_ylabel("R² (jeu de test)")
    ax.set_title("Comparaison des modèles (R² test)")
    for i, v in enumerate(metrics["R2_test"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    _save(fig, "09_comparaison_modeles.png")

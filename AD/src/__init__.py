"""Package d'analyse de régression linéaire multiple sur les données Mpox (RDC).

Ce package regroupe les modules de prétraitement, d'analyse exploratoire,
de modélisation, de diagnostics, d'évaluation et de visualisation.
"""

from pathlib import Path

# --- Chemins du projet (relatifs, portables entre machines) --------------
# On remonte d'un niveau depuis src/ pour atteindre la racine du projet.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT_DIR / "data" / "raw" / "mpox_original.csv"
DATA_PROCESSED = ROOT_DIR / "data" / "processed" / "mpox_cleaned.csv"
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR = ROOT_DIR / "results"
REPORTS_DIR = ROOT_DIR / "reports"

# --- Reproductibilité ----------------------------------------------------
RANDOM_STATE = 42

# --- Définition métier des variables ------------------------------------
# Cible : nombre de cas confirmés (comptage).
TARGET = "Cas_Confirmes"

# Variables à EXCLURE des prédicteurs :
#   - ID, Semaine        : identifiants / index de ligne (non informatifs).
#   - Taux_Positivite_pct: fuite de données -> identité mathématique
#     Taux_Positivite_pct = Cas_Confirmes / Tests_Realises * 100.
LEAK_OR_ID = ["ID", "Semaine", "Taux_Positivite_pct"]

# Variables explicatives numériques.
NUMERIC_FEATURES = [
    "Pluviometrie_mm",
    "Temperature_C",
    "NDVI",
    "Humidite_pct",
    "Densite_Population",
    "Couverture_Vaccinale_pct",
    "Tests_Realises",
    "Distance_Centre_Sante_km",
    "Mobilite_Humaine",
    "Population_Risque",
    "Reservoirs_Animaux",  # binaire 0/1, traitée comme numérique
]

# Variables explicatives qualitatives (encodage one-hot).
CATEGORICAL_FEATURES = ["Province", "Saison"]


def ensure_dirs() -> None:
    """Crée les dossiers de sortie s'ils n'existent pas encore."""
    for d in (FIGURES_DIR, RESULTS_DIR, REPORTS_DIR, DATA_PROCESSED.parent):
        d.mkdir(parents=True, exist_ok=True)

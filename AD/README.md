# Régression linéaire multiple — Cas confirmés de Mpox en RDC

Analyse statistique complète, modélisant le **nombre de cas confirmés de Mpox** en République Démocratique du
Congo à partir de facteurs climatiques, démographiques, écologiques et sanitaires.

## Description

Ce projet applique une régression linéaire multiple (moindres carrés ordinaires)
à un jeu de données de 3 000 observations, en respectant toutes les étapes d'une
analyse professionnelle : exploration, prétraitement, modélisation, diagnostics
des hypothèses, évaluation prédictive, validation croisée et comparaison avec des
modèles complémentaires (Poisson, binomial négatif, Ridge, LASSO, Elastic Net).

## Contexte

Le Mpox reste une menace de santé publique majeure en RDC. Identifier les facteurs
associés au nombre de cas confirmés aide à orienter la surveillance, la prévention
et l'allocation des ressources.

## Objectif

Expliquer et prédire le nombre de cas confirmés de Mpox, puis interpréter les
facteurs associés de façon rigoureuse et prudente (association, non causalité).

## Structure du projet

```
AD/
├── data/
│   ├── raw/mpox\_original.csv        # données brutes
│   └── processed/mpox\_cleaned.csv   # données nettoyées
├── notebooks/
│   └── mpox\_multiple\_regression.ipynb
├── src/
│   ├── \_\_init\_\_.py                  # config : chemins, graine, variables
│   ├── data\_preprocessing.py        # importation et nettoyage
│   ├── exploratory\_analysis.py      # statistiques et corrélations
│   ├── regression\_model.py          # OLS, sélection, Poisson/NB, régularisation
│   ├── diagnostics.py               # tests des hypothèses, VIF, influence
│   ├── model\_evaluation.py          # métriques et validation croisée
│   └── visualizations.py            # figures
├── figures/                         # graphiques générés
├── results/                         # tableaux de résultats (CSV/JSON)
├── reports/report.md                # rapport scientifique
├── main.py                          # pipeline complet
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Jeu de données

3 000 observations, 17 variables, aucune valeur manquante ni doublon.

* **Variable cible** : Cas\_Confirmes (nombre de cas confirmés, comptage).
* **Variables explicatives** : Pluviometrie\_mm, Temperature\_C, NDVI,
Humidite\_pct, Densite\_Population, Couverture\_Vaccinale\_pct,
Tests\_Realises, Distance\_Centre\_Sante\_km, Reservoirs\_Animaux,
Mobilite\_Humaine, Population\_Risque, Saison, Province.
* **Exclue** : `Taux\_Positivite\_pct` car
`Taux\_Positivite\_pct = Cas\_Confirmes / Tests\_Realises × 100` (fuite de données).
`ID` et `Semaine` sont des identifiants non informatifs.

## Méthodologie

Prétraitement → analyse exploratoire → modèle OLS complet → sélection backward
(AIC) → diagnostics (normalité, homoscédasticité, indépendance, VIF, influence) →
évaluation (MSE, RMSE, MAE, R²) → validation croisée 10-plis → modèles
complémentaires → visualisations et rapport.

## Prérequis

* Python 3.10 ou supérieur.
* Les bibliothèques listées dans 'requirements.txt'.

## Installation

### Créer un environnement virtuel

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\\Scripts\\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Installer les dépendances

```bash
pip install -r requirements.txt
```

## Exécution

Lancer l'analyse complète depuis la racine du projet :

```bash
python main.py
```

Le script régénère les données nettoyées, tous les tableaux de 'results/' et
toutes les figures de `figures/`.

## Principaux résultats

* Modèle final parcimonieux à **14 variables**.
* **R² (test) = 0,826** ; R² apprentissage = 0,823 → bonne généralisation.
* Validation croisée 10-plis : R² moyen = 0,820 (écart-type 0,013), très stable.
* Aucune multicolinéarité (VIF max ≈ 1,01), pas d'autocorrélation (DW ≈ 2,02).
* Hétéroscédasticité traitée par erreurs standards robustes (HC3).
* Associations fortes : pluviométrie, température, NDVI et effort de test
(positives) ; couverture vaccinale et distance au centre de santé (négatives).

## Limites

Cible de type comptage (modèle de comptage théoriquement préférable), possible
non-linéarité résiduelle (RESET significatif), données observationnelles (pas de
causalité) et biais de détection lié au nombre de tests et à l'accès aux soins.

## Auteur

* &#x20;Aimé DIUMI DIKOMO, aimediumi2@gmail.com, Apprenant M2 MASG, INOHA, UNIKIN

## Licence

Distribué sous licence MIT. Voir le fichier 'LICENSE'.


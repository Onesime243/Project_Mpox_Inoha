# Analyse de régression linéaire multiple des cas confirmés de Mpox en République Démocratique du Congo

*Rapport scientifique Par Aimé DIUMI DIKOLO*

## 1\. Introduction

Le Mpox (anciennement « variole du singe ») constitue une menace de santé publique
majeure en République Démocratique du Congo (RDC), pays qui concentre l'essentiel
des cas signalés en Afrique centrale. Comprendre les facteurs environnementaux,
démographiques et liés au système de santé associés au nombre de cas confirmés est
essentiel pour orienter la surveillance et la prévention. Ce rapport présente une
analyse de régression linéaire multiple appliquée à un jeu de données de
3 000 observations décrivant la situation du Mpox par province et par période.

## 2\. Contexte

Les données couvrent 26 provinces de la RDC et combinent des indicateurs
climatiques (pluviométrie, température, humidité, indice de végétation NDVI),
démographiques (densité de population, population à risque, mobilité humaine),
écologiques (présence de réservoirs animaux) et sanitaires (couverture vaccinale,
nombre de tests réalisés, distance au centre de santé). L'enjeu est d'identifier
quels facteurs sont statistiquement associés au fardeau de la maladie.

## 3\. Problématique

Quels facteurs environnementaux, démographiques et sanitaires sont associés au
nombre de cas confirmés de Mpox en RDC, et dans quelle mesure ces facteurs
permettent-ils de prédire ce nombre de cas ?

## 4\. Objectif général

Construire, valider et interpréter un modèle de régression linéaire multiple
expliquant le nombre de cas confirmés de Mpox à partir des variables disponibles.

## 5\. Objectifs spécifiques

* Décrire et nettoyer le jeu de données.
* Identifier une variable cible pertinente et justifier ce choix.
* Détecter et écarter les sources de fuite de données.
* Ajuster un modèle complet puis un modèle final parcimonieux.
* Vérifier les hypothèses de la régression linéaire.
* Évaluer la qualité prédictive du modèle et sa robustesse.
* Comparer le modèle à des alternatives (comptage, régularisation).
* Formuler des recommandations décisionnelles prudentes.

## 6\. Description des données

Le jeu de données comporte **3 000 observations et 17 variables**, sans aucune
valeur manquante ni doublon. Les variables se répartissent ainsi :

|Type|Variables|
|-|-|
|Identifiant / index|'ID', 'Semaine'|
|Géographique|'Province' (26 modalités)|
|Climatique|'Pluviometrie\_mm', 'Temperature\_C', 'NDVI', 'Humidite\_pct', 'Saison'|
|Démographique|'Densite\_Population', 'Population\_Risque', 'Mobilite\_Humaine'|
|Écologique|'Reservoirs\_Animaux' (binaire)|
|Sanitaire|'Couverture\_Vaccinale\_pct', 'Tests\_Realises', 'Distance\_Centre\_Sante\_km'|
|Résultat|'Cas\_Confirmes', 'Taux\_Positivite\_pct'|

### Choix de la variable cible

Deux variables pouvaient jouer le rôle de cible : 'Cas\_Confirmes' (comptage) et
'Taux\_Positivite\_pct' (taux continu). **'Cas\_Confirmes' a été retenue** car elle
représente directement le fardeau épidémiologique de la maladie et constitue le
résultat le plus pertinent à expliquer par des facteurs de risque.

**Découverte critique — fuite de données.** La vérification a montré que
'Taux\_Positivite\_pct = Cas\_Confirmes / Tests\_Realises × 100' de façon **exacte**
sur les 3 000 lignes (écart maximal = 0). Cette identité mathématique impose
d'**exclure `Taux\_Positivite\_pct` des variables explicatives** : l'inclure ferait
grimper artificiellement le R² à environ 0,93 sans aucune valeur explicative
réelle. Les identifiants 'ID' et 'Semaine' sont également exclus (non informatifs ;
'Semaine' vaut simplement 1, 2, …, 3000 et ne code pas de dynamique temporelle).

### Nature de la cible et limite de la régression linéaire

'Cas\_Confirmes' est un **comptage** (entier ≥ 0, moyenne 79,6, minimum 0,
maximum 479). La régression linéaire ordinaire (MCO) reste utilisable et
informative, mais elle n'est pas théoriquement idéale pour un comptage : elle peut
prédire des valeurs négatives et suppose une variance constante. Des modèles de
comptage (Poisson, binomial négatif) sont donc ajustés en complément (section 12).

## 7\. Méthodologie

L'analyse suit une chaîne reproductible en Python ('pandas', 'numpy',
'statsmodels', 'scikit-learn', 'scipy', 'matplotlib'). La cible est expliquée par
un modèle linéaire de la forme :

> \*\*Y = β₀ + β₁X₁ + β₂X₂ + … + βₚXₚ + ε\*\*

où Y = 'Cas\_Confirmes', les Xⱼ sont les variables explicatives, β₀ l'ordonnée à
l'origine, βⱼ les coefficients estimés par moindres carrés ordinaires, et ε le
terme d'erreur. Les hypothèses examinées sont : linéarité, indépendance des
erreurs, espérance nulle et homoscédasticité des erreurs, normalité des résidus
(pour l'inférence), absence de multicolinéarité parfaite et absence d'observations
excessivement influentes.

## 8\. Prétraitement des données

Les noms de colonnes ont été nettoyés, les types vérifiés et convertis. Aucune
valeur manquante ni doublon n'a été détecté, donc aucune imputation ni suppression
de lignes n'a été nécessaire. Les variables qualitatives (`Province`, `Saison`)
ont été encodées en indicatrices (one-hot) avec suppression de la première
modalité (`drop\_first=True`) afin d'éviter la colinéarité parfaite avec la
constante. La standardisation n'est pas nécessaire pour l'ajustement MCO (elle
n'affecte pas le R² ni les tests) ; elle est appliquée uniquement dans les
pipelines de régularisation, où l'échelle des variables importe.

## 9\. Analyse exploratoire

Les corrélations les plus fortes avec la cible sont :

|Variable|Corrélation avec 'Cas\_Confirmes'|
|-|-|
|Pluviométrie|+0,63|
|Tests réalisés|+0,46|
|Couverture vaccinale|−0,35|
|Densité de population|+0,18|
|Distance au centre de santé|−0,15|

La pluviométrie et le nombre de tests sont positivement associés au nombre de cas,
tandis que la couverture vaccinale est négativement associée — cohérent avec un
effet protecteur de la vaccination. La distribution des cas est asymétrique à
droite (voir 'figures/01\_distribution\_cible.png'), ce qui est typique d'un
comptage. La matrice de corrélation (`figures/02\_matrice\_correlation.png`) ne
révèle pas de corrélations extrêmes entre variables explicatives, ce qui anticipe
une faible multicolinéarité.

## 10\. Construction du modèle

### Séparation des données

Les données étant transversales (l'index 'Semaine' ne code pas de temps réel), une
séparation **aléatoire 80 % / 20 %** a été utilisée ('random\_state = 42') :
2 400 observations d'apprentissage et 600 de test. Une séparation chronologique
serait incorrecte ici puisqu'il n'existe pas d'ordre temporel signifiant.

### Modèle complet puis modèle final

Le **modèle complet** (toutes les variables + 25 indicatrices de province) obtient
R² = 0,824, R² ajusté = 0,821, AIC = 23 064, BIC = 23 284. Une **sélection
backward fondée sur l'AIC** a réduit le modèle à **14 variables** sans perte de
qualité : R² = 0,823, R² ajusté = 0,822, **AIC = 23 030** et **BIC = 23 116**,
tous deux inférieurs à ceux du modèle complet. Le modèle final est donc à la fois
plus parcimonieux et statistiquement préférable.

## 11\. Vérification des hypothèses

|Hypothèse|Test / outil|Résultat|Conclusion|
|-|-|-|-|
|Linéarité|RESET de Ramsey|p < 0,001|Léger défaut de spécification signalé|
|Normalité des résidus|Shapiro-Wilk, Jarque-Bera|p < 0,001 ; asymétrie 1,04 ; aplatissement 5,96|Résidus non parfaitement normaux|
|Homoscédasticité|Breusch-Pagan|p < 0,001|**Hétéroscédasticité présente**|
|Indépendance|Durbin-Watson|2,02|Pas d'autocorrélation|
|Multicolinéarité|VIF|VIF max ≈ 1,01|**Aucune multicolinéarité**|
|Observations influentes|Distance de Cook|Cook max = 0,028 (≪ 1)|Aucune observation dominante|

**Interprétation.** L'absence quasi totale de multicolinéarité (VIF ≈ 1) et
d'autocorrélation (Durbin-Watson ≈ 2) est très favorable. En revanche, le test de
Breusch-Pagan indique une **hétéroscédasticité** et les résidus s'écartent de la
normalité (asymétrie et queues épaisses, cohérentes avec la nature de comptage de
la cible). Avec un grand échantillon (n = 2 400), le théorème central limite
atténue l'impact de la non-normalité sur l'inférence. Pour l'hétéroscédasticité,
des **erreurs standards robustes (HC3)** ont été calculées : elles laissent toutes
les variables principales hautement significatives (voir
`results/coefficients\_robustes.csv`), ce qui confirme la robustesse des
conclusions. Le RESET significatif suggère qu'une transformation logarithmique de
la cible ou l'ajout de termes non linéaires pourrait encore améliorer la
spécification — piste d'amélioration future.

## 12\. Modèles complémentaires

* **Poisson** : AIC = 34 686. **Binomial négatif** : AIC = 30 379. Le binomial
négatif domine largement Poisson, signalant une surdispersion des comptages
(la variance dépasse la moyenne), ce qui est attendu pour des données
épidémiologiques.
* **Régularisation** (Ridge, LASSO, Elastic Net, standardisées dans un pipeline,
hyperparamètres optimisés par validation croisée) : R² de test entre 0,824 et
0,825, soit des performances **quasi identiques** au MCO. Comme la
multicolinéarité est nulle, la régularisation n'apporte pas de gain — résultat
attendu et rassurant.

## 13\. Résultats

### Équation estimée (modèle final)

En arrondissant les coefficients principaux :

```
Cas\_Confirmes ≈ −164,8
   + 0,451 · Pluviometrie\_mm
   + 2,618 · Temperature\_C
   + 60,10 · NDVI
   + 0,377 · Humidite\_pct
   + 0,143 · Densite\_Population
   − 1,723 · Couverture\_Vaccinale\_pct
   + 0,361 · Tests\_Realises
   − 0,818 · Distance\_Centre\_Sante\_km
   − 2,907 · Reservoirs\_Animaux
   + 7,199 · Saison\_Seche
   + (effets de province : Sankuru, Haut-Uélé, Kasaï-Central, Lomami)
```

### Performances prédictives

|Jeu|MSE|RMSE|MAE|R²|
|-|-|-|-|-|
|Apprentissage|850,0|29,2|21,7|0,823|
|Test|933,2|30,5|22,3|**0,826**|

Les performances sont **quasi identiques** entre apprentissage et test, signe d'un
**bon ajustement sans surajustement**. La validation croisée 10-plis confirme la
stabilité : R² moyen = 0,820, écart-type = 0,013.

## 14\. Interprétation des coefficients

Toutes choses égales par ailleurs (les autres variables étant tenues constantes) :

* **Pluviométrie (+0,45)** : chaque millimètre de pluie supplémentaire est associé
à environ 0,45 cas de plus. La saison des pluies favoriserait la transmission.
* **Température (+2,62)** : chaque degré Celsius de plus est associé à \~2,6 cas
supplémentaires.
* **NDVI (+60,1)** : une végétation plus dense est fortement associée à davantage
de cas (une hausse de 0,1 d'indice ≈ +6 cas), cohérent avec un contact accru avec
des réservoirs sauvages en milieu forestier.
* **Humidité (+0,38)** et **densité de population (+0,14)** : associations
positives modérées.
* **Couverture vaccinale (−1,72)** : chaque point de pourcentage de couverture
supplémentaire est associé à \~1,7 cas de moins — l'effet protecteur le plus
marqué du modèle.
* **Tests réalisés (+0,36)** : davantage de tests détectent davantage de cas. Cet
effet reflète l'**effort de détection**, et non nécessairement une hausse de
l'incidence réelle — à interpréter avec prudence.
* **Distance au centre de santé (−0,82)** : les zones plus éloignées enregistrent
moins de cas, ce qui peut traduire une **sous-détection** plutôt qu'une moindre
incidence réelle.
* **Réservoirs animaux (−2,91)** et **saison sèche (+7,20 par rapport à la saison
des pluies)** : effets significatifs plus modestes.

Pour les provinces, la modalité de référence est la première province par ordre
alphabétique ; seules quelques provinces (Sankuru, Haut-Uélé, etc.) présentent un
écart significatif au reste. Les erreurs robustes HC3 ne modifient pas ces
conclusions.

**Avertissement causal.** Ces coefficients décrivent des **associations**, non des
relations de cause à effet. Les données étant observationnelles, aucune conclusion
causale ne peut être tirée.

## 15\. Discussion

Le modèle explique environ **82 % de la variance** du nombre de cas confirmés, ce
qui est élevé. Les facteurs climatiques (pluie, température, végétation) et
sanitaires (vaccination, effort de test, accès aux soins) ressortent de façon
cohérente avec la littérature épidémiologique sur les zoonoses. L'effet du nombre
de tests et de la distance au centre de santé rappelle que le nombre de cas
*confirmés* dépend fortement de la capacité de détection, ce qui peut masquer la
distribution réelle de la maladie.

## 16\. Limites

* La cible est un **comptage** ; le MCO reste une approximation (le binomial
négatif est théoriquement supérieur).
* Le test RESET signale une possible **non-linéarité** non capturée.
* Une **hétéroscédasticité** est présente (traitée par erreurs robustes).
* Les données sont **observationnelles** : pas d'inférence causale.
* Le biais de **détection** (tests, distance aux soins) peut fausser le nombre de
cas confirmés par rapport à l'incidence réelle.
* La significativité de certaines provinces peut dépendre de la modalité de
référence choisie.

## 17\. Recommandations

* **Surveillance** : renforcer la détection dans les zones éloignées des centres de
santé, où la sous-détection est probable.
* **Prévention** : la forte association négative avec la couverture vaccinale
soutient l'intensification des campagnes de vaccination.
* **Ciblage saisonnier et écologique** : accroître la vigilance pendant les
périodes pluvieuses et dans les zones à forte densité végétale (NDVI élevé).
* **Allocation des ressources** : prioriser les provinces et contextes climatiques
associés à un nombre de cas plus élevé.
* **Analyses futures** : tester un modèle binomial négatif et des transformations
(log) pour affiner la spécification.

## 18\. Conclusion

Un modèle de régression linéaire multiple parcimonieux (14 variables) explique
82 % de la variance des cas confirmés de Mpox en RDC, avec une bonne
généralisation (R² test = 0,826, validation croisée stable). Les facteurs
climatiques, la couverture vaccinale et l'accès aux soins sont les associations les
plus marquantes. Ces résultats, de nature associative, fournissent des pistes utiles
pour la surveillance et la prévention, tout en appelant à la prudence sur
l'interprétation causale et sur les biais de détection.




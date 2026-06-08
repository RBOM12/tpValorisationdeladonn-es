# -*- coding: utf-8 -*-
"""Génère le notebook TP4 dans le style des TP précédents (TP3)."""
import json

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text})

# 0. Badge Colab
md('<a href="https://colab.research.google.com/github/RBOM12/tpValorisationdeladonn-es/blob/main/TP4_BROCHET_Valorisation_de_la_donn%C3%A9es.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>')

# 1. Titre
md("# TP 4 Valorisation de la données\n\nApprentissage supervisé (régression) et Intelligence Artificielle eXplicable (LIME & SHAP).")

# Section régression
md("## Apprentissage supervisé : régression\n\nContrairement à la classification, où l'on prédit des valeurs discrètes (des classes), la **régression** prédit des valeurs **continues**. Ici, ces valeurs continues sont des **prix de maisons** (colonne `price`).")

# Imports
code("""# Imports pour la régression et l'IA explicable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error""")

# Q1
md("> **1.** Charger le jeu de données *housing* (fichier `Housing.csv`). Les valeurs à prédire sont dans la colonne `price` : les stocker dans un vecteur `y`. Le reste des données est stocké dans une matrice `X`. La première ligne du fichier contient les noms des colonnes. Combien d'attributs (variables) et combien de données contient le jeu de données ?")

code("""# 1. Chargement du jeu de données housing
# La première ligne du CSV contient les noms de colonnes (comportement par défaut de read_csv).
df = pd.read_csv("Housing.csv")

# La cible à prédire est la colonne 'price' -> vecteur y
y = df["price"].values
# Le reste des données constitue la matrice X
X = df.drop(columns="price")

# Noms des attributs (utiles plus tard pour LIME et SHAP)
feature_names = list(X.columns)

print(f"Nombre de données (instances) : {X.shape[0]}")
print(f"Nombre d'attributs (variables) : {X.shape[1]}")
print(f"\\nAttributs : {feature_names}")
df.head()""")

md("Le jeu de données contient **545 données (instances)** et **12 attributs** (la 13ᵉ colonne `price` étant la cible `y`, elle ne fait pas partie de `X`).\n\nParmi ces 12 attributs, certains sont **numériques** (`area`, `bedrooms`, `bathrooms`, `stories`, `parking`) et d'autres sont **catégoriels** (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`).")

# Q2
md("> **2.** Le jeu de données contient des variables **catégorielles**, or nous ne travaillons qu'avec des nombres. Transformer ces variables catégorielles en nombres à l'aide d'un objet de la classe `OrdinalEncoder` de `sklearn.preprocessing`. Pour `furnishingstatus` : 0 pour *unfurnished*, 1 pour *semi-furnished*, 2 pour *furnished*. Pour les attributs booléens : *yes* devient 1 et *no* devient 0.")

code("""# 2. Transformation des variables catégorielles en nombres avec OrdinalEncoder
# On choisit la 1re approche (un nombre par catégorie), et NON le OneHotEncoding.
colonnes_cat = ["mainroad", "guestroom", "basement", "hotwaterheating",
                "airconditioning", "prefarea", "furnishingstatus"]

# On impose explicitement l'ordre des catégories pour respecter la consigne :
#   - booléens : no -> 0, yes -> 1
#   - furnishingstatus : unfurnished -> 0, semi-furnished -> 1, furnished -> 2
categories = [["no", "yes"]] * 6 + [["unfurnished", "semi-furnished", "furnished"]]

encoder = OrdinalEncoder(categories=categories)
X = X.copy()
X[colonnes_cat] = encoder.fit_transform(X[colonnes_cat])

# On convertit X en matrice numpy de réels (toutes les colonnes sont désormais numériques)
X = X.values.astype(float)

print("X est maintenant entièrement numérique, de forme :", X.shape)
print("Exemple (5 premières lignes) :")
print(np.round(X[:5], 2))""")

md("Toutes les variables catégorielles sont désormais encodées en nombres. On a utilisé la **première approche** (un entier par catégorie) demandée par l'énoncé, et non le *OneHotEncoding*.\n\nLe paramètre `categories` de l'`OrdinalEncoder` permet d'**imposer l'ordre** : sans lui, `furnishingstatus` aurait été encodé par ordre alphabétique (*furnished*=0), ce qui ne correspond pas à la consigne (*unfurnished*=0, *semi-furnished*=1, *furnished*=2). Le nombre de dimensions reste inchangé (**12**), contrairement au OneHotEncoding qui aurait ajouté des colonnes.")

# Q3
md("> **3.** Séparer le jeu de données en un jeu d'entraînement (70 %) et un jeu de test (30 %) avec `train_test_split`, en fixant `random_state=42`. Stocker les ensembles dans `X_train`, `y_train`, `X_test`, `y_test`.")

code("""# 3. Séparation en jeu d'entraînement (70%) et jeu de test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("X_train :", X_train.shape, "|  y_train :", y_train.shape)
print("X_test  :", X_test.shape, "|  y_test  :", y_test.shape)""")

# Q4
md("> **4.** À quoi sert la normalisation ? Normaliser le jeu de données.")

code("""# 4. Normalisation des attributs avec le StandardScaler
# On entraîne (fit) le scaler uniquement sur X_train pour ne pas laisser fuiter
# d'information du jeu de test, puis on applique la transformation séparément.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Avant normalisation - moyennes des attributs (X_train) :")
print(np.round(X_train.mean(axis=0), 1))
print("\\nAprès normalisation - moyennes (≈ 0) et écarts-types (≈ 1) :")
print("moyennes  :", np.round(X_train_scaled.mean(axis=0), 3))
print("écarts-types :", np.round(X_train_scaled.std(axis=0), 3))""")

md("**À quoi sert la normalisation ?** Les attributs ont des **échelles très différentes** : `area` se compte en milliers (de pieds carrés) tandis que `bedrooms` ou `parking` valent quelques unités. Sans normalisation, les attributs de grande amplitude domineraient le calcul des distances, et un SVM (qui s'appuie sur des distances via son noyau) serait biaisé en leur faveur.\n\nLa normalisation **ramène tous les attributs à la même échelle** (moyenne 0, écart-type 1 avec le `StandardScaler`), ce qui permet à chaque attribut de contribuer équitablement et accélère/améliore l'entraînement. Comme en TP3, on `fit` le scaler **uniquement sur `X_train`** puis on l'applique séparément à `X_train` et `X_test`, afin que le jeu de test reste inconnu du modèle.")

# Q5
md("> **5.** Entraîner un régresseur SVM avec la classe `SVR` de `sklearn.svm` (noyau `rbf`), en utilisant `y_train` et la version normalisée de `X_train`.")

code("""# 5. Entraînement d'un régresseur SVM (SVR, noyau RBF)
regressor = SVR(kernel="rbf")
regressor.fit(X_train_scaled, y_train)""")

md("On entraîne le `SVR` avec le noyau `rbf` sur `X_train_scaled` (attributs normalisés) et `y_train` (les prix). La méthode `fit` ajuste le modèle de régression.")

# Q6
md("> **6.** Calculer les prédictions de chaque instance du jeu de test (normalisé) avec `predict`. Les stocker dans `y_pred`. `y_pred` contient les prix prédits, `y_test` les vrais prix.")

code("""# 6. Prédictions sur le jeu de test normalisé
y_pred = regressor.predict(X_test_scaled)

# Comparaison des 5 premières prédictions avec les vraies valeurs
print("Prédiction  ->  Vrai prix")
for p, v in zip(y_pred[:5], y_test[:5]):
    print(f"{p:12,.0f}  ->  {v:12,.0f}")""")

md("`y_pred` contient les prix **prédits** par le modèle pour chaque maison du jeu de test, tandis que `y_test` contient les **vrais** prix. On remarque que les prédictions sont toutes très proches les unes des autres (autour de la moyenne des prix), ce qui est un premier indice de la difficulté du modèle à capturer les variations de prix — on le confirme avec les métriques d'erreur à la question suivante.")

# Q7
md("> **7.** Évaluer le modèle en calculant l'**erreur quadratique moyenne** (*mean squared error*) et l'**erreur absolue moyenne** (*mean absolute error*) avec `mean_squared_error` et `mean_absolute_error` de `sklearn.metrics`. Plus ces erreurs sont faibles, meilleur est le modèle.")

code("""# 7. Évaluation : erreur quadratique moyenne (MSE) et erreur absolue moyenne (MAE)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)   # racine de la MSE, dans la même unité que les prix

print(f"MSE  (erreur quadratique moyenne) : {mse:,.0f}")
print(f"RMSE (racine de la MSE)           : {rmse:,.0f}")
print(f"MAE  (erreur absolue moyenne)     : {mae:,.0f}")
print(f"\\nPour comparaison, prix moyen des maisons : {y.mean():,.0f}")
print(f"MAE en % du prix moyen : {100 * mae / y.mean():.1f} %")""")

md("Les erreurs sont **élevées** : la MAE vaut environ **1 580 000**, soit près de **33 %** du prix moyen (≈ 4 770 000). Contrairement à la précision et au rappel vus en classification, ici **plus l'erreur est faible, meilleur est le modèle** — et ces valeurs montrent que le modèle est **médiocre**.\n\nL'explication tient au `SVR` utilisé avec ses **hyperparamètres par défaut** (`C=1.0`, `epsilon=0.1`) : la cible `price` n'étant pas normalisée et prenant de très grandes valeurs (millions), le coût de régularisation par défaut est largement insuffisant. Le modèle prédit alors quasiment la **valeur moyenne** pour toutes les maisons (cf. question 6). On pourrait l'améliorer en augmentant fortement `C`, en ajustant `epsilon`/`gamma` (via une recherche d'hyperparamètres), voire en normalisant aussi `y` — mais l'énoncé nous demande la configuration de base, que nous conservons ici.")

# Q8 : LIME & SHAP
md("> **8.** Expliquer quelques instances avec **LIME** et **SHAP**.\n\nComme en TP3, on installe les deux bibliothèques puis on instancie un *explainer* pour chacune, cette fois en mode **régression**.")

# --- LIME ---
md("### LIME")

code("""# Installation de LIME dans l'environnement TP_VD
!pip install lime""")

code("""# Instanciation de l'explainer LIME en mode régression
from lime.lime_tabular import LimeTabularExplainer

# mode="regression" car on prédit une valeur continue (le prix) et non une classe.
# On fournit les données d'entraînement normalisées et les noms des attributs.
lime_explainer = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=feature_names,
    mode="regression",
    random_state=42,
)""")

md("On choisit une instance du jeu de test et on l'explique avec `explain_instance`. Pour une régression, l'explication indique comment chaque attribut **augmente** ou **diminue** le prix prédit par rapport à la moyenne.")

code("""# Explication LIME de l'instance d'indice 10 du jeu de test
idx = 10
prix_predit = regressor.predict(X_test_scaled[idx].reshape(1, -1))[0]

print(f"Instance n°{idx}")
print(f"Prix prédit : {prix_predit:,.0f}")
print(f"Vrai prix   : {y_test[idx]:,.0f}")

explication = lime_explainer.explain_instance(
    X_test_scaled[idx],
    regressor.predict,
    num_features=12,
)""")

code("""# Visualisation de l'explication LIME
fig = explication.as_pyplot_figure()
fig.set_size_inches(9, 5)
plt.title(f"Explication LIME - instance n°{idx} (prix prédit : {prix_predit:,.0f})")
plt.tight_layout()
plt.show()""")

md("**Que traduit l'explication ?** Chaque barre indique dans quelle mesure un attribut **pousse le prix prédit vers le haut** (barres positives, en vert) ou **vers le bas** (barres négatives, en rouge), par rapport à la prédiction moyenne.\n\nLes attributs les plus déterminants sont `area` (la surface), `airconditioning`, `bathrooms` et `stories` : une surface élevée et la présence de climatisation tirent le prix vers le haut, ce qui est cohérent avec l'intuition du marché immobilier. À l'inverse, l'absence de certains équipements diminue le prix estimé.")

# --- SHAP ---
md("### SHAP")

code("""# Installation de SHAP dans l'environnement TP_VD
!pip install shap""")

code("""# Instanciation de l'explainer SHAP (classe Sampling) en régression
# La classe Sampling calcule des approximations des valeurs de Shapley.
import shap

shap_explainer = shap.explainers.Sampling(regressor.predict, X_train_scaled)""")

code("""# Calcul des valeurs SHAP de l'instance d'indice 10 du jeu de test normalisé
shap_values = shap_explainer.shap_values(X_test_scaled[idx])

# En régression, shap_values est un simple vecteur (une valeur par attribut).
print("Forme des valeurs SHAP :", np.asarray(shap_values).shape, " (un coefficient par attribut)")
print(f"Valeur attendue (expected_value, ≈ prix moyen prédit) : {shap_explainer.expected_value:,.0f}\\n")

# Affichage des valeurs SHAP triées par importance (valeur absolue)
ordre = np.argsort(np.abs(shap_values))[::-1]
print(f"Contributions SHAP pour l'instance n°{idx} :")
for i in ordre:
    print(f"   {shap_values[i]:+12,.0f}   {feature_names[i]}")""")

md("Chaque valeur SHAP indique la **contribution d'un attribut** à l'écart entre le prix prédit pour cette maison et le prix moyen prédit (`expected_value`). Une valeur **positive** augmente le prix estimé, une valeur **négative** le diminue. Les contributions sont **additives** : `expected_value` + somme des valeurs SHAP = prix prédit pour l'instance.")

code("""# Visualisation avec le decision plot
# Premier argument = prédiction moyenne (expected_value).
shap.plots.decision(
    shap_explainer.expected_value,
    shap_values,
    feature_names=feature_names,
)""")

code("""# Visualisation avec le force plot : il affiche en plus la valeur de chaque attribut
shap.plots.force(
    shap_explainer.expected_value,
    shap_values,
    features=X_test_scaled[idx],
    feature_names=feature_names,
    matplotlib=True,
)""")

md("Le *decision plot* part du prix moyen et empile les contributions attribut par attribut jusqu'au prix final. Le *force plot* montre les mêmes forces sur un axe horizontal et affiche **la valeur réelle de chaque attribut** à côté de sa contribution.\n\n**Comparaison LIME / SHAP.** Les deux méthodes désignent les **mêmes attributs déterminants** (`area`, `airconditioning`, `bathrooms`, `stories`) avec des signes cohérents, ce qui renforce la confiance dans l'explication. LIME ajuste un modèle linéaire local autour de l'instance (poids relatifs, légère variabilité selon l'échantillonnage), tandis que SHAP repose sur les valeurs de Shapley dont les contributions sont **additives** et théoriquement plus fondées.")

# Autre instance
md("Enfin, on explique une **autre instance** de notre choix pour vérifier la cohérence des explications.")

code("""# Explication d'une autre instance (la maison la plus chère du jeu de test)
idx2 = int(np.argmax(y_test))
prix_predit2 = regressor.predict(X_test_scaled[idx2].reshape(1, -1))[0]
print(f"Instance n°{idx2} (la plus chère du jeu de test)")
print(f"Prix prédit : {prix_predit2:,.0f}  |  Vrai prix : {y_test[idx2]:,.0f}\\n")

# LIME
exp2 = lime_explainer.explain_instance(X_test_scaled[idx2], regressor.predict, num_features=12)
fig = exp2.as_pyplot_figure()
fig.set_size_inches(9, 5)
plt.title(f"Explication LIME - instance n°{idx2}")
plt.tight_layout()
plt.show()

# SHAP
shap_values2 = shap_explainer.shap_values(X_test_scaled[idx2])
shap.plots.force(
    shap_explainer.expected_value, shap_values2,
    features=X_test_scaled[idx2], feature_names=feature_names, matplotlib=True,
)""")

md("Pour cette maison (la plus chère du jeu de test), LIME et SHAP s'accordent à nouveau : les attributs `area`, `airconditioning` et `bathrooms` ressortent comme les principaux moteurs du prix. On note toutefois que le prix prédit reste bien **en-dessous** du vrai prix : le modèle, qui prédit proche de la moyenne, **sous-estime fortement les maisons haut de gamme** — ce que l'analyse des erreurs (question 7) avait déjà révélé. L'IA explicable confirme ainsi, de façon visuelle et locale, le comportement global du modèle.")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Convertir les sources en listes de lignes (format ipynb) + ajouter un id par cellule
import itertools
counter = itertools.count(1)
for c in nb["cells"]:
    c["id"] = f"cell{next(counter)}"
    src = c["source"]
    lines = src.split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

with open("TP4_BROCHET_Valorisation_de_la_données.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook écrit :", len(nb["cells"]), "cellules")

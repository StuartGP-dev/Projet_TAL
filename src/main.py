import pandas as pd


# Chargement des données:
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")

# -------------------------------------------------------------------------------------

# Construction des données:
# Ici TF-IDF ne peut prendre qu'un seul champ en entrée
# On va donc concatener les données en les séparants par des espaces.
# Ce qui va nous donner la colonne text.

for df in (train, test):
    # Ajoute la col "text"
    df["text"] = (
        df["titre"].fillna("") + " " +
        df["ingredients"].fillna("") + " " +
        df["recette"].fillna("")
    )
# -------------------------------------------------------------------------------------

# Exploration EDA
print("Exploratory Data Analysis")
print("Train:", train.shape, "\nTest:", test.shape)
print("\nRépartition classes (train):")
print(train["type"].value_counts())
print("\nEn pourecentage:")
print(train["type"].value_counts(normalize=True) * 100)

#On compte le nombre de mots et affiche les stats sur ces longueurs.
print("\nLongueur (mots) - stats:")
for col in ["ingredients", "recette", "text"]:
    print("Longueur", col, ":")
    print(train[col].str.split().apply(len).describe(), "\n")


#count = 12473 : 12 473 recettes
#mean = 175.85 : en moyenne, une ligne fait ~176 mots.
#std = 81.01 : l’écart-type (dispersion).
#min = 25 : la plus courte ligne a 25 mots.

# -------------------------------------------------------------------------------------

# Baseline “classe majoritaire” (plancher)
# Le but est de faire un modele bete qui trouve tjrs en sortie la classe majoritaire (Plat principal) afin d'avoir une baseline.

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# Afin de faire de la validation croisée en gardant la même proportion de classes dans chaque fold.
# FOLD = morceau du dataset. Si on test sur un fold, on entraine sur tous les autres.
# Ici on decoupe en 5 puis on fait la moy des scores.
cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# faux modèle qui va apprndre la classe la plus fréquente
baseline = DummyClassifier(strategy="most_frequent")

# Entrée: X, Sortie: y
X = train["text"]
y = train["type"]

# Lance l’entrainement
scores = cross_validate(
    baseline, X, y, cv=cross_validation,
    scoring={"micro_f1":"f1_micro", "macro_f1":"f1_macro", "acc":"accuracy"}
)

# micro-F1 : F1 globale en agrégeant toutes les classes (les classes fréquentes pèsent plus)
# macro-F1 : moyenne des F1 calculées séparément pour chaque classe (toutes les classes pèsent pareil)
# accuracy : pourcentage de prédictions correctes

print("Baseline par classe majoritaire:")
for metrics in ["test_micro_f1" , "test_macro_f1", "test_acc"]:
    print(metrics, ":", scores[metrics].mean())
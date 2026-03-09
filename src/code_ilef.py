import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC



test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")


# TF-IDF ne peut prendre qu'un seul champ en entrée
# df["text"] = données concatenees par avec des espaces.

for df in (train, test):
    df["text"] = (
        df["titre"].fillna("") + " " +
        df["ingredients"].fillna("") + " " +
        df["recette"].fillna("")
    )



# I - Exploration des données 
print("Exploratory Data Analysis")
print("Train:", train.shape, "\nTest:", test.shape)
print("\nRépartition classes (train):")
print(train["type"].value_counts())
print("\nEn pourecentage:")
print(train["type"].value_counts(normalize=True) * 100)

# On compte le nombre de mots et affiche les stats sur ces longueurs.
print("\nLongueur (mots) - stats:")
for col in ["ingredients", "recette", "text"]:
    print("Longueur", col, ":")
    print(train[col].str.split().apply(len).describe(), "\n")

#count = 12473 : 12 473 recettes
#mean = 175.85 : en moyenne, une ligne fait ~176 mots.
#std = 81.01 : l’écart-type (dispersion).
#min = 25 : la plus courte ligne a 25 mots.




# II - Baseline
# Baseline 2 : Classe majoritaire
# Le but est de faire un modele bete qui trouve tjrs en sortie la classe majoritaire (Plat principal) afin d'avoir une baseline.

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate


# faux modèle qui va apprndre la classe la plus fréquente
majoritaire_baseline = DummyClassifier(strategy="most_frequent")

# Entrée: X, Sortie: y
X = train["text"]
y = train["type"]

# Lance l’entrainement
scores = cross_validate(
    majoritaire_baseline, X, y,
    scoring={"micro_f1":"f1_micro", "macro_f1":"f1_macro", "acc":"accuracy"}
)

# micro-F1 : F1 globale en agrégeant toutes les classes (les classes fréquentes pèsent plus)
# macro-F1 : moyenne des F1 calculées séparément pour chaque classe (toutes les classes pèsent pareil)
# accuracy : pourcentage de prédictions correctes

print("Baseline par classe majoritaire:")
for metrics in ["test_micro_f1" , "test_macro_f1", "test_acc"]:
    print(metrics, ":", scores[metrics].mean())



# Baseline Aleatoire
rand_baseline = DummyClassifier(strategy="uniform", random_state=42)
scores = cross_validate(
    rand_baseline, X, y,
    scoring={"micro_f1":"f1_micro", "macro_f1":"f1_macro", "acc":"accuracy"}
)

print("\nBaseline aléatoire:")
for m in ["test_micro_f1","test_macro_f1","test_acc"]:
    print(m, scores[m].mean())
    
# Partie 3 : Méthode A : Sac de mots + Naïve Bayes
# TF-IDF unigrammes (1,1) + MultinomialNB


# Même découpage pour toutes les expériences (important)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

methode_A = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_df=0.95)),
    ("clf", MultinomialNB())
])

scores_A = cross_validate(
    methode_A, X, y, cv=cv,
    scoring={"micro_f1":"f1_micro", "macro_f1":"f1_macro", "acc":"accuracy"}
)

print("\nMéthode A (TF-IDF unigrammes + Naïve Bayes):")
for m in ["test_micro_f1","test_macro_f1","test_acc"]:
    print(m, scores_A[m].mean())


______________________________________________________________________


methode_B = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
    ("clf", LinearSVC())
])

scores_B = cross_validate(
    methode_B, X, y, cv=cv,
    scoring={"micro_f1":"f1_micro", "macro_f1":"f1_macro", "acc":"accuracy"}
)

print("\nMéthode B (TF-IDF 1-2 grams + LinearSVC):")
for m in ["test_micro_f1", "test_macro_f1", "test_acc"]:
    print(m, scores_B[m].mean())
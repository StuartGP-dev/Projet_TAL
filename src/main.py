import pandas as pd


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


random_state = 42

# Baseline Aleatoire
rand_baseline = DummyClassifier(strategy="uniform", random_state=random_state)
scores = cross_validate(
    rand_baseline, X, y,
    scoring={"micro_f1":"f1_micro", "macro_f1":"f1_macro", "acc":"accuracy"}
)

print("\nBaseline aléatoire:")
for m in ["test_micro_f1","test_macro_f1","test_acc"]:
    print(m, scores[m].mean())



# Méthode A:

# Représentation sac de mots : TF-IDF
# Unités : unigrammes (ngram_range=(1,1))
# Classifieur : Naive Bayes (en pratique MultinomialNB)


from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

# Méthode A : TF-IDF unigrammes + Naïve Bayes
runA = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),   # unigrammes
        min_df=2              # ignore mots trop rares (tu peux tester 1 ou 2)
    )),
    ("clf", MultinomialNB(alpha=1.0))  # alpha = lissage de Laplace (testable)
])

scores = cross_validate(
    runA, X, y, cv=cv,
    scoring={"micro_f1": "f1_micro", "macro_f1": "f1_macro", "acc": "accuracy"}
)

print("\n=== Run A (TF-IDF unigrammes + MultinomialNB) ===")
print("micro-F1 :", scores["test_micro_f1"].mean())
print("macro-F1 :", scores["test_macro_f1"].mean())
print("accuracy :", scores["test_acc"].mean())

# 2) Rapport détaillé + matrice de confusion (sur prédictions CV)
y_pred = cross_val_predict(runA, X, y, cv=cv)

print("\n Classification report (CV) ")
print(classification_report(y, y_pred, digits=3))

labels = sorted(y.unique())
cm = confusion_matrix(y, y_pred, labels=labels)
print("\n Matrice de confusion ( ", labels, ") ")
print(cm)
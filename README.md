# DEFT2013 Tâche 2 : NOMEQUIPE (optionnel)


Dabin Yanis - Mhabrech Ilef 


## Description de la tâche

L’objectif du projet est de **prédire automatiquement le type d’une recette de cuisine** parmi trois classes :
- Entrée
- Plat principal
- Dessert

- **Entrées (inputs)** :
  - Titre
  - Ingrédients
  - Recette 

- **Sortie (output)** :
  - Type de la recette (Entrée / Plat / Dessert)

==> Il s’agit donc d’un problème de **classification supervisée**.

### Exemple de contenu de train.csv :

- doc_id : recette_221358.xml  
- Titre : Feuilleté de saumon et de poireau, sauce aux crevettes  
- Type : Plat principal  
- Difficulté : Facile  
- Ingrédients : ...  
- Recette : ...  

---

## Installation et exécution

### Installation

```bash
pip install -r requirements.txt
```

### Exécution

```bash
cd src
jupyter notebook
```

---
## Protocole expérimental
Afin de garantir la comparabilité des résultats, nous avons respecté le protocole expérimental suivant :

- Le découpage train/test fourni a été respecté
- Le jeu de test n’a jamais été utilisé pour l’entraînement
- Toutes les expériences utilisent le même protocole
- random_state fixé à 42 pour garantir la reproductibilité 

## Statistiques corpus

Nous avons réalisé une analyse exploratoire du corpus :
- distribution des classes
- longueur des recettes
- analyse du vocabulaire
- spécificités lexicales par classe

Pour mieux comprendre les données avant modélisation

### Nombre de documents

- Train : 12 473 documents  
- Test : 1 388 documents  

### Répartition des classes

| Classe          | Train (%) | Test (%) |
|----------------|----------:|---------:|
| Plat principal | 46.5 %    | 46.4 %   |
| Dessert        | 30.2 %    | 29.3 %   |
| Entrée         | 23.3 %    | 24.3 %   |


### Visualisation

![Répartition des classes](figure/proportion_classes_train_test.png)

Observation :  
- **déséquilibre des classes : les plats principals sont majoritaires**
---

### Statistiques des longueurs (mots)

| Champ       | Moyenne | Min | Max |
|------------|--------:|----:|----:|
| ingrédients | 47      | 7   | 232 |
| recette     | 123     | 9   | 1334 |
| texte total | 176     | 25  | -   |


### Distribution des longueurs

#### Texte complet
![Longueur textes](figure/longueur_textes_hist_par_classe.png)

#### Recettes
![Longueur recettes](figure/longueur_recettes_hist_par_classe.png)

#### Ingrédients
![Longueur ingrédients](figure/longueur_ingredients_hist_par_classe.png)

Observations :
- Les recettes sont en moyenne **assez longues (~176 mots)**
- Forte **variabilité**

---

### Spécificités lexicales

Nous avons calculé un **ratio de spécificité** :

ratio = fréquence dans la classe / fréquence hors classe

#### Dessert

Exemples marquants :

- **biscuit** → 311 occurrences dans desserts vs 1 hors classe  
  → ratio = 377.38 (extrêmement discriminant)

- **ganache** → 93 dans desserts vs 0 ailleurs  
  → mot exclusivement lié aux desserts

Conclusion :  
Les desserts ont un vocabulaire **très spécifique et sucré**, ce qui facilite la classification.

#### Entrée

Exemples :

- **huîtres** → 70 vs 6  
- **avocats** → 175 vs 25  
...  

Conclusion :  
Les entrées sont liées à des aliments **frais, légers ou froids**, mais avec un vocabulaire moins spécifique , classification plus difficile.


#### Plat principal

Exemples :

- **cailles** → 125 vs 0  
- **pintade** → 92 vs 0  

 Conclusion :  
Les plats principaux sont associés à des **viandes, plats chauds et plats complets**, avec un vocabulaire riche mais plus varié.

### Visualisation des mots spécifiques

![Spécificités lexicales](figure/specificites_lexicales_par_classe.png)

### interprétations : 
- Les desserts sont les plus faciles à classifier (vocabulaire très spécifique)
- Les entrées et plats principaux sont plus proches → erreurs fréquentes
- Certaines recettes sont ambiguës (ex : quiches, salades composées)
- Le déséquilibre des classes influence les résultats
---


## Méthodes proposées
Nous avons commencé par comparer différentes sources d’information textuelle :
- Titre seul
- Recette seule
- Texte complet (titre + ingrédients + recette)

###  Run1 : Titre seul

**Description :**
Utilisation uniquement du titre de la recette comme entrée.

**Résultats :**
- micro-F1 : 0.823  
- macro-F1 : 0.808  

**Analyse :**
- Les titres sont **déjà très informatifs**
- Très bons résultats pour les desserts (F1 ≈ 0.92)
- Difficulté à distinguer **Entrée vs Plat principal**

Conclusion :  
Les titres contiennent des **indices lexicaux forts**, mais insuffisants pour certaines classes.

###  Run2 : Recette seule

**Description :**
Utilisation uniquement des instructions de la recette.

**Résultats :**
- micro-F1 : 0.861  
- macro-F1 : 0.845  

**Analyse :**
- Amélioration nette par rapport au titre
- Les instructions contiennent plus d’information
- Toujours des erreurs entre Entrée et Plat principal

Conclusion :  
Les instructions sont **plus riches que le titre** pour la classification.

###  Run3 : Texte complet

**Description :**
Utilisation de toutes les informations disponibles :
titre + ingrédients + recette

**Résultats :**
- micro-F1 : 0.870  
- macro-F1 : 0.857  

**Analyse :**
- Meilleure performance globale
- Les différentes sources d’information sont **complémentaires**
- Les desserts sont très bien classés (F1 ≈ 0.986)

Conclusion :  
Le texte complet est la meilleure représentation.

### Impact du champ utilisé

![Comparaison champs](figure/comparaison_champs.png)

---

## Analyse des questions du projet

### Les titres seuls sont-ils discriminants ?

Oui, les titres seuls donnent déjà de bons résultats (micro-F1 ≈ 0.82).  
Cela montre qu’ils contiennent des mots clés importants (ex : gâteau, salade, soupe).

Cependant, ils ne suffisent pas toujours à distinguer :
- Entrée vs Plat principal

### Les instructions seules suffisent-elles ?

Les instructions donnent de meilleures performances (micro-F1 ≈ 0.86).  
Elles apportent plus de détails (cuisson, ingrédients, techniques).

Mais le texte complet reste meilleur → les autres champs apportent une information complémentaire.

### Certaines recettes semblent-elles ambiguës ?

Oui, l’ambiguïté est clairement visible dans les matrices de confusion :

- 865 entrées classées comme plats  
- 643 plats classés comme entrées  

Les classes **Entrée et Plat principal sont proches**

À l’inverse :
- Les desserts sont très bien identifiés (F1 ≈ 0.986)

---

## Baselines

nous avons implémenté deux baselines simples afin d’établir un point de comparaison.

###  Baseline 1 : Aléatoire

**Description :**  
Le modèle attribue une classe au hasard parmi les trois classes possibles.

**Résultats :**
- micro-F1 : 0.316  
- macro-F1 : 0.309  
- accuracy : 0.316  

**Analyse :**
- Les performances sont faibles, proches du hasard.
- La matrice de confusion montre que les prédictions sont réparties aléatoirement.
- Aucune structure n’est capturée.

Conclusion :  
Cette baseline sert uniquement de référence minimale.


###  Baseline 2 : Classe majoritaire

**Description :**  
Le modèle prédit toujours la classe la plus fréquente : **Plat principal**.

**Résultats :**
- micro-F1 : 0.464  
- macro-F1 : 0.211  
- accuracy : 0.464  

**Analyse :**
- L’accuracy est plus élevée que l’aléatoire car la classe majoritaire est souvent correcte.
- Cependant :
  - F1 = 0 pour Dessert et Entrée
  - Le modèle ignore complètement ces classes

Observation importante :
- La **macro-F1 chute fortement** (0.21), ce qui montre que le modèle est très mauvais globalement.
- Ce modèle est fortement biaisé par le déséquilibre des classes.

Conclusion :  
- Bonne accuracy mais modèle inutile en pratique  
- Montre l’importance d’utiliser des métriques adaptées (macro-F1)

### Conclusion sur les baselines

- La baseline aléatoire donne une performance très faible → point de départ minimal
- La baseline majoritaire améliore l’accuracy mais échoue à capturer la diversité des classes
- Ces résultats justifient l’utilisation de modèles plus avancés (TF-IDF, modèles linéaires, etc.)

---

## Modèles de classification

Après les baselines, nous avons implémenté plusieurs modèles plus avancés basés sur TF-IDF.

###  Méthode A : TF-IDF + Naive Bayes

**Description :**
- Représentation : TF-IDF (unigrammes)
- Modèle : Multinomial Naive Bayes

**Résultats :**
- micro-F1 : 0.805  
- macro-F1 : 0.732  

**Analyse :**
- Très bon score pour les **desserts (F1 ≈ 0.98)** → vocabulaire très spécifique
- Très mauvais score pour les **entrées (F1 ≈ 0.39)**  
  → beaucoup d’entrées classées comme plats
- Le modèle est biaisé vers les classes dominantes

Conclusion :  
- Modèle simple mais limité  
- Sensible au déséquilibre des classes  
- Mauvaise gestion des classes difficiles (Entrée)

---

###  Méthode B : TF-IDF + Logistic Regression

**Description :**
- Représentation : TF-IDF avec n-grammes (1,2)
- Modèle : Régression logistique

**Résultats :**
- micro-F1 : 0.872  
- macro-F1 : 0.859  

**Analyse :**
- Forte amélioration par rapport à Naive Bayes
- Meilleur équilibre entre les classes
- Entrée mieux reconnue (F1 ≈ 0.72)

Conclusion :  
- Modèle robuste  
- Les **n-grammes apportent du contexte**  
- Bon compromis global

###  Méthode B (variante) : TF-IDF + SVM (LinearSVC)

**Description :**
- Représentation : TF-IDF (uni + bi-grammes)
- Modèle : SVM linéaire

**Résultats :**
- micro-F1 : 0.878  
- macro-F1 : 0.868  

**Analyse :**
- Meilleur modèle global
- Très bonnes performances sur toutes les classes
- Réduction des erreurs entre Entrée et Plat

### Matrice de confusion

La matrice de confusion suivante montre les performances du modèle SVM :

![Matrice brute](figure/matrice_confusion_runB_svm.png)

![Matrice normalisée](figure/matrice_confusion_normalisee_runB_svm.png)

### F1-score par classe

La figure suivante montre les performances détaillées par classe :

![F1-score](figure/f1_par_classe_runB_svm.png)

 Conclusion :  
- Le SVM est le **meilleur modèle parmi ceux testés**
- Très adapté aux données textuelles

###  Méthode C : Modèle enrichi

**Description :**
- TF-IDF (uni + bi-grammes)
- Ajout de features :
  - mots spécifiques par classe
  - longueurs des textes

**Résultats :**
- micro-F1 : 0.869  
- macro-F1 : 0.860  

**Analyse :**
- Les features ajoutées n’améliorent pas significativement les performances
- Le modèle reste proche du SVM simple

Conclusion :  
- Les informations ajoutées sont déjà capturées par TF-IDF
- Complexifier le modèle n’apporte pas toujours un gain

## Comparaison des méthodes

| Méthode                     | micro-F1 | macro-F1 |
|----------------------------|----------:|----------:|
| Naive Bayes                | 0.805     | 0.732     |
| Logistic Regression        | 0.872     | 0.859     |
| SVM (LinearSVC)            | 0.878     | 0.868     |
| Modèle enrichi             | 0.869     | 0.860     |

### visualisation : 


![Comparaison méthodes](figure/comparaison_methodes.png)


observation : 
- Les modèles linéaires (Logistic Regression, SVM) sont **nettement meilleurs**
- Le passage aux **n-grammes améliore les performances**
- Le SVM donne les meilleurs résultats globaux
- Le problème principal reste :
  → distinction **Entrée vs Plat principal**


## Analyse des résultats

###  Comparaison globale des méthodes

| Méthode | micro-F1 | macro-F1 | accuracy |
|--------|---------:|---------:|---------:|
| Baseline aléatoire | 0.316 | 0.309 | 0.316 |
| Baseline majoritaire | 0.464 | 0.211 | 0.464 |
| Naive Bayes | 0.805 | 0.732 | 0.805 |
| Logistic Regression | 0.872 | 0.859 | 0.872 |
| SVM (LinearSVC) | **0.878** | **0.868** | **0.878** |
| Modèle enrichi | 0.869 | 0.860 | 0.869 |
| Modèle enrichi + longueurs | 0.867 | 0.858 | 0.867 |

#### visualisation : 

![Comparaison méthodes](figure/comparaison_des_methodes_globales.png)


### interprétations : 

- Les performances augmentent avec la **complexité des modèles**, mais seulement jusqu’à un certain point
- Le **meilleur compromis** est obtenu avec un modèle simple mais efficace : **TF-IDF + SVM**
- Le principal facteur limitant n’est pas le modèle, mais :
  - la **proximité entre certaines classes (Entrée vs Plat)**
  - et le **déséquilibre des données**


---

## Réflexion critique sur le projet

###  Points forts

- Les résultats montrent que le problème est **bien modélisable** avec des méthodes classiques.
- Le pipeline expérimental est **rigoureux** :
  - séparation train/test respectée
  - comparaison équitable entre méthodes
- Les analyses sont **cohérentes avec les données** :
  - forte performance sur les desserts
  - difficulté sur les entrées

###  Limites du projet

1. **Ambiguïté de la tâche**

Certaines recettes sont difficiles à catégoriser :
- une quiche peut être entrée ou plat
- une salade peut être entrée ou plat

 La tâche n’est pas parfaitement définie → limite intrinsèque.

2. **Déséquilibre des classes**

- Les plats principaux sont majoritaires
- Cela influence les modèles (biais vers cette classe)

 Importance de la macro-F1 pour corriger ce biais.

3. **Difficulté Entrée vs Plat principal**

- Les erreurs sont concentrées entre ces deux classes
- Le vocabulaire est très proche

Limite liée aux données, pas seulement au modèle.

## Conclusion générale

Ce projet montre que :
- les approches classiques (TF-IDF + modèles linéaires) sont **très efficaces**
- le principal défi n’est pas technique, mais **lié aux données (ambiguïté, similarité des classes)**

 Le meilleur modèle obtenu (SVM) atteint des performances élevées (~0.88),  
ce qui confirme la **pertinence des méthodes utilisées**.

---
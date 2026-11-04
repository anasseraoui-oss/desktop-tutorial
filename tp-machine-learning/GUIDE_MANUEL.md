# Guide manuel : refaire les TPs à la main (Anaconda + Jupyter Notebook)

Ce guide est pour quelqu'un qui veut tout faire **à la main**, sans Docker : installer l'outil, créer le notebook, taper le code cellule par cellule et prendre ses captures.

---

## Étape 0 : préparer l'environnement (une seule fois)

1. **Installer Anaconda** depuis https://www.anaconda.com/download (Windows, 64 bits). Pendant l'installation, garder les options par défaut.
2. **Ouvrir Anaconda Navigator** (menu Démarrer ▸ *Anaconda Navigator*).
3. Sur la tuile **Jupyter Notebook**, cliquer sur **Launch**. Le navigateur s'ouvre sur `http://localhost:8888/tree`.
4. **Créer un dossier de travail**, par exemple `Documents\TP_ML`. Dans Jupyter, ouvrir ce dossier.
5. **Installer `mglearn`**, qui n'est pas inclus dans Anaconda et sert au TP1. Ouvrir *Anaconda Prompt* (menu Démarrer) et taper :
   ```
   pip install mglearn
   ```
   Les autres bibliothèques (numpy, pandas, matplotlib, scikit-learn) sont déjà dans Anaconda.

### Gestes de base dans Jupyter

| Action | Comment |
|---|---|
| Nouveau notebook | *New ▸ Python 3 (ipykernel)* |
| Renommer | Cliquer sur le titre `Untitled` en haut |
| Exécuter une cellule | `Shift + Entrée` |
| Ajouter une cellule | Bouton **+** (ou touche `B` hors édition) |
| Cellule de texte | Menu déroulant **Code ▸ Markdown** |
| Tout exécuter | *Run ▸ Run All Cells* |
| Sauvegarder | `Ctrl + S` |
| Recommencer à zéro | *Kernel ▸ Restart Kernel and Run All Cells* |
| Exporter | *File ▸ Save and Export Notebook As ▸ HTML* (ou PDF) |
| Capture d'écran (Windows) | `Windows + Shift + S` |

> Règle d'or : exécuter les cellules **dans l'ordre**, de haut en bas. Si une cellule donne `NameError`, c'est qu'une cellule précédente n'a pas été exécutée.

---

## TP1 : KNN, classification des fleurs d'Iris

Créer le notebook `TP1_KNN_iris.ipynb`, puis taper une cellule par étape.

1. **Imports et chargement**
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import mglearn
   from pandas.plotting import scatter_matrix
   from sklearn.datasets import load_iris
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.model_selection import train_test_split

   iris_dataset = load_iris()
   print(iris_dataset.keys())
   print(iris_dataset['target_names'])
   print(iris_dataset['feature_names'])
   print(iris_dataset['data'].shape)      # (150, 4)
   ```
2. **Explorer les données**
   ```python
   print(iris_dataset['data'][:5])
   print(iris_dataset['target'])
   ```
3. **Séparer entraînement / test**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       iris_dataset['data'], iris_dataset['target'], random_state=0)
   print(X_train.shape, X_test.shape)     # (112, 4) (38, 4)
   ```
4. **Afficher la matrice de dispersion**
   ```python
   iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
   scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                  hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
   plt.show()
   ```
5. **Créer et entraîner le modèle**
   ```python
   knn = KNeighborsClassifier(n_neighbors=1)
   knn.fit(X_train, y_train)
   ```
6. **Prédire pour une nouvelle fleur**
   ```python
   X_new = np.array([[5, 2.9, 1, 0.2]])
   prediction = knn.predict(X_new)
   print(iris_dataset['target_names'][prediction])   # ['setosa']
   ```
7. **Évaluer**
   ```python
   y_pred = knn.predict(X_test)
   print("Exactitude : {:.2f}".format(knn.score(X_test, y_test)))   # 0.97
   ```

**Résultat attendu : exactitude = 0.97.**

---

## TP2 : régression linéaire, prix Airbnb à Brooklyn

1. **Télécharger les données** : Kaggle ▸ *New York City Airbnb Open Data* ▸ `AB_NYC_2019.csv`. Le fichier est aussi dans ce repo : `tp2_regression_lineaire/data/`. Le placer dans le dossier du notebook.
2. **Charger et filtrer**
   ```python
   import pandas as pd, numpy as np, matplotlib.pyplot as plt
   data = pd.read_csv('AB_NYC_2019.csv')           # ou r'c:\AB_NYC_2019.csv'
   data = data[data['neighbourhood_group'] == 'Brooklyn']
   ```
3. **Écrire les fonctions de la régression simple**
   ```python
   def calculate_slope(x, y):
       mx = x - x.mean()
       my = y - y.mean()
       return sum(mx * my) / sum(mx**2)

   def get_params(x, y):
       a = calculate_slope(x, y)
       b = y.mean() - a * x.mean()
       return a, b
   ```
4. **Calculer a et b, puis tracer la droite**
   ```python
   d = data[data.price > 0]
   x = d.number_of_reviews
   y = np.log(d.price)
   a, b = get_params(x, y)
   print(a, b)                                    # ≈ 0.00068, 4.55
   lin_reg = a*x + b
   plt.xlabel("Nombre d'avis"); plt.ylabel('Prix (log)')
   plt.scatter(x, y); plt.plot(x, lin_reg, color='red'); plt.show()
   ```
5. **Régression multiple avec scikit-learn**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   d = data[data.price > 0].copy()
   d['reviews_per_month'] = d['reviews_per_month'].fillna(0)
   x = d.loc[:, ('reviews_per_month', 'calculated_host_listings_count', 'number_of_reviews')]
   y = np.log(d.loc[:, 'price'])        # garder le log, voir la remarque
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
   model = LinearRegression().fit(x_train, y_train)
   y_pred = model.predict(x_test)
   ```
6. **Calculer MSE et MAE**
   ```python
   from sklearn.metrics import mean_squared_error, mean_absolute_error
   y_test = np.exp(y_test); y_pred = np.exp(y_pred)
   print('mse: %.3f, mae: %.3f' % (mean_squared_error(y_test, y_pred),
                                    mean_absolute_error(y_test, y_pred)))
   ```

> **Remarque :** l'énoncé prend `y = price` sans logarithme, puis applique `np.exp`, d'où une mse absurde (≈ 1e26). En gardant `np.log(price)`, on obtient MAE ≈ 63 $ et RMSE ≈ 156 $.

---

## TP3 : cartes de Kohonen (SOM)

1. **Récupérer le module SOM** : copier `tp3_kohonen_som/notebooks/som_kohonen.py` dans le **même dossier** que le notebook. Sinon, `from som_kohonen import ...` donne `ModuleNotFoundError`.
2. **Imports**
   ```python
   import numpy as np, matplotlib.pyplot as plt
   from som_kohonen import SOM, plot_data_on_map
   from mpl_toolkits.mplot3d import Axes3D
   ```
3. **Générer les données** : 1000 couleurs RGB, affichées en 3D.
   ```python
   dataset = np.random.randint(0, 2, size=(1000, 3)).astype(np.float32) * 2
   dataset[:500, 1] = np.linspace(0, 2, 500)
   dataset[:500, 0] = 0
   dataset += np.random.rand(1000, 3) * .7
   colors = (dataset - dataset.min()) / (dataset - dataset.min()).max()
   fig = plt.figure(figsize=(10, 10)); ax = fig.add_subplot(projection='3d')
   ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=colors, s=100, marker='.')
   plt.show()
   ```
4. **Créer et entraîner la carte**
   ```python
   som = SOM()
   som.create(width=15, height=15, ch=3)
   som.fit(dataset, random_sampling=0.5, n_iter=20, dist_method='euclidean')
   ```
5. **Trouver la position des données sur la carte**
   ```python
   data_locations, vect_distances = som.find_maching_nodes(dataset)
   ```
6. **Afficher la carte RGB, puis la U-matrice**
   ```python
   mv = som.get_map_vectors().copy(); mv -= mv.min(); mv /= mv.max()
   plt.imshow(mv); plt.show()
   umatrix = som.get_umatrix()
   plt.imshow(umatrix, cmap='gray'); plt.show()
   ```
7. **Projeter les données sur la carte**
   ```python
   plot_data_on_map(umatrix, data_locations, data_colors=colors, node_width=50, data_marker_size=100);
   ```
8. **Afficher les plans de composantes** : faire une boucle `for i in range(3)` avec `som.get_component_plane(i)` et **appeler `plt.show()` dans la boucle**, sinon seul le dernier plan s'affiche.

---

## TP4 : régression logistique, cancer du sein

1. **Télécharger les données** : Kaggle ▸ *Breast Cancer Wisconsin (Diagnostic) Data Set* ▸ `data.csv`. Le fichier est aussi dans le repo : `tp4_regression_logistique/notebooks/`.
2. **Imports**
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   ```
3. **Charger et explorer** (une cellule par commande)
   ```python
   bc_df = pd.read_csv(r"c:\data.csv")     # adapter le chemin
   bc_df.head()
   ```
   ```python
   bc_df.info()
   ```
   ```python
   bc_df.isnull().sum()                    # seule 'Unnamed: 32' est vide
   ```
4. **Supprimer les colonnes inutiles**
   ```python
   bc_df.drop(['Unnamed: 32', 'id'], axis = 1, inplace = True)
   bc_df.info()
   ```
   ⚠️ N'exécuter cette cellule **qu'une seule fois**. Une deuxième exécution donne `KeyError`, car les colonnes ont déjà été supprimées. Dans ce cas, relancer depuis le `read_csv`.
5. **Encoder le diagnostic** (M = 1, B = 0)
   ```python
   bc_df['diagnosis'] = bc_df['diagnosis'].map({'M': 1, 'B': 0})
   bc_df['diagnosis'].value_counts()       # 0: 357, 1: 212
   ```
6. **Séparer et standardiser**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   X = bc_df.drop('diagnosis', axis = 1)
   y = bc_df['diagnosis']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
   scaler = StandardScaler()
   X_train_sc = scaler.fit_transform(X_train)
   X_test_sc = scaler.transform(X_test)
   ```
7. **Entraîner le modèle**
   ```python
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression(max_iter = 1000)
   model.fit(X_train_sc, y_train)
   y_pred = model.predict(X_test_sc)
   ```
8. **Évaluer**
   ```python
   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
   print(accuracy_score(y_test, y_pred))          # ≈ 0.965
   print(confusion_matrix(y_test, y_pred))        # [[71 1] [3 39]]
   print(classification_report(y_test, y_pred))
   ```
9. **Bonus : courbe ROC** avec `roc_curve` et `roc_auc_score` (AUC ≈ 0.996).

---

## Erreurs fréquentes

| Erreur | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: mglearn` | bibliothèque absente | `pip install mglearn` dans *Anaconda Prompt*, puis *Kernel ▸ Restart* |
| `ModuleNotFoundError: som_kohonen` | `som_kohonen.py` n'est pas à côté du notebook | copier le fichier dans le même dossier |
| `FileNotFoundError` sur `read_csv` | mauvais chemin | mettre le CSV à côté du notebook et écrire `read_csv("data.csv")` |
| `NameError: ... is not defined` | une cellule précédente n'a pas été exécutée | *Run ▸ Run All Cells* |
| `KeyError: "['Unnamed: 32', 'id'] not found"` | le `drop` a été exécuté deux fois | relancer depuis le `read_csv` |
| `bc_df.head` affiche une méthode | les parenthèses manquent | écrire `bc_df.head()` |

## Pour rendre le TP

1. *Kernel ▸ Restart Kernel and Run All Cells*, pour vérifier que tout s'exécute sans erreur et dans l'ordre.
2. `Ctrl + S`.
3. Faire les captures avec `Windows + Shift + S` : le code **et** son résultat (tableau, graphique, score).
4. *File ▸ Save and Export Notebook As ▸ HTML* (ou PDF).
5. Rendre le `.ipynb`, l'export et les captures.

# AutoML – Projet Méthodologie IA

Ce dépôt contient une implémentation **AutoML légère et robuste** développée dans le cadre du module *Méthodologie IA*.  
L’objectif est de proposer un pipeline automatique capable de :

- Charger différents jeux de données hétérogènes
- Détecter automatiquement le type de tâche (classification / régression)
- Gérer les datasets multi-output (y de grande dimension)
- Entraîner plusieurs modèles et sélectionner le meilleur
- Produire des métriques cohérentes (F1-macro, accuracy, R², etc.)

---

## 📁 Structure du projet

├── automl.py # Pipeline AutoML principal
├── dataio.py # Chargement des datasets (.data / .solution / .type)
├── preprocessing.py # Pré-traitement des données (imputation, encodage, scaling)
├── README.md


---

## ⚙️ Fonctionnalités principales

- Détection automatique du type de tâche
- Support des datasets :
  - classification / régression
  - mono-output et multi-output
- Pré-traitement automatique :
  - gestion des valeurs manquantes
  - encodage des variables catégorielles
  - normalisation des variables numériques
- Sélection automatique du meilleur modèle parmi :
  - SGD
  - Logistic Regression
  - Random Forest
  - Histogram Gradient Boosting
- Optimisations spécifiques pour les datasets volumineux et multi-output

---

## ▶️ Exemple d’utilisation

```python
import automl

am = automl.AutoML(verbose=True, time_budget_sec=1800)
am.fit("/info/corpus/ChallengeMachineLearning/data_A/data_A")

result = am.eval()
print(result)

am.refit_full_data()
am.save("model_best.joblib")

---

📊 Métriques utilisées
Classification
•	F1-score (macro)
•	F1-score (weighted)
•	Accuracy
Régression
•	R²
•	MSE
•	MAE

🧪 Jeux de données

Les datasets utilisés suivent le format :

.data : variables explicatives

.solution : variables cibles

.type : types des features (Numerical / Categorical)

⚠️ Les datasets ne sont pas inclus dans ce dépôt.

👨‍🎓 Contexte académique

Projet réalisé dans le cadre du Master 1 – Intelligence Artificielle
Module : Méthodologie IA

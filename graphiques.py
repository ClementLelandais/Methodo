# Palette utilisée pour les graphiques
colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E', '#C73E1D']

# Graphique AVANT regroupement (raw winners)
import matplotlib.pyplot as plt

# Données brutes
datasets = ['A','B','C','D','E','F','G','H','I','J','K']
algos = [
    'RF (MultiOutput)', 'GB (Hist, Regr)', 'RF (MultiOutput)',
    'GB (Hist, Class)', 'GB (Hist, Class)', 'LogReg (MultiOutput)',
    'GB (Hist, Class)', 'GB (MultiOutput)', 'GB (MultiOutput)',
    'LogReg (MultiOutput)', 'SGD (MultiOutput)'
]
scores = [0.7355,0.7887,0.9704,0.8126,0.8614,0.4935,
          0.9463,0.4996,0.9814,0.7989,0.4920]

colors_raw = ['#2E86AB','#6A994E','#2E86AB','#6A994E','#6A994E',
              '#F18F01','#6A994E','#A23B72','#A23B72','#F18F01','#C73E1D']

plt.figure(figsize=(12,5))
bars = plt.bar(datasets, scores, color=colors_raw)
plt.title("Scores gagnants par dataset (avant regroupement)")
plt.xlabel("Dataset")
plt.ylabel("Score (F1 / R²)")
plt.ylim(0,1.05)

for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height+0.01,
             f"{height:.3f}", ha='center')

plt.show()

# Graphique APRÈS regroupement (familles)
familles = ['Gradient Boosting', 'Random Forest', 'Logistic Regression',
            'MultiOutput Gradient Boosting', 'SGD Classifier']
victoires = [4,2,2,2,1]

colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E', '#C73E1D']

plt.figure(figsize=(10,5))
bars = plt.bar(familles, victoires, color=colors)
plt.title("Nombre de victoires par famille d'algorithmes")
plt.ylabel("Nombre de victoires")
plt.xticks(rotation=20)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2., height+0.05,
             str(int(height)), ha='center')

plt.show()

# Moyenne des scores gagnants
scores_groupes = {
    'Gradient Boosting': [0.7887, 0.8126, 0.8614, 0.9463],
    'Random Forest': [0.7355, 0.9704],
    'Logistic Regression': [0.4935, 0.7989],
    'MultiOutput Gradient Boosting': [0.4996, 0.9814],
    'SGD Classifier': [0.4920]
}

familles = list(scores_groupes.keys())
moyennes = [sum(scores_groupes[f])/len(scores_groupes[f]) for f in familles]

colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E', '#C73E1D']

plt.figure(figsize=(10,5))
bars = plt.bar(familles, moyennes, color=colors)
plt.title("Score moyen des algorithmes gagnants")
plt.ylabel("Score moyen (F1 / R²)")
plt.ylim(0,1.05)
plt.xticks(rotation=20)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2., height+0.01,
             f"{height:.3f}", ha='center')

plt.show()


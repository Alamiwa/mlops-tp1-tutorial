import os
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# 1. Charger le dataset Iris
data = load_iris(as_frame=True)
df = data.frame

# 2. Séparer les variables X et y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. Séparer train et test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Créer et entraîner le modèle
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Créer le dossier models
os.makedirs("models", exist_ok=True)

# 6. Sauvegarder le modèle
with open("models/iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

# 7. Calculer le score
score = model.score(X_test, y_test)

print(f"Model successfully saved.")
print(f"Score: {score:.4f}")
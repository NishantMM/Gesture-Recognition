import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

data_dir = 'data'
df = pd.concat([pd.read_csv(f'data/{f}') for f in os.listdir(data_dir)])

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)

print(f"[INFO] Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/gesture_model.pkl')
print("[INFO] Model saved to models/gesture_model.pkl")

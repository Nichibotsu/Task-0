from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

def SVM_Model():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=42)

    # GridSearchCV für die Optimierung von Hyperparametern
    param_grid = {
        'C': [0.1, 1, 10, 100,50],
        'gamma': [0.001, 0.01, 0.1, 1,0.002,0.0015,0.0009,0.0005,0.0008],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    print(f"Beste Parameter: {grid.best_params_}")

    accuracy = grid.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    plt.gray()
    plt.matshow(digits.images[0])

SVM_Model()

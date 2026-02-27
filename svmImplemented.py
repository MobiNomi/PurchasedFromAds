# Support Vector Machine (SVM) - Complete code (raw scatter + training/test decision regions)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

# =========================
# 1) Load dataset
# =========================
df = pd.read_csv("Social_Network_Ads.csv")

# Use Age and EstimatedSalary as features, Purchased as target
X = df.iloc[:, [2, 3]].values        # Age, EstimatedSalary
y = df.iloc[:, 4].values             # Purchased (0/1)

# =========================
# 2) Raw data scatter plot (before scaling)
# =========================
plt.figure(figsize=(7, 5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Purchased = 0", alpha=0.7)
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Purchased = 1", alpha=0.7)
plt.title("Raw Data: Age vs Estimated Salary")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# =========================
# 3) Train/test split
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# =========================
# 4) Feature scaling
# =========================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 5) Train SVM classifier
# =========================
from sklearn.svm import SVC

clf = SVC(kernel="linear", random_state=0)
clf.fit(X_train_scaled, y_train)

# =========================
# 6) Predict + confusion matrix
# =========================
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = clf.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", acc)

# =========================
# 7) Helper function to plot decision regions
# =========================
def plot_decision_regions(X_set, y_set, title):
    cmap_bg = ListedColormap(("red", "green"))
    cmap_points = ListedColormap(("red", "green"))

    # Create dense grid in scaled feature space
    x1_min, x1_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
    x2_min, x2_max = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1
    X1, X2 = np.meshgrid(
        np.arange(start=x1_min, stop=x1_max, step=0.01),
        np.arange(start=x2_min, stop=x2_max, step=0.01),
    )

    # Predict on each grid point
    grid = np.c_[X1.ravel(), X2.ravel()]
    Z = clf.predict(grid).reshape(X1.shape)

    # Plot decision surface
    plt.figure(figsize=(7, 5))
    plt.contourf(X1, X2, Z, alpha=0.75, cmap=cmap_bg)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    # Plot points
    for i, cls in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == cls, 0],
            X_set[y_set == cls, 1],
            c=cmap_points(i),
            label=f"Class {cls}",
            edgecolor="k",
            alpha=0.8,
        )

    plt.title(title)
    plt.xlabel("Age (scaled)")
    plt.ylabel("Estimated Salary (scaled)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

# =========================
# 8) Visualize training and test results (scaled space)
# =========================
plot_decision_regions(X_train_scaled, y_train, "SVM (Training set) - Decision Regions")
plot_decision_regions(X_test_scaled, y_test, "SVM (Test set) - Decision Regions")
 Project Description: Support Vector Machine Classification on Social Network Ads

This project implements a Support Vector Machine (SVM) classifier to predict whether a user will purchase a product based on their Age and Estimated Salary.

The dataset contains customer demographic information along with a binary target variable Purchased (0 or 1), where:

0 → User did not purchase

1 → User purchased

Objective

The goal of this project is to:

Understand how SVM works for binary classification

Visualize decision boundaries

Evaluate model performance using a confusion matrix

Observe how feature scaling affects SVM performance

⚙️ Methodology

Data Loading

The dataset is imported using pandas.

Features selected: Age and EstimatedSalary

Target variable: Purchased

Data Visualization

A scatter plot of the raw data is created to observe class distribution.

This helps understand how separable the data is.

Train-Test Split

The dataset is split into training (75%) and testing (25%) sets.

Feature Scaling

Standardization is applied using StandardScaler.

This is essential because SVM is sensitive to feature magnitudes.

Model Training

A linear SVM classifier (SVC(kernel='linear')) is trained on the scaled training data.

The model finds the optimal hyperplane that maximizes the margin between classes.

Prediction & Evaluation

Predictions are made on the test set.

A confusion matrix and accuracy score are computed to evaluate performance.

Decision Boundary Visualization

The decision regions are plotted for both training and test sets.

This visually demonstrates how SVM separates the two classes.

🧠 Why SVM?

SVM was chosen because:

It maximizes the margin between classes.

It performs well on medium-sized datasets.

It is effective in high-dimensional spaces.

It provides strong theoretical guarantees through margin maximization.

📊 Expected Outcome

The model should learn a linear boundary that separates users likely to purchase from those who are not, based on age and salary patterns.

🚀 Skills Demonstrated

Data preprocessing

Feature scaling

Binary classification

SVM implementation

Model evaluation

Decision boundary visualization

Understanding of margin-based classifiers

If you'd like, I can also write:

A more beginner-friendly version

A more technical version for LinkedIn/GitHub

Or an interview-style explanation of this project

You're building solid ML fundamentals 🔥

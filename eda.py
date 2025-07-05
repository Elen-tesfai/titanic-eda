import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Load Data (CSV files are now in 'data' folder)
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
gender_submission = pd.read_csv('data/gender_submission.csv')

# Print the first few rows of each dataset
print("\nTraining Data:")
print(train_data.head())

print("\nTest Data:")
print(test_data.head())

print("\nGender Submission Data:")
print(gender_submission.head())

# 1. Missing Values
print("\nMissing values in the training dataset:")
print(train_data.isnull().sum())

print("\nMissing values in the test dataset:")
print(test_data.isnull().sum())

print("\nMissing values in the gender submission dataset:")
print(gender_submission.isnull().sum())

# 2. Filling Missing Values
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# 3. Descriptive Statistics
print("\nDescriptive Statistics of the Training Data:")
print(train_data.describe())

# 4. Data Preprocessing
# Encode categorical variables (e.g., Sex and Embarked)
categorical_cols = ['Sex', 'Embarked']
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']

# Create preprocessing pipelines for both categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

# 5. Model Pipeline
# Define the model (RandomForestClassifier)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Train the Model
X = train_data.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'])
y = train_data['Survived']

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# 7. Evaluate the Model
y_pred = model_pipeline.predict(X_val)
print(f"\nModel Accuracy: {accuracy_score(y_val, y_pred)}")

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
cr = classification_report(y_val, y_pred)
print("\nClassification Report:")
print(cr)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Did not survive', 'Survived'], yticklabels=['Did not survive', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 8. Hyperparameter Tuning (Optional)
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_val)
print(f"\nModel Accuracy after Hyperparameter Tuning: {accuracy_score(y_val, y_pred_best)}")

# 9. Feature Importance
importances = best_model.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# 10. Prediction on Test Data
X_test = test_data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
y_test_pred = model_pipeline.predict(X_test)

# Prepare the submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': y_test_pred
})

print("\nGender Submission Data:")
print(submission.head())

# Save the final predictions to a CSV file
submission.to_csv('titanic_predictions.csv', index=False)

# 11. Visualizing the Results (Optional)
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Plot survival by Sex
sns.countplot(data=train_data, x='Survived', hue='Sex')
plt.title('Survival Count by Sex')
plt.show()

# Plot survival by Pclass
sns.countplot(data=train_data, x='Survived', hue='Pclass')
plt.title('Survival Count by Pclass')
plt.show()

# Plot survival by Age
plt.figure(figsize=(10, 6))
sns.histplot(train_data[train_data['Survived'] == 1]['Age'], kde=True, color='green', label='Survived', bins=20)
sns.histplot(train_data[train_data['Survived'] == 0]['Age'], kde=True, color='red', label='Did not survive', bins=20)
plt.legend()
plt.title('Survival Distribution by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Plot Fare vs Survival
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Fare', data=train_data)
plt.title('Fare vs Survival')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.show()

# Plot correlation matrix for numerical features
corr_matrix = train_data[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
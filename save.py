import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV

# Load dataset
df = pd.read_csv('./Dataset-Mental-Disorders.csv', index_col='Patient Number')

# Preprocessing
df.loc[df[df['Suicidal thoughts'] == 'YES ']['Suicidal thoughts'].index[0],'Suicidal thoughts'] = 'YES'

# Encoding categorical features
encoders = {} 
for col in df.columns[:-1]:
    encoders[col] = LabelEncoder()
    encoders[col].fit(df[col])
    df[col] = encoders[col].transform(df[col])

# Feature selection
feature_selection = SelectKBest(score_func=mutual_info_classif)
feature_selection = feature_selection.fit(df[df.columns[:-1]], df[df.columns[-1]])

# Get the indices of top 5 features based on their scores
selected_feature_indices = feature_selection.scores_.argsort()[-5:][::-1]

# Get the names of the selected features
selected_features = df.columns[:-1][selected_feature_indices]
print("Selected Features:", selected_features)  # Print selected features here

# Define features and target variable
X = df[selected_features]
y = df[df.columns[-1]]

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, shuffle=True)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [120],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [10, 12],
    'n_estimators': [300, 1000]
}

model = RandomForestClassifier()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Save the trained model
joblib.dump(grid_search.best_estimator_, 'trained_model.pkl')

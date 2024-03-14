import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Reading the dataset
df = pd.read_csv('./Dataset-Mental-Disorders.csv')
for col in df.columns:
    print(col, df[col].unique())

# Preprocessing
# Replace 'YES ' with 'YES' in 'Suicidal thoughts' column
df.loc[df[df['Suicidal thoughts'] == 'YES ']['Suicidal thoughts'].index[0],'Suicidal thoughts'] = 'YES'

# Encoding categorical features
encoders = {} 
for col in df.columns[:-1]:
    encoders[col] = LabelEncoder() # define individual encoder for each column
    encoders[col].fit(df[col])
    df[col] = encoders[col].transform(df[col])

# Selecting relevant features based on your specified features
selected_features = ['Suicidal thoughts', 'Mood Swing', 'Sadness', 'Aggressive Response', 'Euphoric', 'Expert Diagnose']

# Defining input features and target variable
X = df[selected_features]
y = df['Expert Diagnose']

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, shuffle=True)

# Model training
model = RandomForestClassifier(n_estimators=300, max_depth=120, max_features=3, min_samples_leaf=3, min_samples_split=10, bootstrap=True)
model.fit(x_train, y_train)

# Model evaluation
y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))

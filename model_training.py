import pandas as pd

# training data read from the csv file
df = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv")

# print the first 5 rows of the dataframe
print(df.head())

# size check
print("Number of rows and columns:", df.shape)

# missing values check
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing categorical values with mode
for column in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Fill missing numerical value with median
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Verify again
print("\nMissing values after filling:")
print(df.isnull().sum())

# convert categorical variables to numerical variables
from sklearn.preprocessing import LabelEncoder

# Create a label encoder
encoder = LabelEncoder()

# List of categorical columns to encode
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education',
                    'Self_Employed', 'Property_Area', 'Loan_Status']

# Encode each column
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# View the changes
print("\nAfter encoding:")
print(df.head())

# Drop columns we don't need
X = df.drop(columns=["Loan_ID", "Loan_Status"])  # Features
y = df["Loan_Status"]                            # Target

print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", y.shape)


from sklearn.model_selection import train_test_split

# Split the data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel accuracy on test set:", round(accuracy * 100, 2), "%")


##########################
# Save the trained model #
##########################

import joblib

# Save the trained model to a file
joblib.dump(model, "loan_model.pkl")
print("Model saved to loan_model.pkl")

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Start an MLflow experiment
mlflow.set_experiment("test_experiment")

# Load and prepare dataset
data_path = "C:\\Users\\Evelyn\\azure-ml-cicd\\production\\mydata.csv"
print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)
print("Dataset loaded successfully!")
print(df.head())

# Extract features and target
X = df[['x-axis', 'y-axis', 'z-axis']].values
y = df['activity'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Train model
with mlflow.start_run(run_name="RandomForestClassifier"):
    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Log additional metrics
    mlflow.log_metric("accuracy", accuracy)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Step 1: Load data
train_data = pd.read_excel('DataForPerceptron.xlsx', sheet_name='TRAINData')
test_data = pd.read_excel('DataForPerceptron.xlsx', sheet_name='TESTData')

# Step 2: Handle missing values in training data
train_data = train_data.dropna()

# Step 3: Separate features (X) and target (y)
X_train = train_data.drop(columns=['SubjectID', 'Class'])
y_train = train_data['Class'].values  # Convert to numpy array for compatibility

X_test = test_data.drop(columns=['SubjectID', 'Class'])
y_test_ids = test_data['SubjectID']  # Store SubjectID for output file

# Step 4: Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Implement perceptron learning algorithm
class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Convert labels to {1, -1}
        y = np.where(y == y[0], 1, -1)

        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Training loop
        for _ in range(self.max_iter):
            for i in range(len(X)):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0 else -1

                # Update weights and bias if prediction is incorrect
                if y[i] != y_predicted:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)

# Step 6: Train perceptron model and obtain the hypothesis
model = Perceptron(learning_rate=0.01, max_iter=1000)
model.fit(X_train, y_train)

# Hypothesis: The model parameters (weights and bias) represent the hypothesis
print("Hypothesis (weights):", model.weights)
print("Hypothesis (bias):", model.bias)

# Step 7: Predict on training data to evaluate performance
y_train_pred = model.predict(X_train)
y_train_pred_original = np.where(y_train_pred == 1, y_train[0], np.unique(y_train)[1])  # Map back to original labels
training_accuracy = accuracy_score(y_train, y_train_pred_original)
print(f"Training Accuracy: {training_accuracy * 100:.2f}%")

# Step 8: Predict class values for TESTData
y_test_pred = model.predict(X_test)
y_test_pred_original = np.where(y_test_pred == 1, y_train[0], np.unique(y_train)[1])  # Map back to original labels

# Step 9: Save results to a DataFrame
test_results = pd.DataFrame({
    'SubjectID': y_test_ids,
    'Predicted': y_test_pred_original
})
test_results.to_excel('TestResults.xlsx', index=False)
print("Test results saved to 'TestResults.xlsx'")

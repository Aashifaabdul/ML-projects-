from sklearn import datasets

# Load Breast Cancer dataset
cancer = datasets.load_breast_cancer()
print(cancer.feature_names)
print(cancer.target_names)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=1)
print(X_train.shape, X_test.shape)

from sklearn.linear_model import LogisticRegression
# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

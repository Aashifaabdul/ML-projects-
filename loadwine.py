from sklearn import datasets

# Load Wine dataset
wine = datasets.load_wine()
print(wine.feature_names)
print(wine.target_names)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=1)
print(X_train.shape, X_test.shape)

from sklearn.linear_model import LogisticRegression
# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print(y_pred)

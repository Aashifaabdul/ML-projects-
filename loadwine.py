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


OUTPUT:
['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
['class_0' 'class_1' 'class_2']
(142, 13) (36, 13)
[2 1 0 1 0 2 1 0 2 1 0 1 1 0 1 1 2 0 1 0 0 1 1 1 0 2 0 0 0 2 1 2 2 0 1 1]

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy

OUTPUT:
0.9444444444444444

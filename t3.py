from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import cv2
import os
import numpy as np
import pandas as pd


train_data_dir = r'dogs-vs-cats\train'
test_data_dir = r'dogs-vs-cats\test1'

categories = ['Cat', 'Dog']


def create_training_data(data_dir):
    training_data = []
    for img in os.listdir(data_dir):
        try:
            category = 'Dog' if 'dog' in img else 'Cat'
            class_num = categories.index(category)
            img_array = cv2.imread(os.path.join(data_dir, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (50, 50))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass
    return training_data


training_data = create_training_data(train_data_dir)


np.random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, 50 * 50)


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC()

model.fit(X_train, y_train)
test_predictions = model.predict(X_test)

correct = sum(test_predictions == y_test)
total = len(y_test)
accuracy = correct / total

odds = accuracy ** 12

print(f"Accuracy: {accuracy * 100}%")
print(f"Odds of breaking the Asirra CAPTCHA: {odds}")

submission = pd.DataFrame({
    'id': range(1, len(test_predictions) + 1),
    'label': test_predictions
})


submission.to_csv('t3submission.csv', index=False)
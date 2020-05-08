# ML testing
Библиотека тестов для валидации модели.

Валидационные тесты заключаются в расчёте метрик качества модели на обучающей и отложенной выборках для проверки качества и стабильности модели.

* Classification Tests
    * ROC AUC score
    * F1 score
* Regression Tests
    * R2 score
    * RMSE
    
## Сборка и установка
Убедитесь, что у вас установлены последние версии setuptools и wheel:
```shell script
python3 -m pip install --user --upgrade setuptools wheel
```
Теперь запустите эту команду из той же директории, где находится setup.py:
```shell script
python3 setup.py sdist
```
После ее завершения должен сгенерироваться файл в каталоге dist:
```shell script
dist/
  ml-testing-0.0.1.tar.gz
```
Для установки пакета выполните следующую команду из той же директории, где находится ml-testing-0.0.1.tar.gz:
```shell script
pip install $(pwd)/ml-testing-0.0.1.tar.gz
```

## Примеры использования
## Classification Tests
```shell script
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# Shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
classifier.fit(X_train, y_train)

y_train_pred = classifier.decision_function(X_train)
y_test_pred = classifier.decision_function(X_test)
```
Для расчета метрики ROC AUC используйте обучающую и отложенную выборки, а также результаты раюоты вашей модели.
```shell script
from ml_testing.classification_tests import test_f1_score, test_roc_auc_score
test_roc_auc_score(y_train, y_train_pred, y_test, y_test_pred)
```
Результат:
```shell script
Train ROC AUC = 1.000, holdout ROC AUC = 0.768
```
Расчет F1 score:
```shell script
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)
test_f1_score(y_train, y_train_pred, y_test, y_test_pred, average='micro')
```
Результат:
```shell script
Train f1 = 1.000, holdout f1 = 0.396
```

## Regression Tests 
```shell script
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.33, random_state=42)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_train_pred = regr.predict(X_train)
y_test_pred = regr.predict(X_test)
```
Для расчета метрик R^2, RMSE используйте соответствующие методы:
```shell script
from ml_testing.regression_tests import test_r2_score, test_rmse
test_r2_score(y_train, y_train_pred, y_test, y_test_pred)
test_rmse(y_train, y_train_pred, y_test, y_test_pred)
```
Результат:
```shell script
Train Coefficient of determination = 0.510, Holdout Coefficient of determination = 0.510
Train RMSE = 54.307, Holdout RMSE = 53.083
```


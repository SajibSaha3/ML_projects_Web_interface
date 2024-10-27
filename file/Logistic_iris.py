import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

class MachineLearningModel:
    def train(self):
        df = pd.read_csv('iris.csv')
        df.fillna(0, inplace=True)  # specify a value or method to fill NaNs
        y = df['species']
        x = df.drop(['Unnamed: 0', 'species'], axis=1)
        clf = LogisticRegression(random_state=0).fit(x, y)
        with open('logistic_regression_model.pickle', 'wb') as files:
            pickle.dump(clf, files)

    def test(self, sepal_length, sepal_width, petal_length, petal_width):
        with open('logistic_regression_model.pickle', 'rb') as f:
            clf = pickle.load(f)
        y_pred = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        return y_pred[0]

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)



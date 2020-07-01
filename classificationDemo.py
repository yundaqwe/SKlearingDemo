import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
X,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(type(X))
print("hello")
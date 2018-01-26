from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
iris=load_iris()
n_samples,n_features=iris.data.shape
x=iris.data
y=iris.target
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y)
y_new=knn.predict(np.array([[3,5,4,2],[5,4,3,2]]))
print(y_new)
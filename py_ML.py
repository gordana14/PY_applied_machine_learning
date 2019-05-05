# -*- coding: utf-8 -*-
"""
Spyder Editor


This is a temporary script file.
"""


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap, BoundaryNorm


# new
import seaborn as sn
from sklearn import neighbors
import matplotlib.patches as mpatches
from sklearn.tree import export_graphviz
import matplotlib.patches as mpatches

# auxillary
def plot_fruit_knn(X, y, n_neighbors, weights):
    if isinstance(X, (pd.DataFrame,)):
        X_mat = X[['height', 'width']].as_matrix()
        y_mat = y.values
    elif isinstance(X, (np.ndarray,)):
        # When X was scaled is already a matrix
        X_mat = X_mat[:, :2]
        y_mat = y.values
        print(X_mat)

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#AFAFAF'])

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_mat, y_mat)

    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.

    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    # numpy.c_ Translates slice objects to concatenation along the second axis
    # e.g. np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
    # ravel() Returns a contiguous flattened array.
    # x = np.array([[1, 2, 3], [4, 5, 6]])
    # np.ravel(x) = [1 2 3 4 5 6]

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor = 'black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    patch0 = mpatches.Patch(color='#FF0000', label='apple')
    patch1 = mpatches.Patch(color='#00FF00', label='mandarin')
    patch2 = mpatches.Patch(color='#0000FF', label='orange')
    patch3 = mpatches.Patch(color='#AFAFAF', label='lemon')
    plt.legend(handles=[patch0, patch1, patch2, patch3])


    plt.xlabel('height (cm)')
    plt.ylabel('width (cm)')

    plt.show()

fruits= pd.read_table('C:/Users/Marica/Desktop/FAKULTET/svasara folder/Downloads/Machine-Learning-with-Python-master/Machine-Learning-with-Python-master/fruit_data_with_colors.txt')

""" 
fruits.head()

"""

# create traint test split 

X=fruits[['mass', 'width', 'height']]
Y=fruits['fruit_label']
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, random_state=0)

# visualization
""" Doesn't work but is very useful 
cmap= cm.get_cmap('gnuplot')
scatter= pd.scatter_matrix(X_train, c=Y_train, marker='o', s=40, hist_kwds={'bins':15}, figsize={12,12}, cmap=cmap)
plt.show()
"""
"""
Each point represents a single piece of fruit and its color according to its fruit label  value.
Conclusion: Diff fruit types are in prety well-defined clusters that arealso well-separated in feature space. 
"""
fig=plt.figure()
ax= fig.add_subplot(111, projection='3d')
ax.scatter(X_train['width'], X_train['height'], X_train['mass'], c=Y_train, marker='o', s=100)
ax.set_xlabel('width')
ax.set_xlabel('height')
ax.set_xlabel('mass')
plt.show()



# K- NN nearest neighbors a simple classification task 

lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique(),  ))
knn= KNeighborsClassifier(n_neighbors=5)
# Train the classifier using training data
knn.fit(X_train, Y_train)
# estimate the accuracy of the classifier on the future data using test data
knn.score(X_test, Y_test)
# using the trained k-NN classifier model to classify new, previously unseen objects
fruit_prediction = knn.predict([[20, 4.3, 5.5]])
lookup_fruit_name[fruit_prediction[0]]
# you could put diff values

#Plot the decision boundaries of the K-NN classifier
# 5 must be defined; uniform can be changed for Euclidian or something by yourself 
plot_fruit_knn(X_train, Y_train, 5, 'uniform')

#Simple regression dataset
"""
x axis - shows the future values 
y axis - ahows regression target
"""
from sklearn.datasets import make_regression
X_r1 , Y_r1 =make_regression(n_samples =100, n_features=1, n_informative=1, bias = 150.0,noise=30, random_state=0)

plt.scatter(X_r1, Y_r1, marker='o', s=50)
plt.show()

# New dataset
#from adspy_shared_utilities import load_crime_dataset
#crime=load_crime_dataset()

#linear regression
"""
from sklearn.linear_model import LinearReggression

ImportError: cannot import name 'LinearReggression' from 'sklearn.linear_model'
linreg = LinearReggression().fit(X_train, Y_train)

print('coef w is '.format(linreg.coef_) )
print('coef b is '.format(linreg.intercept_))

#plot 
plt.figure(figuresize=(5,4))
plt.scatter(X_r1, Y_r1, marker='o', s=50, alpha=0.8)
plt.plot(X_r1, linreg.coef_*X_r1+linreg.intercept_, 'r-')
plt.show()

"""



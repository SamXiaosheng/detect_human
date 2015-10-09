import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Feature vector
X = np.array([[0, 0], [1, 1], [3, 3], [2, 2]])

# Training classification
y = [0, 0, 1, 1]

# Creating the svm and train
clf = svm.SVC(kernel='rbf', gamma=0.7)
clf.fit(X, y)
print clf
print X
print clf.predict([[2., 2.]])


# create a mesh to plot in
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

title = 'SVC plot'
print 'xx', xx
print 'xx.', xx.ravel()
print 'xx yy.', np.c_[xx.ravel(), yy.ravel()]

# list the data points as ordered pairs
# predict and display as color
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
print 'Z', Z

Z = Z.reshape(xx.shape)
print 'Z', Z
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.show()

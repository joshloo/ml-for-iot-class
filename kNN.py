# This code uses breast cancer data in SK learn
# as a classifier example to attempt predicting
# with a test step derived from the whole dataset

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as pl

pl.style.use('ggplot')

cancer = load_breast_cancer()
print(cancer.data)
print(cancer.target)

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    stratify=cancer.target, random_state=50)

train_acc   = []
test_acc    = []
k_neighbors = range(1,11)

for k in k_neighbors:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    train_acc.append( clf.score(X_train, y_train) )
    test_acc.append(  clf.score(X_test,  y_test ) )
                                        
pl.plot(k_neighbors, train_acc)
pl.plot(k_neighbors, test_acc)
pl.xlabel(r'$k$')
pl.ylabel('score')
pl.legend(['training accuracy', 'test accuracy'])
pl.savefig('kNN plot.png')
pl.show()

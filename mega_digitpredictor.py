# author : Loo Tung Lun

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

import glob
import os
import cv2
import numpy as np

# Adding mega classifier superset from sklearn library
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# The name and classifier list below must be aligned, else will have error.
names = [
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA"
            ]

classifiers = [
                KNeighborsClassifier(3),
                SVC(kernel="linear", C=0.025),
                SVC(gamma=0.001, C=1),
                GaussianProcessClassifier(1.0 * RBF(1.0)),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                MLPClassifier(alpha=1, max_iter=1000),
                AdaBoostClassifier(),
                GaussianNB(),
                QuadraticDiscriminantAnalysis()
                ]

# The digits dataset
digits = load_digits()

print("The Digit data set imported has " , len(digits.data) , " data.")
print("It comes in pixels format of 8x8, for example:")
print(digits.data[0])
print("The results are the digit itself: ")
print(digits.target)

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(digits.images, digits.target))

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

best_score = 0
current_score = 0
best_classifier = ''

for name, clf in zip(names, classifiers):
    # We learn the digits on the first half of the digits because test size is 0.5
    clf.fit(X_train, y_train)

    current_score = clf.score(X_test, y_test)
    if (current_score > best_score):
        best_score = current_score
        best_classifier = name

    print("Accuracy score for ", name, " : ", current_score)
    # Now predict the value of the digit on the second half:
    predicted = clf.predict(X_test)

    images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
    
    for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Training: %i' % label)
    for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Prediction: %i' % prediction)

    # filename = 'output/Precision_With' + name + '.png'
    # plt.savefig(filename)

    # print("Classification report for classifier %s:\n%s\n"
          # % (clf, metrics.classification_report(y_test, predicted)))

    # Uncomment below to generate confusion matrix
    #
    disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    # print("Confusion matrix:\n%s" % disp.confusion_matrix)
    filename = 'output/ConfusionMatrix_With' + name + '.png'
    plt.savefig(filename)
    # print("#################################")


print("With all the classifier compared, the best one for this data set is ")
print(best_classifier + " with accuracy of " + str(best_score))

img_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(img_dir,'*png') 
files = glob.glob(data_path) 
data = []
i=0
for f1 in files:
    i = i+1
    img = cv2.imread(f1)
    cv2.imshow(str(f1) ,img)

cv2.waitKey(0)
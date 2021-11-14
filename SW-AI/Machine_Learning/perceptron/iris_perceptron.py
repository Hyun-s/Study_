from sklearn import datasets
import numpy as np



LF = '\n'
iris = datasets.load_iris()

# check the target features of the dataset
print( iris['target_names'])
print( ['target'])

# see the descriptive features of the sample
print( iris['data'])
print( iris['data'].shape)

X = iris.data[:, [2, 3]]
y = iris.target

# class의 대소관계가 미치는 영향을 알아보기 위한 실험 코드
# y = np.where(y==2,3,y)
# y = np.where(y==1,2,y)
# y = np.where(y==3,1,y)
# y = np.where(y!=1,-1,y)
print( X, '\n', y)

#----------------------------------------------

# pima dataset
# 출처 https://www.kaggle.com/uciml/pima-indians-diabetes-database
# load data, data dimension 8 -> 2 (pca)
# import pandas as pd
# from sklearn.decomposition import PCA
# pima = pd.read_csv('pima/diabetes.csv')
#
# pca = PCA(n_components=2)
# X = pima[pima.columns[:-1]].to_numpy()
# y = pima['Outcome'].to_numpy()
# y = np.where(y==0,-1,y)
# X = pca.fit_transform(X)

#----------------------------------------------
# drug dataset
# 출처 https://www.kaggle.com/prathamtripathi/drug-classification
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.decomposition import PCA
#
# df = pd.read_csv('drug/drug200.csv')
# cols = ['Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']
# le = LabelEncoder()
# pca = PCA(n_components=2)
#
# for col in cols:
#     df[col] = le.fit_transform(df[col])
#
# X = df[df.columns[:-1]].to_numpy()
# X = pca.fit_transform(X)
#
# y = df[df.columns[-1]].to_numpy()
#
# from sklearn.datasets import make_moons
#
# X, y = make_moons(n_samples=400, noise=0.1, random_state=0)
#----------------------------------------------

# # emnist dataset
# import pandas as pd
# from sklearn.decomposition import PCA
# X = pd.read_csv('./emnist_data/train.csv')
# test = pd.read_csv('./emnist_data/test.csv')
#
# y = X['digit'].to_numpy()
# letter = X['letter'].to_numpy()
# X = X[[str(x) for x in range(784)]].to_numpy()
#
# pca = PCA(n_components=2)
# X = pca.fit_transform(X)

#----------------------------------------------

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=0)
#----------------------------------------------

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print( X_train_std[0], LF, X_train[0])

#----------------------------------------------

from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)

# from perceptron import Perceptron, ovr_perceptron, my_mlp
#
# # # one vs rest classifier
# # ppn1 = Perceptron(n_iter=40, eta=0.01)
# # ppn2 = Perceptron(n_iter=40, eta=0.01)
# # ppn3 = Perceptron(n_iter=40, eta=0.01)
# # ppn = ovr_perceptron([ppn1,ppn2,ppn3])
#
# # multi layer perceptron
# ppn = my_mlp(n_iter=2000,hidden_unit=3)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

#----------------------------------------------

from sklearn.metrics import (accuracy_score,precision_score,
                             recall_score, confusion_matrix,f1_score)
print('----------measure metrics----------')
def get_clf_eval(y_test,pred, average='macro'):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test, pred, average=average)
    recall = recall_score(y_test, pred, average=average)
    f1 = f1_score(y_test, pred, average=average)
    print('confusion matrix')
    print(confusion)
    print('acc: {0:.2f}, precision: {1:.2f}, recall: {2:.2f}, F1: {3:.2f}'\
          .format(accuracy, precision, recall, f1))
get_clf_eval(y_test,y_pred)

import cp_utils as cpu
import matplotlib.pyplot as plt

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


cpu.plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105,150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.savefig('v6.png')
plt.show()


print( 'End of Program')

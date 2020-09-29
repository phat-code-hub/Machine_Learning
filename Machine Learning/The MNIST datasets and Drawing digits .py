# -*- coding: utf-8 -*-
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
X,y=load_digits(n_class=2,return_X_y=True)
print(X[0].reshape(8,8))

# plt.matshow(X[0].reshape(8,8),cmap=plt.cm.gray_r)
plt.matshow(X[0].reshape(8,8),cmap="Greys")
# plt.axis('off')
plt.xticks(())
plt.yticks(())
plt.show()
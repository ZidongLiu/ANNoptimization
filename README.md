# ANNoptimization
This code implement `Simulated Annealing`, `Stochastic Gradient descent` and 
`Conjugate gradient descents` algorithm to optimize the Artificial Neural Networks. 

# Structure of the networks
This is the general BP network

# Example
The following code is available in `demo.py`
```
from sklearn import datasets
import numpy as np
import class_ANN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

##### import iris data
iris = datasets.load_iris()
iris_X = iris['data']
iris_X = np.transpose(iris_X)
target = iris['target']
iris_Y = np.zeros((3,150))

### transform the data into the shape we need
### data should be column-wise, each column of data is an instance
for i in range(150):
    iris_Y[target[i],i] = 1

myANN = class_ANN.ANN([4,5,3],[1,1])

## fit the data
myANN.BP_fit(iris_X,iris_Y)

#run this line of code if you want you use the the SA algorithm
## myANN.SA_fit(iris_X,iris_Y)

# run the following line of code if CG algorithm is required
## myANN.CG_fit(iris_X,iris_Y)

#### plot the learning histroty
plt.plot(myANN.train_error[0,:])
plt.show()
#### show the confusion matrix
iris_pred = myANN.predict(iris_X)
iris_class = np.argmax(iris_pred,axis=0)
confusion_matrix(target,iris_class)
```

import urllib, csv
import numpy as np

def computeCost(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha=0.01, iters=75):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
response = urllib.urlopen(url)

datapoints = []
X = []
y = []
class_map = {
	'Iris-setosa': 1,
	'Iris-versicolor': 2,
	'Iris-virginica': 3
}
for row in response.read().split("\n")[:-2]:
	datapoint = row.split(",")
	X.append(datapoint[:4])
	y.append(class_map[datapoint[4]])
	datapoints.append(datapoint)

cols = len(datapoints[0])
X = np.array(X,dtype=float)
y = np.array(y,dtype=float)
theta = np.matrix(np.array([0]*4,dtype=float))

X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()
print X,y

print theta

g, cost = gradientDescent(X, y, theta)

print g,cost

print "%0.5f" % computeCost(X, y, g)

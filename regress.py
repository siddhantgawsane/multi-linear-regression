import urllib
from random import randrange, shuffle
from math import sqrt
from pylab import arange, plot, title, grid, show

def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def predict(row, coefficients):
	b0 = coefficients[0]
	for i in range(len(row)-1):
		b0 += coefficients[i + 1] * row[i]
	return b0
 
def coefficients_sgd(train, learning_rate=0.001, n_epoch=100):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			b0 = predict(row, weights)
			error = b0 - row[-1]
			sum_error += error**2
			weights[0] = weights[0] - learning_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] - learning_rate * error * row[i]
		# print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
	return weights


def cross_validation_split(dataset, n_folds=5):
	dataset_split = []
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	shuffle(dataset_copy)
	for i in range(n_folds):
		fold = []
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = []
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = []
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		# print predicted
		# print actual
		t = arange(0.0, 30.0, 1)
		# s = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
		# s2 = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
		plot(t, actual)
		plot(t, predicted)
		 
		# xlabel('Item (s)')
		# ylabel('Value')
		title('Predicted vs Actual Regression values for Fold %d' % (folds.index(fold)+1))
		grid(True)
		show()

		rmse = calc_rmse(actual, predicted)
		scores.append(rmse)
	return scores

def linear_regression(train, test, learning_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, learning_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		predictions.append(yhat)
	return predictions

def calc_rmse(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
response = urllib.urlopen(url)

datapoints = []
class_map = {
	'Iris-setosa': 1,
	'Iris-versicolor': 2,
	'Iris-virginica': 3
}
for row in response.read().split("\n")[:-2]:
	datapoint = row.split(",")
	datapoint[-1] = class_map[datapoint[-1]]
	datapoint = [float(x) for x in datapoint]
	datapoints.append(datapoint)

minmax = [[0, 10]]*4 + [[1, 3]]
normalize_dataset(datapoints, minmax)

n_folds = 5
learn_rate = 0.01
n_epoch = 100

scores = evaluate_algorithm(datapoints, linear_regression, n_folds, learn_rate, n_epoch)

print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores)/float(len(scores))))

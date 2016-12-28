'''
Intro to ML - HW2
Ofer Orgal 300459898
oferorgal@mail.tau.ac.il
'''
from PIL import Image
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from numpy import *
import numpy.random
from sklearn.datasets import fetch_mldata
from astropy.table import Table, Column
import sklearn.preprocessing

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0,8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_unscaled = data[train_idx[:6000], :].astype(float)
train_labels = (labels[train_idx[:6000]] == pos)*2-1

validation_data_unscaled = data[train_idx[6000:], :].astype(float)
validation_labels = (labels[train_idx[6000:]] == pos)*2-1

test_data_unscaled = data[60000+test_idx, :].astype(float)
test_labels = (labels[60000+test_idx] == pos)*2-1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)

train_data_norm = sklearn.preprocessing.normalize(train_data, norm='l2', axis=1, copy=True)
validation_data_norm = sklearn.preprocessing.normalize(validation_data, norm='l2', axis=1, copy=True)
test_data_norm = sklearn.preprocessing.normalize(test_data, norm='l2', axis=1, copy=True)

def main():

	#1a
	#run_perceptron()

	#1b
	#print_image("weights.png", train(train_data_norm, train_labels), 200)
	
	#1c
	#print "The Accuracy of the Algorithm:" ,(accuracy(test_data_norm, test_labels, train(train_data_norm, train_labels))*100), "%"
	
	#1d
	index, digit = find_misclassified()
	print_image("misclassified_%d.png" %digit[0], test_data_unscaled[index[0]], 1)
	print_image("misclassified_%d.png" %digit[1], test_data_unscaled[index[1]], 1)
	return

'''
1d
'''
def find_misclassified():
	index = [0]*2
	digit = [0]*2
	w = train(train_data_norm, train_labels)
	for i in range(0,len(test_data_norm)):
		if(predict(test_data_norm[i],w) != test_labels[i]):
			if (index[0] == 0):
				index[0] = i
				digit[0] = train_labels[i]
			else:
				if(digit[0] != train_labels[i]):
					index[1] = i
					digit[1] = train_labels[i]
	digit[0] = 0 if digit[0] == -1 else 8
	digit[1] = 0 if digit[1] == -1 else 8
	return index, digit

'''
1a
'''
def run_perceptron_X_times(input_data, input_labels, times):
	accuracyVector = [0.0]*times
	for i in range(0,times):
		temp = zip(input_data, input_labels)
		numpy.random.shuffle(temp)
		data, labels = zip(*temp)
		w = train(data, labels)
		accuracyVector[i] = accuracy(test_data, test_labels, w)
		#print i
	accuracyVector.sort()
	return sum(accuracyVector)/times, accuracyVector[int(times*0.95)], accuracyVector[int(times*0.05)]

def run_perceptron():
	t = Table(names =("Sample Size", "Accuracy %", "percentile 95%", "percentile 5%"))
	sample_size = [5, 10, 50, 100, 500, 1000, 5000]
	for i in sample_size:
		mean, percentile_95, percentile_05 = run_perceptron_X_times(train_data_norm[0:i], train_labels[0:i], 100)
		print i, mean, percentile_95, percentile_05
		t.add_row([i, mean*100, percentile_95*100, percentile_05*100])
	t["Sample Size"] = t["Sample Size"].astype(int)
	print t
	return

def accuracy(data, labels, weights):
	accuracy = 0
	for i in range(0,len(data)):
		accuracy += 1 if labels[i] == predict(data[i],weights) else 0
	return float(accuracy)/len(data)

def predict(data, weights):
	prediction = 0
	for i in range(0,len(data)):
		prediction += weights[i]*data[i]
	return 1 if prediction >= 0 else -1
	return

def train(data_norm, labels):
	weights = [0.0]*len(data_norm[0])
	for i in range(0,len(data_norm)):
		prediction = predict(data_norm[i], weights)
		error = (labels[i] - prediction)/2
		weights += error*data_norm[i]
	return weights

def print_image(name, array, mult):
	w, h = 28, 28
	data1 = numpy.zeros((h, w, 3), dtype=numpy.uint8)
	for i in range(0,h):
		for j in range(0,w):
			temp = array[(i*h+j)]
			if temp < 0:
				data1[i,j] = [0,0,-temp*mult]
			elif temp > 0:
				data1[i,j] = [0,temp*mult,0]
			else:
				data1[i,j] = [255,0,0]
	img = Image.fromarray(data1, 'RGB')
	img.save(name)
	#img.show()
	return


if __name__ == '__main__':
    main()

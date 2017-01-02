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
#from astropy.table import Table, Column
import sklearn.preprocessing
from sklearn import svm

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

	if(len(sys.argv) < 2):
		print "Input format:\n \
        hw2.py 1a // Perceptron: calc the accuracy with 5, 10, 50, 100, 500, 1000, 5000 sample sizes\n \
        hw2.py 1b // Perceptron: Print an image of the weights\n \
        hw2.py 1c // Perceptron: Train perceptron on the full training set and test the accuracy on the test set\n \
        hw2.py 1d // Perceptron: Find and print 2 misclassified images \n \
        hw2.py 2a // SVM: Train SVM and find the best C value\n \
        hw2.py 2c // SVM: Print an image of the weights\n \
        hw2.py 2d // SVM: find the SVM accuracy for the test set\n \
        hw2.py 3a // SGD: Train the SGD and find the best eta value for the CV\n \
        hw2.py 3b // SGD: Train the SGD and find the best C value for the CV\n \
        hw2.py 3c // SGD: Print an image of the weights\n \
        hw2.py 3d // SGD: find the SGD accuracy for the test set\n"
		return

	#1a
	if sys.argv[1] == "1a":
		run_perceptron()

	#1b
	if sys.argv[1] == "1b":
		print_image("weights.png", train(train_data_norm, train_labels), 400)
	
	#1c
	if sys.argv[1] == "1c":
		print "The Accuracy of the Algorithm:" ,(Perceptron_accuracy(test_data_norm, test_labels, train(train_data_norm, train_labels))*100), "%"
	
	#1d
	if sys.argv[1] == "1d":
		index, digit = find_misclassified()
		print_image("misclassified_%d.png" %digit[0], test_data_unscaled[index[0]], 1)
		print_image("misclassified_%d.png" %digit[1], test_data_unscaled[index[1]], 1)
	
	#2a
	if sys.argv[1] == "2a":
		best_C = SVM_find_best_C()

	#2c
	if sys.argv[1] == "2c":
		best_C  = 1.00521990733
		SVM_weights(best_C)
	
	#2d
	if sys.argv[1] == "2d":
		best_C  = 1.00521990733
		clf = svm.LinearSVC(loss="hinge", fit_intercept=False, C=best_C)
		clf.fit(train_data_norm, train_labels)
		print "The accuracy of the linear SVM with C =", best_C, "is =", (clf.score(test_data_norm,test_labels)*100), "%"
	
	#3a
	if sys.argv[1] == "3a":
		best_eta = SGD()

	#3b
	if sys.argv[1] == "3b":
		best_eta = 19.19
		best_C = SGD_find_best_C(best_eta)

	#3c
	if sys.argv[1] == "3c":
		best_C = 0.001
		best_eta = 19.19
		SGD_weights(best_C, best_eta)

	#3d
	if sys.argv[1] == "3d":
		best_C = 0.001
		best_eta = 19.19
		accuracy = 0
		w = SGD_weights(best_C, best_eta)
		print "The accuracy of the my SGD with C =", best_C, "and eta =", best_eta, "is =", (SGD_accuracy(test_data, test_labels, w)*100), "%"
	

	return

'''
3c
'''
def SGD_weights(best_C, best_eta):
	T = 20000
	w = [0.0]*len(train_data[0])
	for t in range(1,T+1):
		i = random.choice(len(train_data))
		if(numpy.dot(w,train_data[i])*train_labels[i] < 1):
			w = numpy.multiply(1-(best_eta/t),w) + numpy.multiply((best_eta/t)*best_C*train_labels[i],train_data[i])
	print_image("SGDWeights.png", w, 20000)

	return w

'''
3b
'''
def SGD_find_best_C(eta):
	T = 1000
	RUNS = 10
	accuracy = [0.0]*RUNS
	c = 10e-10
	eta0 = eta
	cVec = []
	accuracyVec = []
	while c < 10e10:
		for run in range(0,10): 
			w = [0.0]*len(train_data[0])
			for t in range(1,T+1):
				i = random.choice(len(train_data))
				if(numpy.dot(w,train_data[i])*train_labels[i] < 1):
					w = numpy.multiply(1-(eta0/t),w) + numpy.multiply((eta0/t)*c*train_labels[i],train_data[i])
			accuracy[run] = SGD_accuracy(validation_data, validation_labels, w)
		cVec.append(c)
		accuracyVec.append(float(sum(accuracy))/RUNS)
		print c, float(sum(accuracy))/RUNS
		if(c >= 10e-5 and c < 10e-3):
			c *= 2
		else:
			c *= 10
	res = zip(cVec, accuracyVec)
	fig = plt.figure()
	plt.plot(cVec, accuracyVec, c = "blue")
	plt.xlabel('C Values')
	plt.xscale("log")
	plt.ylabel('Accuracy')         
	plt.title('SGD accuracy with different C values') 
	res.sort(key=operator.itemgetter(1))
	plt.axvline(x=res[len(res)-1][0], c="red", label = "Best C=%f" %res[len(res)-1][0])
	#plt.axhline(y=res[len(res)-1][1], c="red", label = "Best C=%f" %res[len(res)-1][1])
	plt.legend(loc="lower right")
	plt.savefig('plot3b')
	print "Best C:", res[len(res)-1][0], "Mean:", res[len(res)-1][1]
	return res[len(res)-1][0]


'''
3a
'''
def SGD():
	T = 1000
	RUNS = 10
	accuracy = [0.0]*RUNS
	C = 1
	eta0 = 10e-10
	etaVec = []
	accuracyVec = []
	while eta0 < 100:
		for run in range(0,10): 
			w = [0.0]*len(train_data[0])
			for t in range(1,T+1):
				i = random.choice(len(train_data))
				if(numpy.dot(w,train_data[i])*train_labels[i] < 1):
					w = numpy.multiply(1-(eta0/t),w) + numpy.multiply((eta0/t)*C*train_labels[i],train_data[i])
			accuracy[run] = SGD_accuracy(validation_data, validation_labels, w)
		etaVec.append(eta0)
		accuracyVec.append(float(sum(accuracy))/RUNS)
		print eta0, float(sum(accuracy))/RUNS
		if(eta0 < 1):
			eta0 *= 10
		else:
			eta0 *= 1.1
	res = zip(etaVec, accuracyVec)
	fig = plt.figure()
	plt.plot(etaVec, accuracyVec, c = "blue")
	plt.xlabel('eta Values')
	plt.xscale("log")
	plt.ylabel('Accuracy')         
	plt.title('SGD accuracy with different eta values') 
	res.sort(key=operator.itemgetter(1))
	plt.axvline(x=res[len(res)-1][0], c="red", label = "Best eta=%f" %res[len(res)-1][0])
	#plt.axhline(y=res[len(res)-1][1], c="red", label = "Best C=%f" %res[len(res)-1][1])
	plt.legend(loc="lower right")
	plt.savefig('plot3a')
	print "Best eta:", res[len(res)-1][0], "Mean:", res[len(res)-1][1]
	return res[len(res)-1][0]

def SGD_accuracy(data, labels, w):
	accuracy = 0
	for i in range(0,len(data)):
		accuracy += 1 if(labels[i] == SGD_predict(data[i], w)) else 0
	return float(accuracy)/len(data)

def SGD_predict(data, w):
	prediction = 0
	for i in range(0,len(data)):
		prediction += w[i]*data[i]
	return 1 if prediction >= 0 else -1
	return

'''
2c
'''
def SVM_weights(C):
	clf = svm.LinearSVC(loss="hinge", fit_intercept=False, C=C)
	clf.fit(train_data_norm, train_labels)
	weights = clf.coef_[0] 
	#print weights
	print_image("SVMWeights.png", weights, 200)
	return

'''
2a
'''
def SVM_find_best_C():
	cVec = []
	meanVec = []
	c = 10e-10
	while c < 10e10:
		clf = svm.LinearSVC(loss="hinge", fit_intercept=False, C=c)
		clf.fit(train_data_norm, train_labels)
		cVec.append(c)
		mean = clf.score(validation_data_norm,validation_labels)
		meanVec.append(mean)
		print c, mean
		if(c > 0.8	 and c < 1.1):
			c *= 1.01
		else:
			c *= 1.5
	res = zip(cVec, meanVec)
	fig = plt.figure()
	plt.plot(cVec, meanVec, c = "blue")
	plt.xlabel('C Values')
	plt.xscale("log")
	plt.ylabel('Accuracy')         
	plt.title('SVM accuracy with different C values') 
	res.sort(key=operator.itemgetter(1))
	plt.axvline(x=res[len(res)-1][0], c="red", label = "Best C=%f" %res[len(res)-1][0])
	#plt.axhline(y=res[len(res)-1][1], c="red", label = "Best C=%f" %res[len(res)-1][1])
	plt.legend(loc="lower right")
	plt.savefig('plot2a')
	print "Best C:", res[len(res)-1][0], "Mean:", res[len(res)-1][1]
	return res[len(res)-1][0]

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
		accuracyVector[i] = Perceptron_accuracy(test_data, test_labels, w)
		#print i
	accuracyVector.sort()
	return sum(accuracyVector)/times, accuracyVector[int(times*0.95)], accuracyVector[int(times*0.05)]

def run_perceptron():
	#t = Table(names =("Sample Size", "Accuracy %", "percentile 95%", "percentile 5%"))
	print "Sample Size    ", "Accuracy %    ", "percentile 95%    ", "percentile 5%    "
	sample_size = [5, 10, 50, 100, 500, 1000, 5000]
	for i in sample_size:
		mean, percentile_95, percentile_05 = run_perceptron_X_times(train_data_norm[0:i], train_labels[0:i], 100)
		print i,"         ", mean,"   ", percentile_95,"   ", percentile_05
		#t.add_row([i, mean*100, percentile_95*100, percentile_05*100])
	#t["Sample Size"] = t["Sample Size"].astype(int)
	#print t
	return

def Perceptron_accuracy(data, labels, weights):
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

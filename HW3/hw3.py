'''
Intro to ML - HW3
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
import sklearn.preprocessing

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

train_idx = numpy.random.RandomState(0).permutation(range(60000))

train_data_size = 50000
train_data_unscaled = data[train_idx[:train_data_size], :].astype(float)
train_labels = labels[train_idx[:train_data_size]]

validation_data_unscaled = data[train_idx[train_data_size:60000], :].astype(float)
validation_labels = labels[train_idx[train_data_size:60000]]

test_data_unscaled = data[60000:, :].astype(float)
test_labels = labels[60000:]

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)

def main():

	if(len(sys.argv) < 2):
		print "Input format:\n \
        hw3.py 6a // Find the best eta and C values for multiclass SVM\n \
        hw3.py 6b // Print the images of the weights\n \
        hw3.py 6c // Calc the accuracy of the SVM algorithm \n \
        hw3.py 7a // Find the best eta and C values for multiclass SVM with Kernel\n \
        hw3.py 7b // Calc the accuracy of the SVM with Kernel algorithm\n"
        
		return

	K = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	C = 1.0
	eta = 1.0
	T = 5000 #set to 10000

	#6a
	if sys.argv[1] == "6a":
		best_eta = find_best_eta(train_data, train_labels, C, T, K)
		#best_eta  = 4.39453125e-08
		best_C = find_best_C(train_data, train_labels, best_eta, T, K)
		print "Best C:", best_C, "Best eta:", best_eta

	#6b
	if sys.argv[1] == "6b":
		Best_C = 0.953674316406 
		Best_eta = 4.39453125e-08
		w = multiClassSDG(train_data, train_labels, Best_C, Best_eta, T, K)
	#print w[0]
		for i in K:
			print_image("weights_digit_%d.png" %i, w[i], 500000)

	#6c
	if sys.argv[1] == "6c":
		Best_C = 0.953674316406 
		Best_eta = 4.39453125e-08
		w = multiClassSDG(train_data, train_labels, Best_C, Best_eta, T, K)
		print "The Accuray of the algorithm on the test set is:", accuracy(test_data, test_labels, w, K)*100, "%"
	
	#7a
	if sys.argv[1] == "7a":
		best_eta = find_best_eta_with_Kernel(train_data, train_labels, C, T, K)
		#best_eta  = 10e-13
		best_C = find_best_C_with_Kernel(train_data, train_labels, best_eta, T, K)
		print "Best C:", best_C, "Best eta:", best_eta

	#7b
	if sys.argv[1] == "7b":
		Best_C = 1
		Best_eta = 10e-13
		alpha, kern_prod = multiClassSDG_with_Kernel(train_data, train_labels, Best_C, Best_eta, T, K)
		print "The Accuray of the algorithm on the test set is:", accuracy_with_kernel(test_data, test_labels, alpha, kern_prod, K)*100, "%"

	return;

#-----------------------------------------------------------------------------------#
def find_best_eta_with_Kernel(train, labels, C, T, K):
	print "find best eta value"
	eta = 10e-15
	etaVec = []
	accuracyVec = []
	while eta < 1:
		alpha, kern_prod = multiClassSDG_with_Kernel(train_data, train_labels, C, eta, T, K)
		temp = accuracy_with_kernel(validation_data, validation_labels, alpha, kern_prod, K)
		print "eta:", eta,"Accuracy:", temp
		etaVec.append(eta)
		accuracyVec.append(temp)
		eta *= 10
		#if(eta < 10e-9):
		#		eta *= 5
		#elif(eta < 10e-5):
		#	eta *= 1.5
		#else:
		#	eta *= 5

	res = zip(etaVec, accuracyVec)
	fig = plt.figure()
	plt.plot(etaVec, accuracyVec, c = "blue")
	plt.xlabel('eta Values')
	plt.xscale("log")
	plt.ylabel('Accuracy')         
	plt.title('Accuracy with different eta values') 
	res.sort(key=operator.itemgetter(1))
	plt.axvline(x=res[len(res)-1][0], c="red", label = "Best eta=%.10f" %res[len(res)-1][0])
	#plt.axhline(y=res[len(res)-1][1], c="red", label = "Best C=%f" %res[len(res)-1][1])
	plt.legend(loc="lower right")
	plt.savefig('plot6aEta_with_Kernel')
	print "Best eta:", res[len(res)-1][0], "Mean:", res[len(res)-1][1]
	return res[len(res)-1][0]

def find_best_C_with_Kernel(train, labels, eta, T, K):
	print "find best C value"
	C = 10e-15
	cVec = []
	accuracyVec = []
	while C < 10e10:
		alpha, kern_prod = multiClassSDG_with_Kernel(train_data, train_labels, C, eta, T, K)
		temp = accuracy_with_kernel(validation_data, validation_labels, alpha, kern_prod, K)
		print "C:", C,"Accuracy:", temp
		cVec.append(C)
		accuracyVec.append(temp)
		C *= 10
		#if(C < 10e-1):
		#	C *= 5
		#elif(C < 2):
		#	C *= 1.5
		#else:
		#	C *= 5

	res = zip(cVec, accuracyVec)
	fig = plt.figure()
	plt.plot(cVec, accuracyVec, c = "blue")
	plt.xlabel('C Values')
	plt.xscale("log")
	plt.ylabel('Accuracy')         
	plt.title('Accuracy with different C values') 
	res.sort(key=operator.itemgetter(1))
	plt.axvline(x=res[len(res)-1][0], c="red", label = "Best C=%.10f" %res[len(res)-1][0])
	#plt.axhline(y=res[len(res)-1][1], c="red", label = "Best C=%f" %res[len(res)-1][1])
	plt.legend(loc="lower right")
	plt.savefig('plot6aC_with_Kernel')
	print "Best C:", res[len(res)-1][0], "Mean:", res[len(res)-1][1]
	return res[len(res)-1][0]

def accuracy_with_kernel(data, labels, alpha, kern_prod, K):
	accuracy = 0
	for i in range(0,len(data)):
		k = [numpy.square(numpy.dot(kern_prod[j], data[i])) for j in K]
		temp = numpy.argmax(numpy.array([numpy.dot(alpha[j], k[j]) for j in K]))
		#print "A:", labels[i], "B:", temp	
		accuracy += 1 if(labels[i] == temp) else 0
	return float(accuracy)/len(data)

def multiClassSDG_with_Kernel(train, labels, C, eta, T, K):
	## init weights
	alpha = [numpy.zeros((0))]*10
	kern_prod = [numpy.zeros((0, train.shape[1]))]*10
	for t in range(T):
		i = numpy.random.randint(0, len(train))
		xi = train[i]
		yi = int(labels[i])
		onesVector = [1 if(j != labels[i]) else 0 for j in K]
		k = [numpy.square(numpy.dot(kern_prod[j], xi)) for j in K] # k(x dot x)
		loss_vec = [numpy.dot(k[j], alpha[j]) - numpy.dot(k[yi], alpha[yi]) + onesVector[j] for j in K]
		j_max_func = numpy.argmax(loss_vec)
		alpha = [alpha[j] * (1.0 - eta) for j in K]
		if j_max_func != yi:
			alpha[j_max_func] = numpy.append(alpha[j_max_func], [-1.0 * eta * C], axis=0)
			alpha[yi] = numpy.append(alpha[yi], [1.0 * eta * C], axis=0)
			kern_prod[j_max_func] = numpy.append(kern_prod[j_max_func], [xi], axis=0)
			kern_prod[yi] = numpy.append(kern_prod[yi], [xi], axis=0)
	#print kern_prod
	return alpha, kern_prod

	'''
	for t in range(1, T+1):
		i = int(numpy.random.uniform(0,len(train)))
		
		#find MAX loss function
		onesVector = [1 if(j != labels[i]) else 0 for j in K]
		lossFunction  = [numpy.dot(w[j],train[i]) - numpy.dot(w[int(labels[i])],train[i]) + onesVector[j] for j in K]
		j_max_func = numpy.argmax(lossFunction)

		for j in K:
			w[j] = (1-eta)*w[j]

		w[j_max_func] -= eta*C*train[i] if(j_max_func != labels[i]) else 0
		w[int(labels[i])] += eta*C*train[i] if(j_max_func != labels[i]) else 0

		#print w

	return w
	'''


#-----------------------------------------------------------------------------------#
def find_best_eta(train, labels, C, T, K):
	print "find best eta value"
	eta = 10e-15
	etaVec = []
	accuracyVec = []
	while eta < 1:
		w = multiClassSDG(train_data, train_labels, C, eta, T, K)
		temp = accuracy(validation_data, validation_labels, w, K)
		print "eta:", eta,"Accuracy:", temp
		etaVec.append(eta)
		accuracyVec.append(temp)
		if(eta < 10e-9):
				eta *= 5
		elif(eta < 10e-5):
			eta *= 1.5
		else:
			eta *= 5

	res = zip(etaVec, accuracyVec)
	fig = plt.figure()
	plt.plot(etaVec, accuracyVec, c = "blue")
	plt.xlabel('eta Values')
	plt.xscale("log")
	plt.ylabel('Accuracy')         
	plt.title('Accuracy with different eta values') 
	res.sort(key=operator.itemgetter(1))
	plt.axvline(x=res[len(res)-1][0], c="red", label = "Best eta=%.10f" %res[len(res)-1][0])
	#plt.axhline(y=res[len(res)-1][1], c="red", label = "Best C=%f" %res[len(res)-1][1])
	plt.legend(loc="lower right")
	plt.savefig('plot6aEta')
	print "Best eta:", res[len(res)-1][0], "Mean:", res[len(res)-1][1]
	return res[len(res)-1][0]

def find_best_C(train, labels, eta, T, K):
	print "find best C value"
	C = 10e-15
	cVec = []
	accuracyVec = []
	while C < 10e10:
		w = multiClassSDG(train_data, train_labels, C, eta, T, K)
		temp = accuracy(validation_data, validation_labels, w, K)
		print "C:", C,"Accuracy:", temp
		cVec.append(C)
		accuracyVec.append(temp)
		#C *= 10
		if(C < 10e-1):
			C *= 5
		elif(C < 2):
			C *= 1.5
		else:
			C *= 5

	res = zip(cVec, accuracyVec)
	fig = plt.figure()
	plt.plot(cVec, accuracyVec, c = "blue")
	plt.xlabel('C Values')
	plt.xscale("log")
	plt.ylabel('Accuracy')         
	plt.title('Accuracy with different C values') 
	res.sort(key=operator.itemgetter(1))
	plt.axvline(x=res[len(res)-1][0], c="red", label = "Best C=%.10f" %res[len(res)-1][0])
	#plt.axhline(y=res[len(res)-1][1], c="red", label = "Best C=%f" %res[len(res)-1][1])
	plt.legend(loc="lower right")
	plt.savefig('plot6aC')
	print "Best C:", res[len(res)-1][0], "Mean:", res[len(res)-1][1]
	return res[len(res)-1][0]

def accuracy(data, labels, w, K):
	accuracy = 0
	for i in range(0,len(data)):
		temp = numpy.argmax(numpy.array([numpy.dot(data[i], w[j]) for j in K]))
		#print "A:", labels[i], "B:", temp	
		accuracy += 1 if(labels[i] == temp) else 0
	return float(accuracy)/len(data)
'''
def predict(data, w):
	prediction = 0
	for i in range(0,len(data)):
		prediction += w[i]*data[i]
	return 1 if prediction >= 0 else -1
	return
'''
def multiClassSDG(train, labels, C, eta, T, K):
	## init weights
	w = [numpy.zeros(len(train[0]), dtype='float64')] * 10 # 10 weights vectors each length of the data, 10 is the num of digits
	
	for t in range(1, T+1):
		i = int(numpy.random.uniform(0,len(train)))
		
		#find MAX loss function
		onesVector = [1 if(j != labels[i]) else 0 for j in K]
		lossFunction  = [numpy.dot(w[j],train[i]) - numpy.dot(w[int(labels[i])],train[i]) + onesVector[j] for j in K]
		j_max_func = numpy.argmax(lossFunction)

		for j in K:
			w[j] = (1-eta)*w[j]

		w[j_max_func] -= eta*C*train[i] if(j_max_func != labels[i]) else 0
		w[int(labels[i])] += eta*C*train[i] if(j_max_func != labels[i]) else 0

		#print w

	return w

def print_image(name, array, mult):
	w, h = 28, 28
	data1 = numpy.zeros((h, w, 3), dtype=numpy.uint8)
	for i in range(0,h):
		for j in range(0,w):
			temp = array[(i*h+j)]*mult
			if (temp > 255):
				#print temp
				temp = 255
			if (temp < -255):
				#print temp
				temp = -255
			if temp < 0:
				data1[i,j] = [0,0,-temp]
			elif temp > 0:
				data1[i,j] = [0,temp,0]
			else:
				data1[i,j] = [255,0,0]
	img = Image.fromarray(data1, 'RGB')
	img.save(name)
	#img.show()
	return

if __name__ == '__main__':
    main()
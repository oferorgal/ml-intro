'''
Intro to ML - ex2-q2
Ofer Orgal 300459898
oferorgal@mail.tau.ac.il
'''
from sklearn.datasets import fetch_mldata
import numpy.random
from PIL import Image
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

def main():
	mnist = fetch_mldata('MNIST original')
	data = mnist['data']
	labels = mnist['target']

	idx = numpy.random.RandomState(0).choice(70000, 11000,replace = False)
	train = data[idx[:10000], :].astype(int)
	train_labels = labels[idx[:10000]]
	test = data[idx[10000:], :].astype(int)
	test_labels = labels[idx[10000:]]


	if(len(sys.argv) < 2):
		print "Input format:\nex2q1.py a #img #k // to predict image number #img in the set with #k neighbors\nex2q1.py c // to find the best k value\nex2q1.py d #k // to find the best n value, #k is the k value you want to use"
		return
	
	if sys.argv[1] == "a":
		if(len(sys.argv) < 4):
			print "Input ex2q1.py a #img #k // to predict image number #img in the set with #k neighbors"
			return	
		predict = predictImg(train, train_labels, test[int(sys.argv[2])], int(sys.argv[3]))
		print "prediction: ", predict
		print "Label:      ", test_labels[int(sys.argv[2])]
	elif sys.argv[1] == "c":
		bestK = findBestK(train, train_labels, test, test_labels)
		print "best K: ", bestK
	elif sys.argv[1] == "d":
		if(len(sys.argv) < 3):
			print "Input ex2q1.py d #k // to find the best n value, #k is the k value you want to use"
			return	
		bestN = findBestN(train, train_labels, test, test_labels, int(sys.argv[2]))
		print "best N: ", bestN

	return
'''
1d
finds the best value of n (sample size) for a given k value 
'''
def findBestN(trainingSet, trainingSet_labels, testSet, testSet_labels, k):
	MinN = 100
	MaxN = 5000
	temp = [0.0]*((MaxN/100))
	plotIndex = [0]*((MaxN)/100)
	n = 100
	fig = plt.figure()
	while n <= MaxN:
		for i in range(0,len(testSet)):
			if predictImg(trainingSet[:n], trainingSet_labels, testSet[i], k) == testSet_labels[i]:
				temp[((n/100)-1)] += 1
		temp[((n/100)-1)] /= 10
		plt.scatter(n, temp[((n/100)-1)], c = "blue", marker = 'o')
		plotIndex[((n/100)-1)] = n
		print "n = ", n, "   ", temp[((n/100)-1)] , "%"
		n += 100
	bestN = temp.index(max(temp))*100
	print "MAX:" , bestN
	plt.plot(plotIndex,temp, c = "blue")
	plt.xlabel('n value')
	plt.ylabel('Accuracy %')         
	plt.title('Algorithm accuracy with sample size from 100 to 5000 ') 
	plt.savefig('plotQ1d') 
	print "plotQ1d.png created!"
	return bestN

'''
1c
find the best value of k for a trining set of 1000 samples
'''
def findBestK(trainingSet, trainingSet_labels, testSet, testSet_labels):
	kMax = 100 #100
	temp = [0.0]*kMax
	plotIndex = [0]*kMax
	fig = plt.figure()
	for k in range(0,kMax):
		for i in range(0,len(testSet)):			
			prediction = predictImg(trainingSet[:1000], trainingSet_labels, testSet[i], k+1)
			#print prediction, " ", testSet_labels[i]
			if prediction == testSet_labels[i]:
				temp[k] += 1
		temp[k] /= 10
		plt.scatter(k+1, temp[k], c = "blue", marker = 'o')
		plotIndex[k] = k + 1
		print "k = ", k + 1, "   ", temp[k] , "%"
	bestK = temp.index(max(temp))
	print "MAX:" , bestK+1
	plt.plot(plotIndex,temp, c = "blue")
	plt.xlabel('K value')
	plt.ylabel('Accuracy %')         
	plt.title('Algorithm accuracy with k from 1 to 100') 
	plt.savefig('plotQ1c') 
	print "plotQ1c.png created!"
	return bestK+1

'''
1a
Function predictImg 
	Inputs: training data and lables, the image we want to query 
			and the number of NN
	Output: Prediction on that image, a number from 0 to 9
'''
def predictImg(trainingSet, trainingSet_labels, query, k):
	# First we will calc the distances from the queryImg and the images from the training set
	distVector = []
	votes = [0]*10
	for i in range(0, len(trainingSet)):
		distVector.append((trainingSet_labels[i], numpy.linalg.norm(query - trainingSet[i])))
	distVector.sort(key=operator.itemgetter(1))
	for i in range(0, k):
		votes[int(distVector[i][0])] += 1;
	#print(votes)
	#votes.sort()
	prediction = votes.index(max(votes))
	#print(distVector[:k])
	
	return prediction


def print_image(array):
	w, h = 28, 28
	data1 = numpy.zeros((h, w, 3), dtype=numpy.uint8)
	for i in range(0,h):
		for j in range(0,w):
			data1[i,j] = [array[(i*h+j)],0,0]
	#data1[256, 256] = [255, 0, 0]
	img = Image.fromarray(data1, 'RGB')
	img.save('my.png')
	img.show()
	return


if __name__ == '__main__':
    main()



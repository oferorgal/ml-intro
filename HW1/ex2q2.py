'''
Intro to ML - ex2-q2
Ofer Orgal 300459898
oferorgal@mail.tau.ac.il
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import *
from sklearn.datasets import fetch_mldata
import numpy.random
from PIL import Image
import operator

import sys
import decimal
import random


def main():

    if(len(sys.argv) < 2):
        print "Input format:\n \
            ex2q2.py a // plot according to the distribution with intervals\n \
            ex2q1.py b // to show that the hypothesis that comply with the distribution is with the smallest error \n \
            ex2q1.py c // calc the ERM and True errors for m=10,20,..100 samples size across 100 runs\n \
            ex2q1.py d // find the best k value\n \
            ex2q1.py e // run d 100 times\n \
            ex2q1.py f // preform cross validation for the hypothesis\n"
        return
    
    if sys.argv[1] == "a":
        m = 100
        k = 2
        x, y = custom_distribution_pair(m)
        intervals, besterror = find_best_interval(x, y, k)
        plot_x_y_intervals(k,x,y,intervals)

    elif sys.argv[1] == "b":
        m = 100
        x, y = custom_distribution_pair(m)
        best_hypothesis_intervals(x, y, m)

    elif sys.argv[1] == "c":
        run_hypothesis_error()

    elif sys.argv[1] == "d":   
        findBestK(50, 1)

    elif sys.argv[1] == "e":   
        run100Times_findBestK(50)

    elif sys.argv[1] == "f":
        m = 50
        x, y = custom_distribution_pair(m)
        cv_x, cv_y = custom_distribution_pair(m)
        cv(x, y, cv_x, cv_y)
    return


'''
2a
Gen random number from 0 to 1 and according to x is in the given range it will return 1 or 0 as the 
value of y
'''
def custom_dist(x):
    if (x >= 0.25 and x <= 0.5) or (x >= 0.75):
        return 1 if random.random() < 0.1 else 0
    else:
        return 1 if random.random() < 0.8 else 0
'''
2a
Gen x and y values according to the distribution 
'''
def custom_distribution_pair(m):
    # generate m numbers with uniform dist.
    x_list = [random.uniform(0, 1) for i in xrange(m)] 
    x_list.sort() #sort x for later...
    #   generate m numbers with the given distribution.
    y_list = [custom_dist(x_list[x]) for x in xrange(m)]
    return x_list, y_list
'''
2a
input: k value, a list of x's and y's and intervals
output: a plot of the points and intervals.
'''
def plot_x_y_intervals(k,x,y,intervals):
    fig = plt.figure()
    plt.scatter(x, y, c = y, s = 20)
    plt.ylim(-0.1,1.1)
    for i in xrange(k-1):
        plt.plot(intervals[i],[0.5,0.5], c = "green",linewidth=8)
    plt.plot(intervals[k-1],[0.5,0.5], c = "green",linewidth=8, label = "intervals") # did this for the label
    plt.axvline(x=0.25, c="blue", label = "x=0.25")
    plt.axvline(x=0.5, c="red", label = "x=0.5")
    plt.axvline(x=0.75, c="purple", label = "x=0.75")
    plt.legend(loc="center right")
    plt.title('Data plot according to the distribution with intervals')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('plotQ2a')   
    print "plotQ2a.png created!"       
    #plt.show()

'''
2b
The best_hypothesis with best_hypothesis_intervals functions show that the best hypothesis for the
given distribution is:

h(x) =  1 if 0<x<0.25 or 0.5<x<0.75
        0 else

We show by moving the intervals that the min error is when the intervals fit the distribution.
'''
def best_hypothesis(x,hypothesis_intervals):
    if (x >= hypothesis_intervals[0][0] and x<= hypothesis_intervals[0][1]) or (x >= hypothesis_intervals[1][0] and x<= hypothesis_intervals[1][1]):
        return 1
    return 0 

def best_hypothesis_intervals(x, y, m):
    hypothesis_intervals = [(0.0,0.0),(0.0,0.0)] #(0,0.25),(0.5,0.75)
    j = 0
    while j <= 0.25:
        error = 0
        for i in xrange(m):
            hypothesis_intervals = [(j,j+0.25),(j+0.5,j+0.75)]
            if best_hypothesis(x[i],hypothesis_intervals) != y[i]:
                error += 1
        #Print easy to read intervals.
        print "Intervals: [(%.02f," %hypothesis_intervals[0][0],"%.02f)," %hypothesis_intervals[0][1] \
        ,"(%.02f," %hypothesis_intervals[1][0],"%.02f)]," %hypothesis_intervals[1][1] \
        , "error on Intervals: ", float(error)/m*100, "%"
        j += 0.01
    return

def hypothesis(x, intervals):
    for i in range(0,len(intervals)):
        if x >= intervals[i][0] and x<= intervals[i][1]:
            return 1
    return 0 

'''
2c
returns the ERM and True error for a list of x's ans y's for a given intervals.
'''
def hypothesis_error(x, y, intervals):
    error = 0
    m = len(x)
    for i in range(0, m):
        if hypothesis(x[i], intervals) != y[i]:
            error += 1
    ERM = float(error)/m*100
    error = 0
    hypothesis_intervals = [(0,0.25),(0.5,0.75)]
    for i in range(0, m):
        if hypothesis(x[i], hypothesis_intervals) != y[i]:
            error += 1
    TRU = float(error)/m*100

    return ERM, TRU
'''
2c
runs hypothesis_error 100 times.
'''
def run_hypothesis_error():
    T = 100
    #i = 10
    start = 10
    end = 105
    step = 5
    plotDataERM = [0]*((end-start)/step)
    plotDataTRU = [0]*((end-start)/step)
    plotDataInt = [0]*((end-start)/step)
    fig = plt.figure()
    for i in range(start,end,step):
        sum = [0]*3
        for t in range(0,T):
            sample_x, sample_y = custom_distribution_pair(i)
            intervals, besterror = find_best_interval(sample_x, sample_y, 2)
            ERM, TRU = hypothesis_error(sample_x, sample_y, intervals)
            sum[0] += float(TRU)/100
            sum[1] += float(ERM)/100
            sum[2] += float(besterror)/i
        print "Size: ", i
        print "besterror:      ", float(sum[2])/T*100, "%"
        print "Avg ERM Error:  ", float(sum[1])/T*100, "%"
        print "Avg TRUE Error: ", float(sum[0])/T*100, "%"
        print "...................."
        plotDataERM[((i-start)/5)] = float(sum[1])/T*100
        plotDataTRU[((i-start)/5)] = float(sum[0])/T*100
        plotDataInt[((i-start)/5)] = i

        plt.scatter(i, float(sum[1])/T*100, c = "red", marker = 'o')
        plt.scatter(i, float(sum[0])/T*100, c = "blue", marker = 'v')

    plt.plot(plotDataInt, plotDataERM, c = "red", label = "ERM error")
    plt.plot(plotDataInt, plotDataTRU, c = "blue", label = "True error")
    plt.legend(loc="lower right")
    plt.xlabel('Sample size m')
    plt.ylabel('Error %')         
    plt.title('Error avg 100 times')
    plt.savefig('plotQ2c')
    print "plotQ2c.png created!" 
    return
'''
2d
find the best k value for a given list of x's and y's
'''
def findBestK(m, plot):
    maxK = 20
    ERM = [m]*maxK
    TRU = [m]*maxK
    plotIndex = [0]*maxK
    x, y = custom_distribution_pair(m)
    for i in range(0,maxK):
        intervals, besterror = find_best_interval(x, y, i+1)
        plotIndex[i] = i + 1
        ERM[i], TRU[i] = hypothesis_error(x, y, intervals)
        if(plot == 1):
            print "k: ", i, " ERM error: ", ERM[i], " True error:", TRU[i]

    if(plot == 1):
        fig = plt.figure()
        plt.plot(plotIndex, ERM, c = "red", label = "ERM error")
        plt.plot(plotIndex, TRU, c = "blue", label = "True error")
        plt.legend(loc="lower right")
        plt.xlabel('K value')
        plt.ylabel('Error %')         
        plt.title('Error across K values')
        plt.savefig('plotQ2d') 
        print "plotQ2d.png created!" 
    return ERM.index(min(ERM)), ERM, TRU
'''
2e
runs findBestK 100 times
'''
def run100Times_findBestK(m):
    T=100
    k = 20
    ERM = [0]*k
    TRU = [0]*k
    bestK = [0]*k
    plotIndex = [0]*k
    sumERM = [0]*k
    sumTRU = [0]*k
    fig = plt.figure()
    for i in range(0,T):
        print i
        bestK, ERM, TRU = findBestK(50, 0)
#        plotIndex[i] = i+1
        for j in range(0,20):
            sumERM[j] += ERM[j]
            sumTRU[j] += TRU[j]   
    for i in range(0,20):
         sumERM[i] = float(sumERM[i])/T
         sumTRU[i] = float(sumTRU[i])/T
    for i in range(0,20):
        plt.scatter(i+1, sumERM[i], c = "red", marker = 'o')
        plt.scatter(i+1, sumTRU[i], c = "blue", marker = 'v')
        plotIndex[i] = i+1
    #plt.scatter(i+1, bestK[i], c = "green", marker = '>')
    plt.plot(plotIndex, sumERM, c = "red", label = "ERM error")
    plt.plot(plotIndex, sumTRU, c = "blue", label = "True error")
    #plt.plot(plotIndex, bestK, c = "green", label = "K value")
    plt.legend(loc="upper right")
    plt.xlabel('K value')
    plt.ylabel('Error %')         
    plt.title('Error across K values avg 100 times for each k')
    plt.savefig('plotQ2e') 
    print "plotQ2e.png created!"
'''
2f
preform cv for all hypothesis with k=1,...20 intervls with 50 points of x's and y's according
to the distribution.
'''
def cv(hy_x, hy_y, cv_x, cv_y):
    ERM = [0]*20
    TRU = [0]*20
    plotIndex = [0]*20
    for k in range(0,20):
        intervals, besterror = find_best_interval(hy_x, hy_y, k+1)
        ERM[k], TRU[k] = hypothesis_error(cv_x, cv_y, intervals)
        print "ERM error: ", ERM[k], " True error: ", TRU[k]
        plotIndex[k] = k+1
        plt.scatter(k+1, ERM[k], c = "red", marker = 'o')
        plt.scatter(k+1, TRU[k], c = "blue", marker = 'v')
    plt.plot(plotIndex, ERM, c = "red", label = "ERM error")
    plt.plot(plotIndex, TRU, c = "blue", label = "True error")
    plt.legend(loc="lower right")
    plt.xlabel('K value')
    plt.ylabel('Error %')         
    plt.title('Cross Validation to find best K value')
    plt.savefig('plotQ2f') 
    print "plotQ2f.png created!"
    print "Best k value with CV: ", ERM.index(min(ERM))+1

    return ERM.index(min(ERM))+1


def find_best_interval(xs, ys, k):
    assert all(array(xs) == array(sorted(xs))), "xs must be sorted!"

    xs = array(xs)
    ys = array(ys)
    m = len(xs)
    P = [[None for j in range(k+1)] for i in range(m+1)]
    E = zeros((m+1, k+1), dtype=int)
    
    # Calculate the cumulative sum of ys, to be used later
    cy = concatenate([[0], cumsum(ys)])
    
    # Initialize boundaries:
    # The error of no intervals, for the first i points
    E[:m+1,0] = cy
    
    # The minimal error of j intervals on 0 points - always 0. No update needed.        
        
    # Fill middle
    for i in range(1, m+1):
        for j in range(1, k+1):
            # The minimal error of j intervals on the first i points:
            
            # Exhaust all the options for the last interval. Each interval boundary is marked as either
            # 0 (Before first point), 1 (after first point, before second), ..., m (after last point)
            options = []
            for l in range(0,i+1):  
                next_errors = E[l,j-1] + (cy[i]-cy[l]) + concatenate([[0], cumsum((-1)**(ys[arange(l, i)] == 1))])
                min_error = argmin(next_errors)
                options.append((next_errors[min_error], (l, arange(l,i+1)[min_error])))

            E[i,j], P[i][j] = min(options)
    
    # Extract best interval set and its error count
    best = []
    cur = P[m][k]
    for i in range(k,0,-1):
        best.append(cur)
        cur = P[cur[0]][i-1]       
        if cur == None:
            break 
    best = sorted(best)
    besterror = E[m,k]
    
    # Convert interval boundaries to numbers in [0,1]
    exs = concatenate([[0], xs, [1]])
    representatives = (exs[1:]+exs[:-1]) / 2.0
    intervals = [(representatives[l], representatives[u]) for l,u in best]
    
    return intervals, besterror

if __name__ == '__main__':
    main()
# Author: Zachary Boster
# Implementation of K-Means Clustering

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns
import random

#the euclidean distance measure method for classifying clusters
def euclideanDistance(pointA, pointsB): 
    #np.sqrt and np.sum used to calculate square root and sum for array-like parameters
    return np.sqrt(np.sum((pointA - pointsB)**2, axis = 1))

#core class
class kMeansClustering:  
    #initialize the number of clusters and how many iterations.
    def __init__(self, clusters, iterations):
        self.clusters = clusters 
        self.iterations = iterations
    
    #core function for training the centroid locations
    def training(self, dataPoints):  
        #choose a random location for the first initial centroid
        self.centroids = [random.choice(dataPoints)] 

        #for loop to choose the initial centroid locations
        for i in range(self.clusters -1):
            #use the euclidean formula to calculate the distance from the centroid to the points
            distance = np.sum([euclideanDistance(g, dataPoints) for g in self.centroids], axis=0) 
            #normalize the distance
            distance /= np.sum(distance) 
            #choose centroid location according to datpoint distances
            nextCentroid, = np.random.choice(range(len(dataPoints)), size = 1, p = distance) 
            #add the next centroid
            self.centroids += [dataPoints[nextCentroid]] 

        #plot the first chosen centroids
        #counter to check if we reach iteration limit
        count = 0
        #store past centroid to check if its location has changed to determine if we should stop
        pastCentroids = None

        #loop for assigning datasets and closest centroids, will stop when centroids do not change after a loop or if it reaches the set iteration limit
        while np.not_equal(self.centroids, pastCentroids).any() and count < self.iterations: 
            #sorted data is assigned by assigning data to its nearest centroid
            sortedData = [[] for i in range(self.clusters)]

            #for loop calculates each data points minimum distance centroid
            for i in dataPoints:
                #distance calculated using euclidean method
                distance = euclideanDistance(i, self.centroids)
                #centroid with the lowest distance from the datapoint is appended to the sorted data
                centroidMin = np.argmin(distance)
                sortedData[centroidMin].append(i)

            #set present centroids to past before updating 
            pastCentroids = self.centroids
            #update the location of the centroids with average of the datapoints that belong to them
            self.centroids = [np.mean(i, axis=0) for i in sortedData] 
            count += 1
            #plot the updated centroids

#Ask the user for their desired cluster and datapoint amounts
clusters = int(input("Please enter the number of clusters: "))
data_points = int(input("Please enter the number of data points: "))
#iterations are added to ensure it does not run infinitely, this value may be changed as needed
iterations = 100

#create the initial (X,Y) data. The n_samples parameter determines the number of data points. 
#Centers determines number of centers to generate, or the fixed center locations, here it is primarily used to classify colors for the data visualization.
# random_state determines the random number generation for each set.
dataPoints, groups = make_blobs(n_samples=data_points, centers=clusters, random_state=100)

#initialize values and then call kmeans training 
kmeans = kMeansClustering(clusters, iterations)
kmeans.training(dataPoints)

#create scatterplot with data points, iterate though the X in the (X,Y) datapoint values for values and through Y values for y. 
# The data label groups from make blobs assosciated with each datapoint is used to classify each hue.
sns.scatterplot(x=[i[0] for i in dataPoints], y=[i[1] for i in dataPoints], hue=groups, palette="bright", legend=False)
#plot the centroids for the clusters, iterate through (X,Y) same as for data points
#'K+' creates black plus symbols. markersize determines the size of those symbols.
plt.plot([x for x, i in kmeans.centroids], [y for i, y in kmeans.centroids], 'k+', markersize=10,)
plt.show()

import os
import csv
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# set file locations relative to project file
dataPath = os.path.join(os.path.dirname(__file__),"dataset")
logPath = os.path.join(os.path.dirname(__file__),"testLogs")


# get list of valid files
# file can be created (if necessary) using makeFileList.py
# making this list just in case some frames are missing or not in order
with open(dataPath + r'\fileNames.csv', newline='') as csvfile:
    frameIDS = list(csv.reader(csvfile))
    # accessed like data[x][0] to return a string containing the identifier
    # where x is <= numFiles


# utility function to check num available frames
def getNumFrames():
    return len(frameIDS)

def preProcessCloud(pointCloud, DISTANCE_THRESHOLD=20, GROUND_LEVEL=-1.2):
    newCloud = []
    for point in pointCloud:
        if point[2] > GROUND_LEVEL: # filter for height
            # calculate distance from (0, 0, 0)
            dist_squared = DISTANCE_THRESHOLD * DISTANCE_THRESHOLD
            if (point[0]**2 + point[1]**2 + point[2]**2) <= dist_squared:
                newCloud.append(point)

    return np.array(newCloud)

def plotCloud(pointCloud, labels):

    x = pointCloud[:,0]
    y = pointCloud[:,1]
    z = pointCloud[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=labels, marker='.', s=1)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


# function to return an array containing velodyne data
# takes in resolution (%, 0->1) and will strip down data as necessary
# can take a while to run when resolution decreases (~15 seconds)
def getVelodyne(index, resolution=1):

    # open correct file based on index
    # get frame ID (filename) from data list
    fileName = frameIDS[index][0]
    points = np.fromfile(os.path.join(dataPath, "velodyne", fileName+".bin"),dtype=np.float32).reshape(-1, 4)
    points = points[:, :3] # exclude luminance

    # remove files to decrease resoution
    # randomly select a percentage of points and remove them from array
    numPoints = len(points)
    numPointsRemove = math.floor(numPoints * (1-resolution))

    for i in range(0,numPointsRemove):
        # randomly select a point to be removed by index
        # get most up to date length of points as it changes
        # make sure point not already selected
        indexSelected = random.randrange(0,len(points))
        points = np.delete(points,indexSelected,0)

    # return updated resolution
    return points


# function to get bounding boxes
# returns as list of each bounding box where each box is formatted
#   [type, [xDim, yDim, zDim],[xLoc, yLoc, zLoc]]
def getBoundingBox(index):

    returnArr = []

    # get filename from file
    fileName = frameIDS[index][0]

    # open file
    # file type is txt, but formatted like csv where delimiter is space
    with open(os.path.join(dataPath,"label_2",fileName + ".txt"), newline='') as csvfile:
        objects = list(csv.reader(csvfile, delimiter= " "))


    # loop through each line
    # each line contains a seperate bounding box
    for object in objects:
        returnArr.append([object[0], [object[8],object[9],object[10]],[object[11],object[12],object[13]]])

    return returnArr


# utility  function to log results to csv file for analysis
def logResults(algorithm, frameID, resolution, f1, time, numPoints=0):

    # open file based on algorithm name
    with open(os.path.join(logPath, algorithm + ".csv".format(algorithm)), 'a+',newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        # log all the information
        csvwriter.writerow([frameID, resolution, f1, time, numPoints])

# sample usage:
# print(getBoundingBox(2))
# print(getVelodyne(2,1))

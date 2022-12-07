import getData
import myAlgorithms as algorithms
import numpy as np
import sklearn.cluster
import sklearn.preprocessing
import time

# Driver code which tests object detection algorithms on pointclouds of different resolutions
# Logs results into csv files for analysis

# ----------------------------------------------------------------------------
#     CONFIG
# ----------------------------------------------------------------------------
# list of resolutions to test each algorithm for
# given as decimals where 100% = 1, 90% = .9, etc.
RESOLUTIONS = [1,.8,.6,.4,.2,.1]

# number of times to run algorithms
# uses different randomized point cloud reduction each time
ITERATIONS = 2

# list of frames to test
# 0 - 99   valid for "smallDataset"
# 0 - 2480 valid for "fullDataset"
FRAMES = [item for item in range(0,50)] # frames 0 - 49 (50 total)

# ----------------------------------------------------------------------------
#     MAIN CODE
# ----------------------------------------------------------------------------

# run algorithms for each resolution given in config
for resolution in RESOLUTIONS:
    # for each requested frame in config
    for frameID in FRAMES:

        print(" ------------------------------------------------- ")
        print("BEGIN FRAME ID {} AT RESOLUTION {}".format(frameID,resolution))
        print(" ------------------------------------------------- \n")

        # do this ITERATIONS number of times
        for iter in range(ITERATIONS):

            # get the frame from the KITTI dataset at appropriate resolution
            frame = getData.getVelodyne(frameID,resolution)

            # get the correct bouding boxes from the KITTI dataset
            knownBoxes = getData.getBoundingBox(frameID)

            # preprecess the data
            #   removes ground points
            #   removes points too far from center
            #   default ground_level is -1.2 meters
            #   default distance_threshold is 20 meters
            frame = getData.preProcessCloud(frame)
            numPoints = len(frame)

            print("\t[{}]: {} points in frame after pre-processing and resolution reduction.".format(iter+1, numPoints))

            # ----------------------------------------------------------------
            # perform DBScan
            # ----------------------------------------------------------------
            #   time how long it takes to run
            #   calculate accuracy by comparing resulting clusters to known boxes
            #   log results and stats into csv file

            start_time = time.time() # start timer

            clustering = sklearn.cluster.DBSCAN().fit(frame)

            total_time = time.time() - start_time # end timer

            labels = clustering.labels_ # get resulting labels
            # allow smaller clusters as resolution decreases, leaving fewer points in cloud
            labels = algorithms.filterClusterBySize(labels, 600*resolution)

            # compute bounding boxes from labels
            calculatedBBOX = []
            label_ids = np.unique(labels)

            for label_id in label_ids:
                if (label_id != -1): # ignore -1, which is the "non-label"
                    calculatedBBOX.append(algorithms.clust_to_KITTI(label_id, labels, frame))

            numClusters = len(calculatedBBOX)

            # compare calculated bounding boxes to known to compute f1 score
            tp, fp, fn = algorithms.calc_stats(calculatedBBOX,knownBoxes)
            f1_DBSCAN = (tp) / (tp + ((1/2) * (fp + fn)))

            getData.logResults("DBSCAN", frameID, resolution, f1_DBSCAN, total_time, numPoints)

            print("\t[{}]: DBSCAN ran in {:.3f}s, finding {} clusters. F1 = {:.2f} (tp={}, fp={}, fn={}) -- Results logged.".format(iter+1, total_time, numClusters,f1_DBSCAN, tp, fp, fn))

            # ----------------------------------------------------------------
            # perform Euclidean Clustering
            # ----------------------------------------------------------------
            #   time how long it takes to run
            #   calculate accuracy by comparing resulting clusters to known boxes
            #   log results and stats into csv file

            start_time = time.time() # start timer

            labels = algorithms.euclidean_clustering(frame)

            total_time = time.time() - start_time # end timer

            # allow smaller clusters as resolution decreases, leaving fewer points in cloud
            labels = algorithms.filterClusterBySize(labels, 600*resolution)

            # compute bounding boxes from labels
            calculatedBBOX = []
            label_ids = np.unique(labels)

            for label_id in label_ids:
                if (label_id != -1): # ignore -1, which is the "non-label"
                    calculatedBBOX.append(algorithms.clust_to_KITTI(label_id, labels, frame))

            numClusters = len(calculatedBBOX)

            # compare calculated bounding boxes to known to compute f1 score
            tp, fp, fn = algorithms.calc_stats(calculatedBBOX,knownBoxes)
            f1_EUCLIDEANCLUSTERING = (tp) / (tp + ((1/2) * (fp + fn)))

            getData.logResults("EUCLIDEANCLUSTERING", frameID, resolution, f1_DBSCAN, total_time, numPoints)

            print("\t[{}]: Euclidean Clustering ran in {:.3f}s, finding {} clusters. F1 = {:.2f} (tp={}, fp={}, fn={}) -- Results logged.".format(iter+1, total_time, numClusters,f1_EUCLIDEANCLUSTERING, tp, fp, fn))


            print ("\n") # new line to differentiate iterations

import numpy as np
import sklearn.cluster
import sklearn.preprocessing
import time
import getData

def euclidean_clustering(cloud, threshold=0.5):

    cluster_labels = np.zeros(len(cloud), dtype=int)
    cluster_ID = 1
    cluster = []

    for p in range(len(cloud)):
        if cluster_labels[p] != 0:
            continue

        cluster.append(cloud[p])
        cluster_labels[p] = cluster_ID

        while len(cluster) > 0:

            first_point = cluster.pop(0)

            # Get boolean mask of neighbors that corresponds to cluster_labels and cloud
            neighbors = ((cloud[:,0]-first_point[0])**2 +
                (cloud[:,1]-first_point[1])**2 +
                (cloud[:,2]-first_point[2])**2) < threshold**2

            # Go through all the actual (true) neighbors that are not already part of a cluster
            relevant_idxs = np.logical_and(neighbors, cluster_labels==0)
            relevant_points = cloud[relevant_idxs]

            for i in range(relevant_points.shape[0]):
                cluster.append(relevant_points[i])

            cluster_labels[relevant_idxs] = cluster_ID

        cluster = []
        cluster_ID += 1

    return cluster_labels

def matches2(pred_info, true_info):

    for i in range(3):
        true_info[1][i] = float(true_info[1][i])
        true_info[2][i] = float(true_info[2][i])

    # Matches if the predicted center falls anywhere within the true bounding box
    x_within = pred_info[2][0] >= true_info[2][0]-(true_info[1][0]/1.3) and pred_info[2][0] <= true_info[2][0]+(true_info[1][0]/1.3)
    y_within = pred_info[2][1] >= true_info[2][1]-(true_info[1][1]/1.3) and pred_info[2][1] <= true_info[2][1]+(true_info[1][1]/1.3)
    z_within = pred_info[2][2] >= true_info[2][2]-(true_info[1][2]/1.3) and pred_info[2][2] <= true_info[2][2]+(true_info[1][2]/1.3)

    # print("bounding location real: {}, bounding location pred: {}".format(true_info[2], pred_info[2]))
    # print("\tX: {}, Y: {}, Z: {}".format(x_within, y_within, z_within))

    return x_within and y_within and z_within


def calc_stats(pred_clusts, true_clusts):

    temp_TP, temp_FP, temp_FN = 0, 0, 0
    total_preds = len(pred_clusts)

    for true_cluster in true_clusts:
        found_match = False

        for pred_cluster in pred_clusts:
            if matches2(pred_cluster, true_cluster):
                # print("Found a match!")
                temp_TP += 1
                total_preds -= 1
                found_match = True
                break

        if not found_match:
            temp_FN += 1

    # if there are any (incorrectly) predicted clusters left
    temp_FP += total_preds
    # print("num true: {}, num pred: {}".format(len(true_clusts), len(pred_clusts)))

    return temp_TP, temp_FP, temp_FN


# remove clusters too small
def filterClusterBySize(labels, minSize=100):

    numClusters = np.amax(labels)

    for i in range(1, numClusters):
        size = np.count_nonzero(labels == i)
        if size < minSize:
            labels[labels == 1] = -1

    return labels

# compute KITTI style bounding box from cluster labels and pc
# computers one bounding box for the given cluster_id
def clust_to_KITTI(cluster_id, cluster_labels, pc):
    cluster_points = pc[cluster_labels == cluster_id]
    x_min = cluster_points[0][0]
    x_max = cluster_points[0][0]
    y_min = cluster_points[0][1]
    y_max = cluster_points[0][1]
    z_min = cluster_points[0][2]
    z_max = cluster_points[0][2]
    for point in cluster_points:
        if point[0] < x_min:
            x_min = point[0]
        elif point[0] > x_max:
            x_max = point[0]
        if point[1] < y_min:
            y_min = point[1]
        elif point[1] > y_max:
            y_max = point[1]
        if point[2] < z_min:
            z_min = point[2]
        elif point[2] > z_max:
            z_max = point[2]
    xDim = x_max - x_min
    yDim = y_max - y_min
    zDim = z_max - z_min
    xLoc = (x_max + x_min) / 2
    yLoc = (y_max + y_min) / 2
    zLoc = (z_max + z_min) / 2
    return ['DontCare', [xDim, yDim, zDim],[xLoc, yLoc, zLoc]]

# pc = getData.getVelodyne(10,.5)
# print("Points in cloud before processing: {}".format(len(pc)))
# pc = getData.preProcessCloud(pc)
# print("Points in cloud after processing: {}".format(len(pc)))
# clustering = sklearn.cluster.DBSCAN().fit(pc)
# labels = clustering.labels_ # get resulting labels
# getData.plotCloud(pc, labels)

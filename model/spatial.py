import json
import copy
import scipy
import openai
import numpy as np
import pandas as pd
import networkx as nx
import datetime
import folium
from python_tsp.heuristics import solve_tsp_simulated_annealing
from model.utils.funcs import get_max_summation_idx, get_top_k_sets, get_topk_location_pairs, find_clusters_containing_all_elements
from itertools import permutations
from pulp import LpVariable, LpProblem, LpMinimize, value, lpSum, LpBinary, PULP_CBC_CMD


class SpatialHandler:
    def __init__(self, data, min_clusters, min_pois, citywalk=False, citywalk_thresh=5000):
        self.data = data
        self.min_pois = min_pois
        self.min_clusters = min_clusters
        self.citywalk = citywalk
        self.citywalk_thresh = citywalk_thresh

    def remove_outliers(self, poi_candidates: list, selected_clusters: list):
        # Fetch the coordinates of the POI candidates
        coordinates = self.data.loc[poi_candidates, ["x", "y"]].astype('float').to_numpy()
        
        # Calculate the centroid of all coordinates
        centroid = np.mean(coordinates, axis=0)
        
        # Calculate the distances of each point to the centroid
        distances = np.linalg.norm(coordinates - centroid, axis=1)
        
        # Calculate mean and standard deviation of the distances
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Filter out outliers based on distance
        non_outliers = [poi for i, poi in enumerate(poi_candidates) if abs(distances[i] - mean_distance) <= 1.5 * std_distance]
        
        # Update clusters to remove outlier POIs
        filtered_clusters = []
        for cluster in selected_clusters:
            filtered_cluster = [poi for poi in cluster if poi in non_outliers]
            if filtered_cluster:
                filtered_clusters.append(filtered_cluster)
                
        return non_outliers, filtered_clusters

    def get_clusters(self, poi_idlist: list, thresh: int = 5000) -> list:
        """
        Identify clusters of points within a given distance threshold in a set of points.

        Args:
            poi_idlist (list): A list of unique point identifiers.
            thresh (int, optional): The distance threshold defining cluster membership. Defaults to 5000.

        Returns:
            list: A list of clusters, where each cluster is represented as a set of point identifiers.
        """
        data = self.data.loc[poi_idlist]
        coords = data[['x', 'y']].astype(float).to_numpy()

        dist_matrix = scipy.spatial.distance.cdist(coords, coords)
        np.fill_diagonal(dist_matrix, thresh + 100)
        N = len(coords)
        G = nx.Graph()
        for i in range(N):
            G.add_edge(i, i)
            for j in range(i+1, N):  # avoid duplicates and self-loops
                if dist_matrix[i, j] < thresh:
                    G.add_edge(i, j)

        all_clusters = []
        
        if G.number_of_edges() == 0:
            all_clusters = [[i for i in poi_idlist]]
            return all_clusters
        
        while G.number_of_nodes() > 0:    
            cliques = list(nx.find_cliques(G))
            index_of_longest = max(enumerate(cliques), key=lambda x: len(x[1]))[0]
            biggest_cluster_list = list(set(cliques[index_of_longest]))
            G.remove_nodes_from(biggest_cluster_list)
            all_clusters.append(set(np.array(poi_idlist)[np.array(biggest_cluster_list)].tolist()))
        
        return all_clusters
    
    def solve_tsp_with_start_end(self, dist_matrix, start_point, end_point):
        """
        Solves the Traveling Salesman Problem (TSP) with specified start and end points using the Linear Programming approach.

        The function aims to find the shortest possible route that visits every node once and returns to the original node, while starting and ending at specified points.

        Arguments:
        - self (object): The object instance on which this method is called.
        - dist_matrix (list of lists): A square matrix representing the distances between nodes. The element `dist_matrix[i][j]` represents the distance from node `i` to node `j`.
        - start_point (int): The index of the starting node for the route.
        - end_point (int): The index of the ending node for the route.

        Returns:
        - tuple: A tuple containing two elements:
            1. The total distance of the optimal route.
            2. A list of nodes representing the optimal path, starting from the `start_point` and ending at the `end_point`.
        """

        n = len(dist_matrix)
        dist = {(i, j): dist_matrix[i][j] for i in range(n) for j in range(n) if i != j}
        prob = LpProblem("TSP", LpMinimize)
        x = LpVariable.dicts('x', dist, 0, 1, LpBinary)
        prob += lpSum([x[(i, j)] * dist[(i, j)] for (i, j) in x])

        for k in range(n):
            if k == start_point:
                prob += lpSum([x[(k, i)] for i in range(n) if i != k]) == 1
                prob += lpSum([x[(i, k)] for i in range(n) if i != k]) == 0
            elif k == end_point:
                prob += lpSum([x[(k, i)] for i in range(n) if i != k]) == 0
                prob += lpSum([x[(i, k)] for i in range(n) if i != k]) == 1
            else:
                prob += lpSum([x[(k, i)] for i in range(n) if i != k]) == 1
                prob += lpSum([x[(i, k)] for i in range(n) if i != k]) == 1

        while True:
            prob.solve(PULP_CBC_CMD(msg=0))
            edges = [(i, j) for (i, j) in x if value(x[(i, j)]) > 0.5]
            G = nx.Graph()
            G.add_edges_from(edges)
            subtours = [c for c in nx.connected_components(G) if len(c) < n]
            if not subtours:
                break
            for s in subtours:
                prob += lpSum([x[(i, j)] for (i, j) in permutations(s, 2)]) <= len(s) - 1

        edges = [(i, j) for (i, j) in x if value(x[(i, j)]) > 0.5]
        path = [start_point]
        while len(edges) > 0:
            (i, j) = [edge for edge in edges if edge[0] == path[-1]][0]
            path.append(j)
            edges.remove((i, j))

        return value(prob.objective), path
    
    def get_tsp_order(self, poi_candidates_list: list = None, locs: list = None):
        """
        Find the optimal traveling salesman problem (TSP) order for a list of points.

        Args:
            poi_candidates_list (list): A list of point identifiers for candidate locations.

        Returns:
            tuple: A tuple containing the following elements:
                - np.ndarray: An array representing the optimal TSP order of point identifiers.
                - np.ndarray: An array containing the coordinates (x, y) of the candidate locations, with shape (n, 2).
                - np.ndarray: A distance matrix representing pairwise distances between candidate locations.
        """

        if locs is None:
            locs = self.data.loc[poi_candidates_list, ["x", "y"]].astype(float).to_numpy()
        dist_matrix = scipy.spatial.distance.cdist(locs, locs)
        if locs.shape[0] > 2:
            order, distance = solve_tsp_simulated_annealing(dist_matrix)
        elif locs.shape[0] == 2:
            order = [0, 1]
        else:
            order = [0]

        return np.array(order), locs, dist_matrix
    
    def get_cluster_centroids(self, clusters: list, lonlat: bool = False) -> list:
        """
        Computes the centroids for the given clusters.

        Arguments:
        - self (object): The object instance on which this method is called.
        - clusters (list): A list of clusters, where each cluster is a subset of the data's index.
        - lonlat (bool, optional): Flag indicating if the coordinates should be in "longitude, latitude" format. If set to False, "x, y" format will be used. Default is False.

        Returns:
        - list: A list of cluster centroids. Each centroid is represented as a list containing two elements, the x (or longitude) and y (or latitude) coordinates.
        """

        if lonlat:
            return [np.mean(self.data.loc[cluster][["lon", "lat"]].to_numpy(), axis=0).tolist() for cluster in clusters]
        else:
            return [np.mean(self.data.loc[cluster][["x", "y"]].astype(float).to_numpy(), axis=0).tolist() for cluster in clusters]
    
    def get_poi_pairs_across_clusters(self, clusters_order: np.ndarray, clusters: list):
        """
        Retrieves the top pairs of Points of Interest (POIs) across the given order of clusters.

        Arguments:
        - self (object): The object instance on which this method is called.
        - clusters_order (np.ndarray): An ordered array representing the sequence in which clusters should be considered.
        - clusters (list): A list of clusters, where each cluster is a subset of the data's index.

        Returns:
        - list: A list of top pairs of POIs between consecutive clusters in the given order. Each pair is represented as a list containing two elements, the indices of the POIs in their respective clusters.
        """

        all_pairs = []
        for i in range(len(clusters_order)-1):
            locsCluster1, locsCluster2 = self.data.loc[clusters[clusters_order[i]]][["x", "y"]].astype(float).to_numpy(), self.data.loc[clusters[clusters_order[i+1]]][["x", "y"]].astype(float).to_numpy()
            pair = get_topk_location_pairs(locsCluster1, locsCluster2, k=min(3, locsCluster1.shape[0], locsCluster2.shape[0]))
            pair = [clusters[clusters_order[i]][pair[0][0]], clusters[clusters_order[i+1]][pair[0][1]]]
            all_pairs.append(pair)

        return all_pairs

    def get_poi_candidates(self, allpoi_idlist: list, must_see_poi_idlist: list, req_topk_pois: np.ndarray, min_num_candidate: int, thresh : int, pseudo_must_see_pois: list = []):
        """
        Perform cluster-based point-of-interest (POI) selection to generate a list of candidate POIs.

        Args:
            ori_req_topk_pois (numpy.ndarray): Original POI ranking with shape (N, 2), where the first column contains
                                            POI IDs and the second column contains their respective scores.
            allpoi_idlist (list): List of all POI identifiers.
            must_see_poi_idlist (list): List of must-see POI identifiers.
            min_num_candidate (int): Minimum number of candidate POIs to select.

        Returns:
            tuple: A tuple containing two elements:
                - list: A list of selected candidate POIs.
                - list: A list of scores corresponding to the selected candidate POIs.        
        """

        poi_candidates, num_clusters, selected_clusters, mark_citywalk = [], 9999, [], True
        cur_ids = req_topk_pois[:, 0].astype(int)

        for poi in pseudo_must_see_pois:
            if poi not in cur_ids:
                req_topk_pois = np.insert(req_topk_pois, 0, np.array([poi, 10]), axis=0)
            else:
                row_idx = cur_ids.tolist().index(poi)
                req_topk_pois[row_idx, 1] = 10

        for poi in must_see_poi_idlist:
            if poi not in cur_ids:
                req_topk_pois = np.insert(req_topk_pois, 0, np.array([poi, 1000]), axis=0)
            else:
                row_idx = cur_ids.tolist().index(poi)
                req_topk_pois[row_idx, 1] = 1000
        
        if self.citywalk:
            clusters = self.get_clusters(allpoi_idlist, thresh=self.citywalk_thresh)
            index_candidates = get_top_k_sets(clusters, req_topk_pois, k=min(len(clusters), 2))
            index = np.random.choice(index_candidates, size=1)[0] 
            selected_cluster = []

            for poi in clusters[index]:
                if poi not in poi_candidates:
                    selected_cluster.append(poi)
                    poi_candidates.append(poi)

            for poi in must_see_poi_idlist:
                if poi not in selected_cluster or len(poi_candidates) < self.min_pois - 2:
                    mark_citywalk = False
                    break
            
            for poi in pseudo_must_see_pois:
                if poi not in selected_cluster or len(poi_candidates) < self.min_pois - 2:
                    mark_citywalk = False
                    break
            
            if len(selected_cluster) > 0 and mark_citywalk:
                selected_clusters.append(selected_cluster)

        if not mark_citywalk or not self.citywalk or len(poi_candidates) < 10:
            poi_candidates, selected_clusters = [], []
            clusters = self.get_clusters(allpoi_idlist, thresh=thresh)
            # The following code guarantees the inclusion of all user-requested POIs in the candidate set.
            merge_must_see_poi_idlist = []
            merge_must_see_poi_idlist.extend(must_see_poi_idlist)

            if pseudo_must_see_pois is not None:
                merge_must_see_poi_idlist.extend(pseudo_must_see_pois)

            merge_must_see_poi_idlist = list(set(merge_must_see_poi_idlist))
            idx = find_clusters_containing_all_elements(clusters, merge_must_see_poi_idlist)
            
            for index in idx:
                selected_cluster = []
                for poi in clusters[index]:
                    if poi not in poi_candidates:
                        selected_cluster.append(poi)
                        poi_candidates.append(poi)
                if len(selected_cluster) > 0:
                    selected_clusters.append(selected_cluster)

            if len(idx) <= self.min_clusters or len(poi_candidates) < min(min_num_candidate, self.data.shape[0]):
                while True: # this loop will end if <<< len(poi_candidates) > min_num_candidate >>> or <<< no remaining candidates in clusters >>>
                    index_candidates = get_top_k_sets(clusters, req_topk_pois, k=min(len(clusters), 2))

                    if len(index_candidates) == 0 or (len(selected_clusters) > self.min_clusters and len(poi_candidates) >= min(min_num_candidate, self.data.shape[0])):
                        break

                    index = np.random.choice(index_candidates, size=1)[0] # this introduces some randomness        
                    selected_cluster = []
                    for poi in clusters[index]:
                        if poi not in poi_candidates:
                            selected_cluster.append(poi)
                            poi_candidates.append(poi)

                    if len(selected_cluster) > 0:
                        selected_clusters.append(selected_cluster)

                    clusters.pop(index)

            poi_candidates, selected_clusters = self.remove_outliers(poi_candidates, selected_clusters)

        for must_see_poi in must_see_poi_idlist:
            if must_see_poi not in poi_candidates:
                poi_candidates.append(must_see_poi)

        poi_candidatescores = []
        for poi in poi_candidates:
            if poi in must_see_poi_idlist:
                poi_candidatescores.append(1000)
            else:
                poi_candidatescores.append(req_topk_pois[req_topk_pois[:, 0] == poi, 1].tolist()[0])
                
        return poi_candidates, poi_candidatescores, selected_clusters, mark_citywalk

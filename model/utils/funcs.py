import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from thefuzz import process


def get_user_data_embedding(city_name, must_see_poi_names, type='zh'):
    data = pd.read_csv(os.path.join("model", "data", f'{city_name}_{type}.csv'))
    embedding = np.load(os.path.join("model", "data", f'{city_name}_{type}.npy'))

    all_poi_names = data["name"].tolist()
    must_see_pois = []

    for must_see_poi_name in must_see_poi_names:
        match, score = process.extract(must_see_poi_name, all_poi_names, limit=1)[0]
        if score > 91:        
            must_see_pois.append(all_poi_names.index(match))

    data = data.reset_index(drop=True)
    
    row_idx = data.index.to_numpy()
    id = data["id"].to_numpy()
    
    r2i = {key: value for key, value in zip(row_idx, id)}
    i2r = {value: key for key, value in zip(row_idx, id)}

    return data, embedding, r2i, i2r, must_see_pois

    

class RecurringList:
    def __init__(self, lst):
        """
        Initializes the RecurringList instance.

        Args:
        - lst (List): A list of elements to be encapsulated in the RecurringList instance.

        Returns:
        None.
        """

        self.lst = lst
    
    def __getitem__(self, index):
        """
        Retrieves an element or a sequence of elements from the list in a cyclic manner using the provided index.
        If the index exceeds the length of the list, it wraps around to the beginning of the list.

        Args:
        - index (int or slice): The index or slice to retrieve items from the list.

        Returns:
        - If a single integer index is provided, returns the item at that index (type of item depends on list content).
        - If a slice is provided, returns a list of items corresponding to that slice.
        """

        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else len(self.lst)
            step = index.step if index.step is not None else 1
            
            return [self.lst[i % len(self.lst)] for i in range(start, stop, step)]
        else:
            return self.lst[index % len(self.lst)]
    
    def __len__(self):
        """
        Returns the number of elements in the encapsulated list.

        Returns:
        - int: The length of the list.
        """

        return len(self.lst)


def compute_consecutive_distances(points: np.ndarray, indices: list) -> np.ndarray:
    """
    Calculate consecutive Euclidean distances between indexed points.

    Parameters:
    - points (np.ndarray): A 2D numpy array representing points in a space.
    - indices (list): A list of indices specifying the order in which to compute distances.

    Returns:
    - np.ndarray: A numpy array containing the consecutive distances between the indexed points.
    """

    # Extract the relevant points using the indices
    relevant_points = points[indices]

    # Compute differences between consecutive points
    differences = relevant_points[1:] - relevant_points[:-1]

    # Compute and return the Euclidean distances
    return np.sqrt(np.sum(differences**2, axis=1))


def find_indices(lst: list, value: any) -> list:
    """
    Finds the first index of a given value in a list.

    This function searches for the specified value in a list and returns its first occurrence index.

    Arguments:
    - lst (list): The list in which to search for the value.
    - value (any): The value to search for within the list.

    Returns:
    - int: The index of the first occurrence of the given value in the list. 
           If the value is not found, a runtime error will occur due to list indexing with an empty list.
    """
    return [index for index, element in enumerate(lst) if element == value][0]


def sample_items(A: list, B: list, selected_clusters: list, threshold: int = 900, keep_prob: float = 0.8, keep_ids: list = None):
    """
    Samples items from lists A and B based on scores provided in B and retains selected clusters.

    This function will keep all items from A and B where the corresponding score in B exceeds a threshold.
    For the remaining items, they are sampled based on their normalized scores from B. If all remaining scores are zero, 
    a uniform sampling is done.

    Arguments:
    - A (list): A list of items that corresponds to the scores in B.
    - B (list): A list of scores where each score corresponds to an item in A.
    - selected_clusters (list of lists): A nested list of clusters, each containing points from A.
    - threshold (int, optional): A score threshold for determining must-keep items. Defaults to 900.
    - keep_prob (float, optional): The proportion of items to sample from the remaining items after applying the threshold. Defaults to 0.8.

    Returns:
    - list: A new list of items from A after sampling.
    - list: A new list of scores from B corresponding to the sampled items from A.
    - list of lists: A new nested list of clusters, each containing sampled points from A.

    Note:
    - We set the similarity score for the must-see points as 1000. Setting the threshold as 900 ensures we could keep those must-see points.
    """
    # Split items based on threshold
    keep_indices = [i for i, score in enumerate(B) if score > threshold or A[i] in keep_ids]
    remaining_indices = [i for i in range(len(B)) if i not in keep_indices]
    
    # If remaining scores are all zero, sample uniformly
    remaining_scores = np.array([B[i] for i in remaining_indices])
    if np.sum(remaining_scores) == 0:
        sample_size = int(len(remaining_indices) * keep_prob)
        sampled_indices = np.random.choice(remaining_indices, size=sample_size, replace=False)
    else:
        # Normalize the scores of the remaining items
        normalized_scores = remaining_scores / np.sum(remaining_scores)
        # Sample based on normalized scores
        sampled_indices = np.random.choice(remaining_indices, size=int(len(remaining_indices) * keep_prob), p=normalized_scores, replace=False)
    
    # Combine the indices
    final_indices = keep_indices + list(sampled_indices)
    
    # Filter A and B using final_indices
    A_new = [A[i] for i in final_indices]
    B_new = [B[i] for i in final_indices]

    idx, newselected_clusters = 0, []
    for cluster in selected_clusters:
        newSelectedCluster = []
        for point in cluster:
            if idx in final_indices:
                newSelectedCluster.append(point)
            idx += 1
        newselected_clusters.append(newSelectedCluster)
    
    return A_new, B_new, newselected_clusters


def reorder_list(A: list, B: list) -> np.ndarray:
    """
    Generates an order index list based on the elements of A as they appear in the flattened version of B.

    This function produces a list of indices representing the order in which the elements of list A 
    appear in the flattened version of list B. Any element of A not found in B will be excluded from the result.

    Arguments:
    - A (list): The reference list of items.
    - B (list of lists): A nested list, whose flattened version is used to determine the order of items from list A.

    Returns:
    - np.ndarray: An array of indices representing the order of elements from list A as they appear in the flattened version of B.
    """
    # Flatten the list B
    flattened_B = [item for sublist in B for item in sublist]

    # Create the order index list based on flattened B
    order_indices = [A.index(val) for val in flattened_B if val in A]

    return np.array(order_indices)


def remove_duplicates(input_list: list) -> list:
    unique_list = []
    for item in input_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list





###### Spatial Functions ######



def convert_to_mercator(lon, lat):
    """
    Convert a geographical point from WGS 84 (EPSG:4326) to Mercator (EPSG:3857) coordinate system.
    
    Parameters:
    - lon (float): The longitude of the point in WGS 84.
    - lat (float): The latitude of the point in WGS 84.

    Returns:
    - tuple: A tuple containing the x (longitude) and y (latitude) coordinates of the point in Mercator.
    """
    
    # Create a GeoSeries for the point using WGS 84 coordinate system (EPSG:4326)
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    
    # Convert the point to Mercator coordinate system (EPSG:3857)
    point_mercator = point.to_crs("EPSG:3857")

    # Extract x and y coordinates from the converted point
    x = point_mercator.geometry.x.iloc[0]
    y = point_mercator.geometry.y.iloc[0]
    
    return x, y



def get_max_summation_idx(A: list, B: np.ndarray) -> int:
    """
    Get the index of the list in A that has the maximum summation of corresponding values from B.

    Parameters:
    - A (list of lists): Each list inside A contains elements that correspond to the first column of B.
    - B (np.ndarray): A 2D numpy array where the first column represents elements and the second column their corresponding values.

    Returns:
    - int: Index of the list in A that has the highest summation of values from B.
    """

    max_sum = 0
    max_idx = 0

    for idx, s in enumerate(A):
        total = sum([B[B[:, 0] == item][0, 1] for item in s])
        
        if total > max_sum:
            max_sum = total
            max_idx = idx

    return max_idx


def get_top_k_sets(A: list, B: np.ndarray, k: int = 2) -> list:
    """
    Retrieve the top k lists from A based on the summation of their corresponding values in B.

    Parameters:
    - A (list of lists): Each list inside A contains elements that correspond to the first column of B.
    - B (np.ndarray): A 2D numpy array where the first column represents elements and the second column their corresponding values.
    - k (int): Number of top lists to return.

    Returns:
    - list: Indices of the top k lists from A based on the summation of values from B.
    """

    # Create a dictionary to store the sum of values for each set in A
    set_sums = {}
    for idx, set_elem in enumerate(A):
        total_value = 0
        for item in set_elem:
            corresponding_value = B[np.where(B[:, 0] == item)][0, 1]
            total_value += corresponding_value
        set_sums[idx] = total_value

    # Sort the dictionary based on the sum of values and get the top k keys
    top_k_sets = sorted(set_sums, key=set_sums.get, reverse=True)[:k]

    return top_k_sets


def get_topk_location_pairs(A: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the top-k location pairs with the minimum distance between arrays A and B.
    
    Args:
        A (ndarray): Array of locations with shape (n, 2).
        B (ndarray): Array of locations with shape (m, 2).
        k (int): Number of top location pairs to retrieve.
    
    Returns:
        ndarray: Array of indices representing the location pairs with the minimum distances, with shape (k, 2).
    """
    # Compute pairwise Euclidean distance between all locations in A and B
    distances = np.sqrt(np.sum((A[:, np.newaxis] - B) ** 2, axis=2))
    
    # Get the indices of the top-k minimum distances
    if distances.shape[0] == 1:
        idx = 0
    else:
        idx = np.argpartition(distances, k, axis=None)[:k]
    idx = np.unravel_index(idx, distances.shape)
    idx = np.column_stack(idx).reshape(k, 2)
    
    return idx



def find_clusters_containing_all_elements(A: list, B: list) -> list:
    """
    Return indexes of clusters in A that contain all elements of B.
    
    Args:
    - A (list of sets): A list where each set represents a cluster of point IDs.
    - B (list): A list of point IDs to search for in clusters.
    
    Returns:
    - list: A list of cluster indexes from A that contain all point IDs from B.
    """
    cluster_indexes = []
    
    for idx, cluster in enumerate(A):
        for pointId in B:    
            if pointId in cluster:
                cluster_indexes.append(idx)
                break
            
    return cluster_indexes


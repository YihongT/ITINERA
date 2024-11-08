import os
import re
import copy
import json
import time
import scipy
import folium
import datetime
import numpy as np
import concurrent.futures
from xyconvert import gcj2wgs

from model.utils.funcs import RecurringList, compute_consecutive_distances, find_indices, sample_items, reorder_list, remove_duplicates
from model.utils.all_prompts import get_start_point_prompt, get_dayplan_prompt, get_system_prompt, get_hour_prompt, check_final_reverse_prompt, process_input_prompt
from model.utils.funcs import get_user_data_embedding # this is only for the open-source version
from model.search import SearchEngine
from model.spatial import SpatialHandler

class ItiNera:
    def __init__(self, user_reqs, min_poi_candidate_num=19, keep_prob=0.8, thresh=1000, hours=0, proxy_call=None, citywalk=True, city=None, type='zh'):
        
        # Initialize core parameters and constants
        self.MODEL = "gpt-4o"
        # (hours, poi_num, distance_thresh)
        self.TIME2NUM = {1: (1, 3, 2000), 2: (1, 5, 3000), 3: (2, 7, 4000), 4: (2, 9, 5000), 5: (3, 11, 6000), 6: (3, 13, 7000), 7: (4, 15, 8000), 8: (4, 17, 9000)}
        
        self.min_poi_candidate_num = min_poi_candidate_num

        # Process user requirements and set parsed input data
        self.type = type
        self.proxy = proxy_call
        self.citywalk = citywalk
        self.user_reqs = user_reqs
        self.keep_prob = keep_prob
        self.thresh = thresh
        self.hours = self.get_hours(user_reqs, hours)
        
        # Process response data and initialize user data
        parsed_resquest = self.parse_user_request(user_reqs)
        self.must_see_poi_names, self.itinerary_pos_reqs, self.itinerary_neg_reqs, self.user_pos_reqs, self.user_neg_reqs, self.start_poi, self.end_poi = self.parse_user_input(parsed_resquest)
        
        # Initialize embeddings and user data
        self.user_favorites, self.embedding, self.r2i, self.i2r, self.must_see_pois = get_user_data_embedding(city_name=city, must_see_poi_names=self.must_see_poi_names, type=type)
        
        # Initialize spatial and search components based on hours
        self.maxPoiNum = self.TIME2NUM[self.hours][1]
        self.search_engine = SearchEngine(embedding=self.embedding, proxy=self.proxy)
        self.spatial_handler = SpatialHandler(data=self.user_favorites, min_clusters=self.TIME2NUM[self.hours][0], min_pois=self.maxPoiNum, citywalk=self.citywalk, citywalk_thresh=self.TIME2NUM[self.hours][2])

    def save_qualitative(self, new_numerical_order, full_response, clusters, lookup_dict):
        """
        Generates and saves interactive maps based on a new POI order and cluster information.
        The maps display markers, paths, and cluster centroids, saved to HTML files.

        Args:
            new_numerical_order (list): Ordered list of POI indices.
            full_response (str): JSON string containing POI response data.
            clusters (list): List of clusters with POI IDs.
            lookup_dict (dict): Dictionary for looking up POI names and IDs.
        """
        # Helper to create and save map with POI markers and paths
        def create_map(df, polyline_color="green", map_filename=""):
            # Convert GCJ-02 coordinates to WGS-84
            gcj_coords = np.array(df[['lon', 'lat']])
            wgs_coords = gcj2wgs(gcj_coords)
            
            # Update DataFrame with WGS-84 coordinates
            df['lon'], df['lat'] = wgs_coords[:, 0], wgs_coords[:, 1]
            
            # Create the map centered around the average WGS-84 coordinates
            m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=13)
            
            # Add markers to the map
            for i, (_, row) in enumerate(df.iterrows()):
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    popup=f"{i} - {row['name']}",
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(m)
            
            # Add polyline route between points
            points = list(zip(df['lat'], df['lon']))
            folium.PolyLine(points, color=polyline_color, weight=2.5, opacity=1).add_to(m)
            
            # Save the map to the specified file path
            m.save(f'./model/output/{map_filename}')

        # Generate current timestamp for filenames
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        current_time += f"_{self.type}"
        
        # Attempt to parse JSON response
        try:
            full_response_data = json.loads(full_response)
        except:
            try: 
                full_response_data = json.loads(full_response[8:-4])
            except:
                print("No JSON string found.")
                return {"error": full_response}

        cluster_list, unique_clusters = [], set()
        
        # Extract clusters for each POI in the response
        for key in full_response_data["pois"].keys():
            poi_id, name = lookup_dict[int(key)]
            for cluster_idx, cluster in enumerate(clusters):
                if self.i2r[poi_id] in cluster:
                    cluster_list.append((cluster_idx, poi_id, name))
                    unique_clusters.add(cluster_idx)

        # Generate the initial full TSP map with all POIs in the new order
        df_full_order = self.user_favorites.loc[new_numerical_order]
        create_map(df_full_order, map_filename=f"{current_time}_fulltsp.html")
        
        # Generate map focused on POIs listed in the response
        response_indices = np.array(new_numerical_order)[np.array(list(full_response_data["pois"].keys())).astype(int) - 1]
        df_response_order = self.user_favorites.loc[response_indices]
        create_map(df_response_order, map_filename=f"{current_time}.html")
        
        # Obtain centroids and cluster radius settings
        centroids = self.spatial_handler.get_cluster_centroids(clusters, lonlat=True)
        radius = self.spatial_handler.citywalk_thresh if self.mark_citywalk else self.thresh
        
        # Add circles around cluster centroids to indicate cluster boundaries
        m = folium.Map(location=[df_response_order['lat'].mean(), df_response_order['lon'].mean()], zoom_start=13)
        for cluster_id in unique_clusters:
            centroid_lon, centroid_lat = centroids[cluster_id]
            folium.Circle(
                location=[centroid_lat, centroid_lon],
                radius=radius / 2,
                color='blue',
                fill=True
            ).add_to(m)
        
        # Overlay the polyline path for POIs in the response
        points = list(zip(df_response_order['lat'], df_response_order['lon']))
        folium.PolyLine(points, color="green", weight=2.5, opacity=1).add_to(m)
        m.save(f'./model/output/{current_time}_response_clusters.html')
        
        with open(f'./model/output/result_{self.type}.json', "w", encoding="utf-8") as f:
            json.dump(full_response_data, f, ensure_ascii=True, indent=4)

        return full_response_data

    def get_hours(self, user_reqs, hours):
        """Get the number of hours for the plan; fetch from proxy if not provided."""
        if hours == 0:
            msg = [{"role": "user", "content": get_hour_prompt(user_reqs=user_reqs)}]
            response = self.proxy.chat(messages=msg, model=self.MODEL).replace("'", '"')
            hours = int(json.loads(response)[0])
        return hours

    def parse_user_request(self, user_reqs):
        """Fetch and parse user response from the proxy."""
        response = self.proxy.chat(messages=[{"role": "user", "content": process_input_prompt(user_input=user_reqs)}], model=self.MODEL).replace("'", '"')
        try:
            return json.loads(response)
        except:
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    print("Found string is not a valid JSON.")
            else:
                print("No JSON found in the string.")
            return {}
        
    def parse_user_input(self, structured_input):
        must_see_poi_names = []
        itinerary_pos_reqs, itinerary_neg_reqs = [], []
        user_pos_reqs, user_neg_reqs = [], []
        start_poi, end_poi = None, None

        for req in structured_input:
            if req["type"] is None:
                req["type"] = "地点"
            if req["type"] == "行程":
                itinerary_pos_reqs.append(req["pos"])
                if req["neg"] != None:
                    itinerary_neg_reqs.append(req["neg"])
            
            elif req["type"] in ["地点", "起点", "终点"]:
                if req["mustsee"] == True:
                    must_see_poi_names.append(req["pos"])    
                user_pos_reqs.append(req["pos"])
                user_neg_reqs.append(req["neg"])
                if req["type"] == "起点":
                    start_poi = req["pos"]
                if req["type"] == "终点":
                    end_poi = req["pos"]
            else:
                raise ValueError
        if len(user_pos_reqs) == 0:
            user_pos_reqs = itinerary_pos_reqs
        
        return must_see_poi_names, itinerary_pos_reqs, itinerary_neg_reqs, user_pos_reqs, user_neg_reqs, start_poi, end_poi
    def get_reqs_topk(self): 
        """
        Retrieves the top-k POIs for each user request and aggregates their scores.

        Returns:
            tuple: A sorted numpy array with unique POIs and their accumulated scores, 
                and a list of pseudo-must-see POIs.
        """
        def process_request(user_pos_req, user_neg_req):
            # Limit top-k to the minimum of available POIs or the defined candidate number
            top_k = min(self.user_favorites.shape[0], self.min_poi_candidate_num)
            req_pois = self.search_engine.query(desc=(user_pos_req, user_neg_req), top_k=top_k)

            # Collect top two POIs as pseudo-must-see if not already present
            pseudo_must_see_local = [int(poi) for poi in req_pois[:2, 0] if poi not in pseudo_must_see_pois]
            return req_pois, pseudo_must_see_local

        all_reqs_topk, result, pseudo_must_see_pois = [], [], []
        
        if len(self.user_pos_reqs) > 1:
            # Use a thread pool for concurrent processing of multiple positive requests
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for req_pois, pseudo_must_see_local in executor.map(process_request, self.user_pos_reqs, self.user_neg_reqs):
                    pseudo_must_see_pois.extend(pseudo_must_see_local)
                    all_reqs_topk.append(req_pois)
        elif len(self.user_pos_reqs) == 1:
            # Handle single request case directly
            neg_req = self.user_neg_reqs[0] if self.user_neg_reqs else None
            req_pois, pseudo_must_see_local = process_request(self.user_pos_reqs[0], neg_req)
            pseudo_must_see_pois.extend(pseudo_must_see_local)
            all_reqs_topk.append(req_pois)
        else:
            raise ValueError("No positive requests found")

        # Concatenate results and aggregate scores for unique POIs
        all_reqs_topk = np.concatenate(all_reqs_topk, axis=0)
        unique_values = np.unique(all_reqs_topk[:, 0])
        result = [[value, all_reqs_topk[all_reqs_topk[:, 0] == value][:, 1].sum()] for value in unique_values]
        result = np.array(result)
        
        # Sort by score in descending order
        sorted_reqs_topk = result[result[:, 1].argsort()[::-1]]

        return sorted_reqs_topk, pseudo_must_see_pois


    def get_poi_candidates(self, req_topk_pois: np.ndarray, must_see_poi_idlist: list, pseudo_must_see_pois):
        """
        Selects POI candidates based on top-k requested POIs, must-see POIs, and pseudo-must-see POIs,
        while maintaining spatial clustering.

        Args:
            req_topk_pois (np.ndarray): Array of top-k POIs from user requests.
            must_see_poi_idlist (list): List of must-see POI IDs.
            pseudo_must_see_pois (list): List of pseudo-must-see POI IDs.

        Returns:
            tuple: A list of POI pairs across clusters, selected clusters, cluster order, 
                POI candidates array, and POI candidate scores array.
        """
        
        req_topk_poi_idList = req_topk_pois[:, 0].astype(int).tolist()
        req_topk_poi_idList.extend(must_see_poi_idlist)
        
        all_poi_idlist = list(set(req_topk_poi_idList))
        poi_candidates, poi_candidate_scores, selected_cluster, self.mark_citywalk = self.spatial_handler.get_poi_candidates(all_poi_idlist, must_see_poi_idlist, req_topk_pois, self.min_poi_candidate_num, self.thresh, pseudo_must_see_pois)

        if min(1, self.min_poi_candidate_num / len(poi_candidates)) < 1:
            self.keep_prob = self.min_poi_candidate_num / len(poi_candidates)
            poi_candidates, poi_candidate_scores, selected_cluster = sample_items(poi_candidates, poi_candidate_scores, selected_cluster, keep_prob=self.keep_prob, keep_ids=pseudo_must_see_pois) # add some randomness
            selected_cluster = [sublist for sublist in selected_cluster if sublist]

        new_selected_cluster = copy.deepcopy(selected_cluster)
        for poi in poi_candidates:
            mark_included = False
            for cluster in selected_cluster:
                if poi in cluster:
                    mark_included = True
                    break
            if not mark_included:
                new_selected_cluster.append([poi])
        selected_cluster = new_selected_cluster
            
        order = reorder_list(poi_candidates, selected_cluster)
        poi_candidates, poi_candidate_scores = np.array(poi_candidates)[order], np.array(poi_candidate_scores)[order]
        clusterCentroids = self.spatial_handler.get_cluster_centroids(selected_cluster)
        clusters_order, _, _ = self.spatial_handler.get_tsp_order(locs=np.array(clusterCentroids))
        
        newclusters_order = []
        recurring_order = [i for i in clusters_order]
        recurring_order.append(recurring_order[0])
        distances = compute_consecutive_distances(np.array(clusterCentroids), recurring_order)
        topmax_distance_ids = distances.argsort()[-1:][::-1][0] # k=1
        newclusters_order.extend(clusters_order[topmax_distance_ids+1:])
        newclusters_order.extend(clusters_order[:topmax_distance_ids+1])
        clusters_order = newclusters_order

        all_pairs = self.spatial_handler.get_poi_pairs_across_clusters(clusters_order, selected_cluster) # those pairs are base on the reordered clusters

        return all_pairs, selected_cluster, clusters_order


    def calculate_ordered_route_info(self, new_numerical_order):
    
        distance_numerical_order = copy.deepcopy(new_numerical_order)
        distance_numerical_order.append(distance_numerical_order[0])
        points = self.user_favorites.loc[distance_numerical_order][["x", "y"]].astype("float").to_numpy()
        # Compute the squared differences
        squared_diffs = np.diff(points, axis=0) ** 2
        # Sum along the columns and then take the square root to get distances
        distances = np.sqrt(np.sum(squared_diffs, axis=1))
        distance_numerical_order_poiname = self.user_favorites.loc[distance_numerical_order, "name"].tolist()

        distance_string = ""
        for i in range(len(distances)):
            distance_string += f"'{distance_numerical_order_poiname[i]}' 距离 '{distance_numerical_order_poiname[i+1]}' {int(distances[i])} 米\n"
        
        new_numerical_ordered_poiname = self.user_favorites.loc[new_numerical_order, "name"].tolist()
        return_candidates = [str(i) for i in range(len(new_numerical_ordered_poiname))]

        if self.start_poi is not None and self.start_poi in new_numerical_ordered_poiname:
            response = [new_numerical_ordered_poiname.index(self.start_poi)]
        else:
            msg = [{"role": "user", "content": get_start_point_prompt(candidate_points=new_numerical_ordered_poiname, user_reqs=self.user_reqs, return_candidates=return_candidates, distance_string=distance_string)}]
            response = self.proxy.chat(messages=msg, model=self.MODEL).replace("'", '"')
            try:
                response = json.loads(response)
            except:
                response = ["0"]

        newnew_numerical_order = []
        newnew_numerical_order.extend(new_numerical_order[int(response[0]):])
        newnew_numerical_order.extend(new_numerical_order[:int(response[0])])
        new_numerical_order = newnew_numerical_order
    
        must_see_string, context_string = "", ""
        must_see_poi_idlist = self.must_see_pois
        if len(must_see_poi_idlist) > 0:
            must_see_poi_namelist = self.user_favorites.loc[must_see_poi_idlist, "name"].tolist()
            for name in self.must_see_poi_names:
                if name not in must_see_poi_namelist:
                    must_see_poi_namelist.append(name)
            must_see_string = str(must_see_poi_namelist)          
        else:
            must_see_string = "暂无必须选择的 POI"
        
        for i, (id, name, context) in enumerate(zip(new_numerical_order, self.user_favorites.loc[new_numerical_order, "name"], self.user_favorites.loc[new_numerical_order, "context"])):
            new_context = f'Sequence number {i+1}' + ':' + '"' + context.replace("\n", "")[:100] + '"'
            context_string += new_context + "\n"
        
        msg = [{"role": "user", "content": check_final_reverse_prompt(context=context_string, user_reqs=self.user_reqs)}]
        response = self.proxy.chat(messages=msg, model=self.MODEL).replace("'", '"')
        
        try:
            response = json.loads(response)
        except:
            try: 
                response = json.loads(response[8:-4])
            except:
                response = ["0"]
                
        if int(response[0]) == 0:
            new_numerical_order.reverse()

        context_string = ""        
        for i, (id, name, context) in enumerate(zip(new_numerical_order, self.user_favorites.loc[new_numerical_order, "name"], self.user_favorites.loc[new_numerical_order, "context"])):
            new_context = f'Sequence number {i+1}' + ':' + '"' + context.replace("\n", "")[:100] + '"'
            context_string += new_context + "\n"

        lookup_dict = {}
        for i, (id, name) in enumerate(zip(new_numerical_order, self.user_favorites.loc[new_numerical_order, "name"])):
            lookup_dict[i+1] = [self.r2i[id], name] # this ensures the real id from the database could be obtained through this lookup

        return context_string, must_see_string, lookup_dict, new_numerical_order


    def get_full_order(self, all_pairs: list, clusters: list, clusters_order:np.ndarray):
        """
        Constructs a full travel order (sequence) for Points of Interest (POIs) based on provided clusters, preferences, and pairings.
        This function organizes the POIs into an optimal order for visiting, starting from a particular point and ensuring that all must-see POIs are included.
        """

        new_numerical_order, lookup_dict = [], {}
        for i, cluster_order in enumerate(clusters_order):
            cluster = clusters[cluster_order]

            if i == 0:
                plan_candidates, possible_ori_context, possible_ori, id2content = [], "", [], {}
                order_within_cluster, _, _ = self.spatial_handler.get_tsp_order(cluster)
                ordered_pois_within_cluster = np.array(cluster)[order_within_cluster]
                ordered_poi_names_within_cluster = np.array(self.user_favorites.loc[ordered_pois_within_cluster, "name"].tolist()) 
                if len(cluster) < 2:
                    new_numerical_order.append(cluster[0])
                    start_poi = cluster[0]
                    
                else:
                    if len(all_pairs) == 0: # only one cluster
                        recurring_order = [i for i in order_within_cluster]
                        recurring_list = RecurringList(cluster)
                        recurring_order.append(recurring_order[0])
                        locs = self.user_favorites.loc[cluster][["x", "y"]].astype("float").to_numpy()
                        distances = compute_consecutive_distances(locs, recurring_order)
                        topmax_distance_ids = distances.argsort()[-1:][::-1] # k=2
                        for dist_id in topmax_distance_ids:
                            possible_ori.append((recurring_list[dist_id], 0)) # 0 means reverse
                            possible_ori.append((recurring_list[dist_id+1], 1)) # 1 means 
                    else:
                        end_poi = all_pairs[i][0]
                        cluster = ordered_pois_within_cluster
                        idx = find_indices(cluster, end_poi)
                        if idx == 0:
                            possible_ori.append((cluster[-1], 0))
                            possible_ori.append((cluster[1], 1))
                        elif idx == len(cluster)-1:
                            possible_ori.append((cluster[-2], 0))
                            possible_ori.append((cluster[0], 1))
                        else:
                            possible_ori.append((cluster[idx-1], 0))
                            possible_ori.append((cluster[idx+1], 1))
                    
                    for i, tup in enumerate(possible_ori):
                        possible_ori_context += f'Plan {i}: starting point: ["{str(self.user_favorites.loc[tup[0], "name"])}"]\n'
                        id2content[i] = (str(self.user_favorites.loc[tup[0], "name"]), tup[1])
                        plan_candidates.append(str(i))
                            
                    start_poi, direction = id2content[0] 
                    idx = find_indices(ordered_poi_names_within_cluster, start_poi)
                    
                    new_numerical_order = []
                    if direction == 0:
                        new_numerical_order.extend(ordered_pois_within_cluster[idx+1:])
                        new_numerical_order.extend(ordered_pois_within_cluster[:idx+1])
                        new_numerical_order.reverse()
                    else:
                        new_numerical_order.extend(ordered_pois_within_cluster[idx:])
                        new_numerical_order.extend(ordered_pois_within_cluster[:idx])

            elif i == len(clusters_order)-1:
                order_within_cluster, _, _ = self.spatial_handler.get_tsp_order(cluster)
                ordered_pois_within_cluster = np.array(cluster)[order_within_cluster]
                start_poi = all_pairs[i-1][1]
                idx = find_indices(ordered_pois_within_cluster, start_poi)
                
                new_numerical_order.extend(ordered_pois_within_cluster[idx:])
                new_numerical_order.extend(ordered_pois_within_cluster[:idx])

            else:                    
                if len(cluster) == 2:
                    start_poi, end_poi = cluster.index(all_pairs[i-1][1]), cluster.index(all_pairs[i][0])
                    new_numerical_order.extend([cluster[start_poi], cluster[end_poi]])
                elif len(cluster) > 2:
                    start_poi, end_poi = cluster.index(all_pairs[i-1][1]), cluster.index(all_pairs[i][0])
                    if end_poi == start_poi: # the function solve_tsp_with_start_end cannot handle this
                        points = self.user_favorites.loc[cluster][["x", "y"]].astype("float").to_numpy()
                        start_point = self.user_favorites.loc[cluster[start_poi]][["x", "y"]].astype("float").to_numpy()
                        # Calculate the squared Euclidean distance to all points in A
                        distances_squared = np.sum((points - start_point) ** 2, axis=1)
                        # Get the indices for the top k minimum distances
                        topk_indices = np.argsort(distances_squared)[:2]
                        end_poi = topk_indices[1]
                        
                    locs = self.user_favorites.loc[cluster, ["x", "y"]].astype("float").to_numpy()
                    dist_matrix = scipy.spatial.distance.cdist(locs, locs)
                    _, order_within_cluster = self.spatial_handler.solve_tsp_with_start_end(dist_matrix, start_poi, end_poi)
                    
                    if order_within_cluster[-1] in order_within_cluster[:-1]:
                        order_within_cluster = order_within_cluster[:-1]

                    ordered_pois_within_cluster = np.array(cluster)[order_within_cluster]
                    new_numerical_order.extend(ordered_pois_within_cluster)
                else:
                    new_numerical_order.extend(cluster)
                    
        new_numerical_order = remove_duplicates(new_numerical_order)
        context_string, must_see_string, lookup_dict, new_numerical_order = self.calculate_ordered_route_info(new_numerical_order)
        
        return new_numerical_order, lookup_dict, clusters, context_string, must_see_string
    
    
    def get_day_plan(self, new_numerical_order, context_string, must_see_string):

        if not self.mark_citywalk and self.citywalk:
            comments = "为满足您的需求，规划的路线可能需要交通工具，如需步行路线，可减少需求或增加收藏地点。"
        else:
            comments = ""
        
        messages = [
            {
                "role": "system", 
                "content": get_system_prompt(self.maxPoiNum, len(self.must_see_pois), len(new_numerical_order))
            }, 
            {
                "role": "user", 
                "content": get_dayplan_prompt(
                    context_string=context_string, must_see_string=must_see_string, keyword_reqs=self.user_pos_reqs, userReqList=self.user_reqs, 
                    maxPoiNum=self.maxPoiNum, numMustSee=len(self.must_see_pois), numCandidates=len(new_numerical_order), 
                    comments=comments, hours=self.hours, mark_citywalk=self.mark_citywalk, itinerary_reqs=(self.itinerary_pos_reqs, self.itinerary_neg_reqs),
                    start_end=(self.start_poi, self.end_poi)
                )
            }
        ]
        
        return self.proxy.chat(messages=messages, model=self.MODEL, temperature=0)

    
    def solve(self):
        
        req_topk_pois, pseudo_must_see_pois = self.get_reqs_topk()
        all_pairs, clusters, clusters_order = self.get_poi_candidates(req_topk_pois, self.must_see_pois, pseudo_must_see_pois)
        new_numerical_order, lookup_dict, clusters, context_string, must_see_string = self.get_full_order(all_pairs, clusters, clusters_order)
        full_response = self.get_day_plan(new_numerical_order, context_string, must_see_string)
        full_response_data = self.save_qualitative(new_numerical_order, full_response, clusters, lookup_dict)
        
        print(f'行程: \n{full_response_data}')
        return full_response_data, lookup_dict
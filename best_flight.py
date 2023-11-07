# https://www.w3schools.com/python/pandas/default.asp
# https://www.udacity.com/blog/2021/10/implementing-dijkstras-algorithm-in-python.html

import pandas as pd
import networkx as nx
import math
import heapq

filename = './Flight_Data.csv'
df = pd.read_csv(filename)
G = nx.DiGraph()


def add_nodes_to_DiWeGraph():
    uniqueSource = df['SourceAirport'].squeeze()
    uniqueDes = df['DestinationAirport'].squeeze()
    uniqueAirport = pd.concat([uniqueDes, uniqueSource]).drop_duplicates()
    uniqueAirport.reset_index()
    n = len(uniqueAirport)
    for i in range(n):
        G.add_node(uniqueAirport.iloc[i], heuristic=0)


def a_generate_cost(param):
    distance = param['Distance']
    time = param['FlyTime']
    price = param['Price']

    return 3*distance + 100*time + 20*price


def d_generated_cost(param):
    pass


def add_edge(i):
    G.clear_edges()
    if i == 1:
        for i in range(12850):
            cost = a_generate_cost(df.iloc[i])
            G.add_edge(df.iloc[i, 1], df.iloc[i, 2], weight=cost, distance='Distance', time='FlyTime', price='Price', flight_number=i)
    else:
        for i in range(df.size):
            cost = d_generated_cost(df.iloc[i])
            G.add_edge(df.iloc[i, 1], df.iloc[i, 2], weight=cost)


# Format the output
def desired_result_string(flight_number, SourceAirport, SourceAirport_Country, DestinationAirport,
                          DestinationAirport_Country, Distance, FlyTime, Price):
    return f'''Flight #{flight_number}: 
    From: {SourceAirport}, {SourceAirport_Country}
    To: {DestinationAirport}, {DestinationAirport_Country}
    Duration: {Distance}km
    Time: {FlyTime}h
    Price: {Price}$
    ----------------------------'''


def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371.0

    # Calculate the distance
    distance = r * c
    return distance


def a_star_heuristic(DestinationAirport):
    desLatitude = 0
    desLongitude = 0
    for i in range(12850):
        if df.iloc[i, 2] == DestinationAirport:
            desLatitude = df.iloc[i, 10]
            desLongitude = df.iloc[i, 11]
            break

    for node in enumerate(G.nodes):
        for i in range(12850):
            if df.iloc[i, 1] == node[1]:
                soLatitude = df.iloc[i, 5]
                soLongitude = df.iloc[i, 6]
                distance = calculate_distance(soLatitude, soLongitude, desLatitude, desLongitude)
                G.nodes[df.iloc[i,1]]['heuristic'] = 67043*distance
                break
            elif df.iloc[i, 2] == node[1]:
                soLatitude = df.iloc[i, 10]
                soLongitude = df.iloc[i, 11]
                distance = calculate_distance(soLatitude, soLongitude, desLatitude, desLongitude)
                G.nodes[df.iloc[i,2]]['heuristic'] = 67043*distance
                break


def a_star_algorithm(SourceAirport,DestinationAirport):
    a_star_heuristic(DestinationAirport)
    queue = [(0, SourceAirport)]
    visited = set()
    cost_so_far = {SourceAirport: 0}
    came_from = {SourceAirport: None}

    while queue:
        cost, current = heapq.heappop(queue)
        if current == DestinationAirport:
            break
        visited.add(current)
        for node in G.neighbors(current):
            new_cost = cost_so_far[current] + G.get_edge_data(current, node)['weight']
            if node not in cost_so_far or new_cost < cost_so_far[node]:
                cost_so_far[node] = new_cost
                priority = new_cost + G.nodes[node]['heuristic']
                heapq.heappush(queue, (priority, node))
                came_from[node] = current

    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()  # Reverse the path
    return path

print(1)
add_nodes_to_DiWeGraph()
print(2)
add_edge(1)
print(3)
path = a_star_algorithm("Imam Khomeini International Airport", "Gaziantep International Airport")
print(4)
print(path)

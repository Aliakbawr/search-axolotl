# https://www.w3schools.com/python/pandas/default.asp
# https://www.udacity.com/blog/2021/10/implementing-dijkstras-algorithm-in-python.html
# https://pypi.org/project/networkx/
# https://blog.enterprisedna.co/python-write-to-file/#:~:text=The%20write()%20method%20is,it%20to%20the%20specified%20file.

import networkx as nx
import pandas as pd
import math
import heapq

filename = 'Flight_Data.csv'
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


def a_star_generated_cost(param):
    distance = param['Distance']
    time = param['FlyTime']
    price = param['Price']

    return 3*distance + 100*time + 20*price


def dijkstra_generated_cost(param):
    distance = param['Distance']
    price = param['Price']
    time = param['FlyTime']
    w2 = 0.3  # Weight for distance
    w3 = 0.3  # Weight for price
    w1 = 0.4  # Weight for time

    cost = w1 * time + w2 * distance + w3 * price
    return cost


# Two options for creating edges using costs
def add_edge(i):
    if i == 1:
        for i in range(12850):
            cost = a_star_generated_cost(df.iloc[i])
            G.add_edge(df.iloc[i, 1], df.iloc[i, 2], weight=cost, distance='Distance', time='FlyTime', price='Price')
    else:
        for i in range(12849):
            cost = dijkstra_generated_cost(df.iloc[i])
            G.add_edge(df.iloc[i, 1], df.iloc[i, 2], weight=cost,
                       Distance=df.iloc[i, 13], FlyTime=df.iloc[i, 14], Price=df.iloc[i, 15])


# Dijkstra Implementation
def dijkstra_shortest_path(source, target):
    try:
        return nx.dijkstra_path(G, source, target, weight='weight')
    except nx.NetworkXNoPath:
        return None


# Desired string output
def desired_result_string(path):
    flight_number = 1
    total_time = 0
    total_price = 0
    total_distance = 0
    result_string = ""  # Initialize an empty string to store the output
    for u, v in zip(path, path[1:]):
        edge_data = G[u][v]
        distance = round(edge_data['Distance'])
        price = round(edge_data['Price'])
        time = round(edge_data['FlyTime'])
        total_time += time
        total_price += price
        total_distance += distance
        if shortest_path is not None:
            result_string += f'''
            Flight #{flight_number}: 
                From: {u}
                To: {v}
                Duration: {distance}km
                Time: {time}h
                Price: {price}$
            ----------------------------'''
            flight_number += 1
        else:
            result_string += "No path found."
    result_string += f'''
                Total Price: {total_price}$
                Total Duration: {total_distance} km
                Total Time: {total_time}h
                '''
    return result_string  # Return the result string


print('Enter Airports:')
# airport_1 = input()
# airport_2 = input()
add_nodes_to_DiWeGraph()
add_edge(2)

first_node = next(iter(G.nodes(data=True)))
airport_1 = 'Imam Khomeini International Airport'
airport_2 = 'John F Kennedy International Airport'

# Testing Dijkstra string output
shortest_path = dijkstra_shortest_path(airport_1, airport_2)
print(desired_result_string(shortest_path))
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
# https://www.w3schools.com/python/pandas/default.asp
# https://www.udacity.com/blog/2021/10/implementing-dijkstras-algorithm-in-python.html
import sys

import pandas as pd
import networkx as nx

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
        G.add_node(uniqueAirport.iloc[i])


def a_generate_cost(param):
    pass


def dijkstra_generated_cost(param):

    distance = param['Distance']
    price = param['Price']
    time = param['FlyTime']
    w2 = 0.3  # Weight for distance
    w3 = 0.3  # Weight for price
    w1 = 0.4  # Weight for time

    cost = w1 * time + w2 * distance + w3 * price
    return cost


def add_edge(i):
    if i == 1:
        for i in range(df.size):
            cost = a_generate_cost(df.iloc[i])
            G.add_edge(df.iloc[i, 1], df.iloc[i, 2], weight=cost)
    else:
        for i in range(12849):
            cost = dijkstra_generated_cost(df.iloc[i])
            G.add_edge(df.iloc[i, 1], df.iloc[i, 2], weight=cost)


# Dijkstra Implementation
def dijkstra_shortest_path(source, target):
    try:
        return nx.dijkstra_path(G, source, target, weight='weight')
    except nx.NetworkXNoPath:
        return None


def sum_of_distances(source, destination):
    try:
        distance = nx.shortest_path_length(G, source=source, target=destination, weight='distance')
        return distance
    except nx.NetworkXNoPath:
        return None

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


print('Enter Airports:')
airport_1 = input()
airport_2 = input()
add_nodes_to_DiWeGraph()
add_edge(2)
first_node = next(iter(G.nodes(data=True)))
airport_1 = 'Imam Khomeini International Airport'
airport_2 = 'Dubai International Airport'
shortest_path = dijkstra_shortest_path(airport_1, airport_2)
if shortest_path is not None:
    print(f"The shortest path from {airport_1} to {airport_2} is: {shortest_path}")
else:
    print("No path found.")
sum_distance = sum_of_distances(airport_1, airport_2)
if sum_distance is not None:
    print(f"The sum of distances from source to destination is: {sum_distance}")
else:
    print("No path found.")

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


def d_generated_cost(param):
    return 1


def add_edge(i):
    if i == 1:
        for i in range(df.size):
            cost = a_generate_cost(df.iloc[i])
            G.add_edge(df.iloc[i, 1], df.iloc[i, 2], weight=cost)
    else:
        for i in range(12849):
            cost = d_generated_cost(1)
            G.add_edge(df.iloc[i, 1], df.iloc[i, 2], weight=cost)


# Dijkstra Implementation
def dijkstra_shortest_path(source, target):
    try:
        return nx.dijkstra_path(G, source, target, weight='weight')
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
# print(G.nodes.get(1))
first_node = next(iter(G.nodes(data=True)))
# print(first_node)
airport_1 = 'Imam Khomeini International Airport'
airport_2 = 'Raleigh Durham International Airport'
shortest_path = dijkstra_shortest_path(airport_1, airport_2)
if shortest_path is not None:
    print(f"The shortest path from {airport_1} to {airport_2} is: {shortest_path}")
else:
    print("No path found.")

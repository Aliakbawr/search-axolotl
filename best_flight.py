# https://www.w3schools.com/python/pandas/default.asp
# https://www.udacity.com/blog/2021/10/implementing-dijkstras-algorithm-in-python.html
# https://pypi.org/project/networkx/
# https://blog.enterprisedna.co/python-write-to-file/#:~:text=The%20write()%20method%20is,it%20to%20the%20specified%20file.

import networkx as nx
import pandas as pd

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


# Two options for creating edges using costs
def add_edge(i):
    if i == 1:
        for i in range(df.size):
            cost = a_generate_cost(df.iloc[i])
            G.add_edge(df.iloc[i, 1], df.iloc[i, 2], weight=cost)
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

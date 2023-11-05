# https://www.w3schools.com/python/pandas/default.asp
# https://www.udacity.com/blog/2021/10/implementing-dijkstras-algorithm-in-python.html

import pandas as pd
import networkx as nx

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
        G.add_node(uniqueAirport.iloc[i])


def a_generate_cost(param):
    pass


def d_generated_cost(param):
    pass


def add_edge(i):
    G.clear_edges()
    if i == 1:
        for i in range(df.size):
            cost = a_generate_cost(df.iloc[i])
            G.add_edge(df.iloc[i, 1], df.iloc[i, 2], weight=cost)
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

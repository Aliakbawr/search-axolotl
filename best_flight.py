#https://www.w3schools.com/python/pandas/default.asp
#https://www.udacity.com/blog/2021/10/implementing-dijkstras-algorithm-in-python.html

import csv
import pandas as pd
import sys
import time

filename = './Flight_Data.csv'
df = pd.read_csv(filename)

#converting csv data into list of dictionaries
def read_csv(filename):
    flights = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            flight = {
                "Airline": row['Airline'],
                "SourceAirport": row['SourceAirport'],
                "DestinationAirport": row['DestinationAirport'],
                "Distance": float(row['Distance']),
                "FlyTime": float(row['FlyTime']),
                "Price": float(row['Price'])
            }
            flights.append(flight)
    return flights

#Format the output 
def desired_result_string(flight_number,SourceAirport,SourceAirport_Country,DestinationAirport,DestinationAirport_Country,Distance,FlyTime,Price):
    return f'''Flight #{flight_number}:
    From: {SourceAirport},{SourceAirport_Country}
    To: {DestinationAirport},{DestinationAirport_Country}
    Duration: {Distance}km
    Time: {FlyTime}h
    Price: {Price}$
    ----------------------------'''

#Implementing the graph class
class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)
        
    def construct_graph(self, nodes, init_graph):
        '''
        This method makes sure that the graph is symmetrical. In other words, if there's a path from node A to B with a value V, there needs to be a path from node B to node A with a value V.
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}
        
        graph.update(init_graph)
        
        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value
                    
        return graph
    
    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes
    
    def get_outgoing_edges(self, node):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections
    
    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]


#Modeling csv rows into lists 
SourceAirports = df['SourceAirport'].tolist()
SourceAirports= [x for i, x in enumerate(SourceAirports) if SourceAirports.index(x) == i]
# print(SourceAirports_list)
init_graph = {}

#Modeling list items into graph nodes
for node in SourceAirports:
    init_graph[node] = {}

#Modeling the distances into graph edges
for n in range(0,5):
    init_graph[df.iloc[n,1]][df.iloc[n,2]] = df.iloc[n,13]
    # print(df.iloc[n,1],df.iloc[n,2],df.iloc[n,13])

graph = Graph(SourceAirports, init_graph)
#execution time for Dijkstra function calculation	
start_time = time.process_time()
end_time = time.process_time()
execution_time = end_time - start_time
minutes = int(execution_time // 60)
seconds = int(execution_time % 60)
print(f"Dijkstra Algorithm\nExecution Time: {minutes}m{seconds}s\n.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-")

    
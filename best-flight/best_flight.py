import networkx as nx
import pandas as pd
import math
import heapq
import time


def create_graph():
    for i in range(6836):
        cost = generated_cost(df.iloc[i])
        G.add_edge(df.iloc[i, 1], df.iloc[i, 2], weight=cost,
                   Distance=df.iloc[i, 13], FlyTime=df.iloc[i, 14], Price=df.iloc[i, 15], Airline=df.iloc[i, 0])


def generated_cost(param):
    fly_time = param['FlyTime']
    distance = param['Distance']
    price = param['Price']

    w1 = 100  # Weight for time
    w2 = 3  # Weight for distance
    w3 = 20  # Weight for price

    cost = w1 * fly_time + w2 * distance + w3 * price
    return cost


# Dijkstra Implementation
def dijkstra_algorithm(source, target):
    shortest_paths = {source: (None, 0)}
    queue = [(0, source)]
    while queue:
        (dist, current) = heapq.heappop(queue)
        for neighbor, data in G[current].items():
            old_cost = shortest_paths.get(neighbor, (None, float('inf')))[1]
            new_cost = dist + data['weight']
            if new_cost < old_cost:
                heapq.heappush(queue, (new_cost, neighbor))
                shortest_paths[neighbor] = (current, new_cost)
    path = []
    while target is not None:
        path.append(target)
        next_node = shortest_paths[target][0]
        target = next_node
    path = path[::-1]
    return path


def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    d_longitude = lon2 - lon1
    d_latitude = lat2 - lat1
    a = math.sin(d_latitude / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_longitude / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371.0

    # Calculate the distance
    distance = r * c
    return distance


def a_star_heuristic(DestinationAirport):
    desLatitude = 0
    desLongitude = 0
    for i in range(6836):
        if df.iloc[i, 2] == DestinationAirport:
            desLatitude = df.iloc[i, 10]
            desLongitude = df.iloc[i, 11]
            break

    for node in enumerate(G.nodes):
        for i in range(6836):
            if df.iloc[i, 1] == node[1]:
                soLatitude = df.iloc[i, 5]
                soLongitude = df.iloc[i, 6]
                distance = calculate_distance(soLatitude, soLongitude, desLatitude, desLongitude)
                G.nodes[df.iloc[i, 1]]['heuristic'] = 67043 * distance
                break
            elif df.iloc[i, 2] == node[1]:
                soLatitude = df.iloc[i, 10]
                soLongitude = df.iloc[i, 11]
                distance = calculate_distance(soLatitude, soLongitude, desLatitude, desLongitude)
                G.nodes[df.iloc[i, 2]]['heuristic'] = 67043 * distance
                break


def a_star_algorithm(SourceAirport, DestinationAirport):
    a_star_heuristic(DestinationAirport)
    queue = [(0, SourceAirport)]
    visited = set()
    cost_so_far = {SourceAirport: 0}
    came_from = {SourceAirport: None}
    current = None

    while queue:
        cost, current = heapq.heappop(queue)
        if current == DestinationAirport:
            break
        visited.add(current)
        for node in list(G.successors(current)):
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
        fly_time = round(edge_data['FlyTime'])
        total_time += fly_time
        total_price += price
        total_distance += distance
        if path is not None:
            result_string += f'''
Flight #{flight_number} ({edge_data['Airline']}): 
From: {u}
To: {v}
Duration: {distance}km
Time: {fly_time}h
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


filename = 'Flight_Data.csv'
df = pd.read_csv(filename)
G = nx.DiGraph()
create_graph()
print("Enter The Source Airport And The Destination Airport")
user_input = input()
source_airport, destination_airport = user_input.split(" - ")
end_line = "\n.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-"

file = open('[2]-UIAI4021-PR1-Q1([A STAR]).txt', 'w')
start_time = time.time()
a_star_path = a_star_algorithm(source_airport, destination_airport)
end_time = time.time()
minute, second = divmod(end_time - start_time, 60)
a_star_time = f'{round(minute)}m{round(second)}s'
a_star_beginner = "A* Algorithm\nExecution Time: "
line = a_star_beginner + str(a_star_time) + end_line
file.write(line)
file.write(desired_result_string(a_star_path))
file.close()
file = open('[2]-UIAI4021-PR1-Q1([DIJKSTRA]).txt', 'w', encoding='utf-8')
start_time = time.time()
dijkstra_path = dijkstra_algorithm(source_airport, destination_airport)
end_time = time.time()
minute, second = divmod(end_time - start_time, 60)
dijkstra_time = f'{round(minute)}m{round(second)}s'
dijkstra_beginner = "Dijkstra Algorithm\nExecution Time: "
line = dijkstra_beginner + str(dijkstra_time) + end_line
file.write(line)
file.write(desired_result_string(dijkstra_path))
file.close()

print("Files Generated")

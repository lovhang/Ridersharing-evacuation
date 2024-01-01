import csv
from pandas import *
import matplotlib.pyplot as plt
import pickle
import numpy as np
import geopy.distance

x_min = -96.82036511
x_max = -94.34020794
y_min = 28.93832832
y_max = 30.88910535
travel_speed = [1666,1333, 666]

demand_data = read_csv(r'evacueeinfo\Zone_Coastal_eva1002_simulation_a0.03_b0.6_g0.4.csv')
x_demand_data = demand_data['x_cord'].tolist()
y_demand_data = demand_data['y_cord'].tolist()
near_demand_node = demand_data['road_ID'].tolist()
evacuee_type = demand_data['type'].tolist()
depature_time = demand_data['departure_time'].tolist()
resident_count = demand_data['resident_count'].tolist()
print(len(x_demand_data))
shelter_data = read_csv(r'evacueeinfo\shelter_location.csv')
x_shelter_data = shelter_data['x_cord'].tolist()
y_shelter_data = shelter_data['y_cord'].tolist()
near_shelter_node = shelter_data['JOIN_FID'].tolist()

data2 = read_csv(r'evacueeinfo\houston_nodes.csv')
x_node_cord = data2['x_cord'].tolist()
y_node_cord = data2['y_cord'].tolist()

x_min = min(min(x_demand_data), min(x_shelter_data))-0.01
y_min = min(min(y_demand_data), min(y_shelter_data))-0.01
x_max = max(max(x_demand_data), max(x_shelter_data))+0.01
y_max = max(max(y_demand_data), max(y_shelter_data))+0.01

x_cord = [x_min]
y_cord = [y_min]
near_node = [0]

for _ in x_demand_data:
    x_cord.append(_)
for _ in x_shelter_data:
    x_cord.append(_)
for _ in y_demand_data:
    y_cord.append(_)
for _ in y_shelter_data:
    y_cord.append(_)
for _ in near_demand_node:
    near_node.append(int(_))
for _ in near_shelter_node:
    near_node.append(int(_))
x_cord.append(x_max)
y_cord.append(y_max)
near_node.append(len(near_node))
ttnum = len(x_cord)
scenario = 2
sc_set = [i for i in range(scenario)]
ttm = np.array([None for i in range(scenario)])
spm = np.array([None for i in range(scenario)])

for i in sc_set:
    with open('realcaseNetwork/case{case_num}_SPM.pkl'.format(case_num = i+1), 'rb') as f:
        spm[i], ttm[i] = pickle.load(f)
    f.close()
num = len(spm[0][0])
print(ttm[1][428][418])
#print(num)
bigM = 10000
tt = np.array([[[bigM for k in sc_set] for j in range(num)] for i in range(num)])
sp = np.array([[[None for k in sc_set] for j in range(num)] for i in range(num)])
for i in range(0, num): # assign node other than dummy node
    for j in range(0, num):
        if j != i:
            for k in sc_set:
                #print(ttm[k][i][j])
                tt[i][j][k] = ttm[k][i][j][0]
                sp[i][j][k] = spm[k][i][j]
print(tt[428][418])
new_num = len(x_cord)
print(new_num)
ttn = np.array([[[bigM for k in sc_set] for j in range(new_num)] for i in range(new_num)])
spn = np.array([[[None for k in sc_set] for j in range(new_num)] for i in range(new_num)])
service_time=2.0
for i in range(1,new_num-1):
    o_node = near_node[i]
    for j in range(i+1,new_num-1):
        d_node = near_node[j]
        #print(o_node, d_node)
        if o_node == d_node:
            for k in sc_set:
                coord_1 = (y_cord[i], x_cord[i])
                coord_2 = (y_cord[j], x_cord[j])
                travel_time = geopy.distance.geodesic(coord_1,coord_2).km*1000/travel_speed[-1]
                ttn[i][j][k] = travel_time
                ttn[j][i][k] = travel_time
                spn[i][j][k] = [i,j]
                spn[j][i][k] = [j,i]
        else:
            for k in sc_set:
                coord_1 = (y_cord[i], x_cord[i])
                coord_1_node = (y_node_cord[o_node], x_node_cord[o_node])
                travel_time_1 = geopy.distance.geodesic(coord_1,coord_1_node).km*1000/travel_speed[-1]
                coord_2 = (y_cord[j], x_cord[j])
                coord_2_node = (y_node_cord[d_node], x_node_cord[d_node])
                travel_time_2 = geopy.distance.geodesic(coord_2, coord_2_node).km * 1000/travel_speed[-1]
                travel_time_3 = tt[o_node][d_node][k] + travel_time_1  + travel_time_2
                ttn[i][j][k] = travel_time_3
                ttn[j][i][k] = travel_time_3
                spn[i][j][k] = sp[o_node][d_node][k]
                spn[j][i][k] = sp[d_node][o_node][k]

sd = new_num-2
for i in range(0, new_num):
    for k in sc_set:
        ttn[0][i][k] = 1.0  # Zero node to all nodes for all scenarios
        ttn[i][0][k] = 1.0  # All nodes to 0 for all scenarios
        #ttn[sd][i][k] = 1.0  # superdriver to all nodes for all scenarios
        #ttn[i][sd][k] = 1.0  # all nodes to superdriver for all scenarios
        ttn[i][i][k] = 1.0  # node to itself for all scenarios
        ttn[i][-1][k] = 1.0  # node to last dummy node
        ttn[-1][i][k] = 1.0  # last dummy node to all node
#print(ttn[165][1007])
#print(near_node[164])
#print(near_node[1007])
#print(ttn[838][725])
#print(tt[411][408][0])
print(len(evacuee_type))
print(len(x_cord))
with open('case/casenum{n}_x_y_ttn_spn_type_dtime_count.pkl'.format(n = ttnum), 'wb') as f:
        pickle.dump([x_cord, y_cord, ttn, spn, evacuee_type, depature_time, resident_count], f)
f.close()
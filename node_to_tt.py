import csv
from pandas import *
import matplotlib.pyplot as plt
import pickle

#print(csvFile)
data = read_csv(r'G:\My Drive\postdoc\ridersharing\houston_road.csv')
links = data['nodes'].tolist()
length = data['Shape_Leng'].tolist()
rd_type = data['road_type'].tolist()

data2 = read_csv(r'G:\My Drive\postdoc\ridersharing\houston_nodes.csv')
x_cord = data2['x_cord'].tolist()
y_cord = data2['y_cord'].tolist()

num = len(x_cord)
print(num)
ns = [i for i in range(0, num)]
conn = [[0.0 for i in range(0,num)] for j in range(0,num)]
dist = [[1000 for i in range(0,num)] for j in range(0,num)]
tt = [[1000 for i in range(0,num)] for j in range(0,num)]
travel_speed = [1666,1333, 666 ]
for i in range(0,len(links)):
    temp = links[i].split(",")
    n1 = int(temp[0])
    n2 = int(temp[1])
    conn[n1][n2]=1.0
    conn[n2][n1]=1.0
    temp2 = length[i]
    dist[n1][n2]=temp2
    dist[n2][n1]=temp2
    match rd_type[i]:
        case 0:
            temp3 = temp2 /travel_speed[0]
            tt[n1][n2] = temp3
            tt[n2][n1] = temp3
        case 1:
            temp3 = temp2 / travel_speed[1]
            tt[n1][n2] = temp3
            tt[n2][n1] = temp3
        case 2:
            temp3 = temp2 / travel_speed[2]
            tt[n1][n2] = temp3
            tt[n2][n1] = temp3
setn1 = [0]
setn2 = [i for i in range(num)]
#============check if network is fully connected================
for k in range(num):
    for i in setn2:
        for j in setn1:
            if conn[i][j] == 1:
                setn1.append(i)
                setn2.remove(i)
                break
print(setn2)
with open('realcaseNetwork/case1.pkl', 'wb') as f:
    pickle.dump([x_cord, y_cord, dist, conn, tt], f)
    f.close()
#plt.scatter(x_cord,y_cord)
#plt.show()
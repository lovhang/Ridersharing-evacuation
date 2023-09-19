import copy

import numpy as np
import random as random
import math
import pickle
import matplotlib.pyplot as plt


class specifcnw():
    """
    Create specific network with known node to node shortest path
    :param address:
    :return: save network information to pkl
    """

    def __init__(self):
        num = 7  # # of nodes
        v_num = 2  # # of drivers
        qnum = 3  # # of passenger types
        Vh = np.array([i for i in range(v_num)])  # Set of drivers
        N = np.array([i for i in range(num)])  # Set of nodes, range from 0 to n+1
        Na = np.array([1, 2])  # Set of departure node of drivers
        Nd = np.array([5])  # Set of destination
        Np = np.array([3, 4])  # Set of pickup node for passengers
        Ns = np.array([1, 2, 3, 4, 5])  # Set of nodes except for dummy node 0,n+1
        Ok = np.array([1, 2])  # Set of departure node of drivers, same with Na
        x_cord = [0, 1.23, 6.85, 1.236, 5.65, 4.13, 8]  # X coordinate for nodes
        y_cord = [0, 1.42, 3.77, 3.56, 2.45, 2.98, 8]  # y coordinate for nodes
        qs = np.array([i for i in range(qnum)])  # Set of passenger types
        sq = np.array([1, 1, 1])  # service time for each types of passengers
        eq = np.array([1, 2, 3])  # number of passenger in each type of passengers
        tm = np.array([70, 80])  # Maximum travel time for drivers
        dp = [10, 10]  # departure time for drivers
        cv = np.array([5, 6])  # capacity for each driver
        Y0 = np.array([1, 2])  # driver family number, initial number of people in each drivers' car
        pr = np.array([0, 0, 0, 1, 2, 0, 0])  # priority of passenger node
        dm = np.zeros((num, qnum))  # demand for passenger at picup node for passengers
        dm[3][0] = 2;
        dm[3][1] = 2;
        dm[3][2] = 1
        dm[4][0] = 1;
        dm[4][1] = 2;
        dm[4][2] = 2
        # print('dm00')
        # print('dm00',dm[0][0])

        dis = np.zeros((num, num))  # distance node to node
        tt = np.zeros((num, num))
        for i in range(1, num - 1):
            for j in range(1, num - 1):
                if (i != j):
                    dis[i][j] = math.sqrt((x_cord[i] - x_cord[j]) ** 2 + (y_cord[i] - y_cord[j]) ** 2)
                    tt[i][j] = round(dis[i][j] * 10, 2)
        self.nwifo = [num, v_num, qnum, Vh, N, Na, Nd, Np, Ns, Ok, x_cord, y_cord, qs, sq, eq, tm, dp, cv, Y0, pr, dm,
                      dis, tt]

    def savenw(self, address: str):
        with open(address, 'wb') as f:
            pickle.dump(self.nwifo, f)

    def readnw(address: str):
        with open(address, 'rb') as f:
            a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23 = pickle.load(
                f)
        print(a23)
        print(a22)
        print(a21)


class generalnw():
    def __init__(self):
        print("start generalnw")

    def gennwcord(self, num: int, address: str):
        """
        randomly generate coordix based on number of node and save cord to address
        :param num: Number of nodes
        :param address: Address to save
        :return:
        """
        max_cord = 100
        cord = np.random.randint(0, max_cord, size=(num - 2, 2))  # random creates other 19 nodes location
        # print(cord.shape)
        cord = np.append(cord, [[max_cord, max_cord]], axis=0)
        cord = np.append([[0, 0]], cord, axis=0)
        # print(cord)
        x_cord = cord[:, 0]
        y_cord = cord[:, 1]
        with open(address, 'wb') as f:
            pickle.dump([x_cord, y_cord], f)
        # plt.scatter(x_cord, y_cord)
        # plt.show()


class manipulateCord():
    def __init__(self, adr: str):
        with open(adr, 'rb') as f:
            self.x_cord, self.y_cord = pickle.load(f)
        f.close()
        self.y_cord[10] = 20
        self.x_cord[10] = 50
        with open(adr, 'wb') as f:
            pickle.dump([self.x_cord, self.y_cord], f)
        f.close()


class readnw():
    """
    Read cord file and transfer to network
    """

    def __init__(self, address: str):
        with open(address, 'rb') as f:
            self.x_cord, self.y_cord = pickle.load(f)
        f.close()
        self.num = len(self.x_cord)
        self.connect = np.array([[0 for i in range(self.num)] for j in range(self.num)])
        self.dis = np.full((self.num, self.num), 0.0)
        for i in range(self.num):
            for j in range(self.num):
                if j != i:
                    self.dis[i][j] = round(
                        math.sqrt((self.x_cord[i] - self.x_cord[j]) ** 2 + (self.y_cord[i] - self.y_cord[j]) ** 2), 2)
        self.maxdis = 10000
        self.tt = np.full((self.num, self.num), self.maxdis)
        # print(self.cord)

    def createnw(self, address: str):
        """
        Read cord file to transfer to network parameter based on manually editing.  Plot the network
        """
        N1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        N2 = [0, 1, 8, 10, 11, 13, 16, 19, 20]
        x_cord = self.x_cord
        y_cord = self.y_cord
        num = self.num
        xmax = np.max(x_cord)
        ymax = np.max(y_cord)
        for i in N1:
            plt.scatter(x_cord[i], y_cord[i], color='blue')
        for i in N2:
            plt.scatter(x_cord[i], y_cord[i], color='yellow')

        maxinx = self.dis.argmin(axis=1)
        # print(maxinx)
        for i in range(num):
            self.connect[i][maxinx[i]] = 1
            self.connect[maxinx[i]][i] = 1
        # mannully add network connection ##################
        self.connect[9][18] = 1;
        self.connect[3][18] = 1;
        self.connect[12][16] = 1;
        self.connect[2][19] = 1;
        self.connect[2][5] = 1;
        self.connect[10][17] = 1;
        self.connect[7][13] = 1;
        self.connect[8][15] = 1;
        self.connect[4][11] = 1
        self.connect[1][9] = 1;
        self.connect[3][16] = 1;
        self.connect[16][19] = 1;
        self.connect[0][16] = 1;
        self.connect[2][10] = 1;
        self.connect[5][17] = 1;
        self.connect[8][17] = 1;
        self.connect[6][15] = 1;
        self.connect[13][14] = 1;
        self.connect[6][14] = 1;
        self.connect[14][20] = 1;
        self.connect[12][18] = 1;
        self.connect[4][12] = 1;
        self.connect[7][11] = 1;
        ############################################
        for i in range(num):
            for j in range(num):
                if j > i:
                    if self.connect[i][j] == 1:
                        self.connect[j][i] = 1
        # print(connect)
        # print(connect[7][13])
        # manipulate the connection of network
        # connect[10][17] = 0; connect[17][10] = 0
        for i in N1:
            for j in N1:
                if j > i and self.connect[i][j] > 0.1:
                    self.tt[i][j] = self.dis[i][j]
                    self.tt[j][i] = self.dis[i][j]
                    # plt.plot([x_cord[i],x_cord[j]],[y_cord[i],y_cord[j]], color='black')
        # print(self.tt)
        with open(address, 'wb') as f:
            pickle.dump([x_cord, y_cord, self.dis, self.connect, self.tt], f)
        f.close()
        # plt.show()


class manipulateNetwork():
    def __init__(self, adr: str):
        with open(adr, 'rb') as f:
            self.x_cord, self.y_cord, self.dist, self.connect, self.tt = pickle.load(f)
        f.close()
        self.maxdis = 10000

    def del_link(self, ori: int, des: int):
        self.connect[ori][des] = 0
        self.connect[des][ori] = 0
        self.tt[ori][des] = self.maxdis
        self.tt[des][ori] = self.maxdis

    def add_link(self, ori: int, des: int):
        self.connect[ori][des] = 1
        self.connect[des][ori] = 1
        self.tt[ori][des] = self.dist[ori][des]
        self.tt[des][ori] = self.dist[ori][des]

    def save(self, adr2: str):
        with open(adr2, 'wb') as f:
            pickle.dump([self.x_cord, self.y_cord, self.dist, self.connect, self.tt], f)
        f.close()


class visualize:
    def __init__(self, adr: str):
        with open(adr, 'rb') as f:
            self.x_cord, self.y_cord, self.dist, self.connect, self.tt = pickle.load(f)
        print(self.tt[10][17])
        print(self.tt[17][10])
        f.close()

    def showplt(self):
        for i in range(len(self.connect[0])):
            for j in range(len(self.connect[0])):
                plt.scatter(self.x_cord, self.y_cord, marker='o', color='blue')
                plt.text(self.x_cord[i] + 1.0, self.y_cord[i] + 1.0, "{i}".format(i=i))
                if j > i and self.connect[i][j] > 0.1:
                    # print(0)
                    plt.plot([self.x_cord[i], self.x_cord[j]], [self.y_cord[i], self.y_cord[j]], color='black')
        #

    def save(self, adr: str):
        plt.savefig(adr)
        plt.show()


# =================================== Farzane Added: August 2023 ===================================
class ShortestPath:
    def __init__(self, filename):
        with open(filename, 'rb') as file:
            self.x_cord, self.y_cord, self.dis, self.con, self.tvt = pickle.load(file)
        self.num = len(self.con[0])
        self.M = np.max(self.tvt)
        file.close()

    def getDijTable(self, nodes, origin, tvt):
        # initialization step
        M = self.M
        UV = [origin]
        for point in nodes:
            if point != origin:
                UV.append(point)
        Dij = [UV.copy(), [M for i in UV], [0 for i in UV]]
        Dij[1][0] = 0
        C_index = origin  # to avoid error: reference before assignment
        C = 0
        ite = 0
        # Condition 1
        while len(UV) > 1:
            # Find the node from UV with the minimum distance from origin
            min_distance = M
            for node in range(len(Dij[0])):
                if (Dij[0][node] in UV) and (Dij[1][node] <= min_distance):
                    min_distance = Dij[1][node]
                    C = Dij[0][node]
                    C_index = node
            # output: C as the selected node

            # Find all connections with C from UV
            # tt_from_C shows the travel time from C to each node in UV
            # if UV = [3, 5, 7], and tt_from_C = [12, M, 6],
            # then travel time from C to 3 is 12, to 5 is infeasible, and to 7 is 6'''
            tt_from_C = [tvt[C][i] for i in UV]
            # output: distance from C to unvisited nodes

            # For each feasible connection, find the path through C
            # Compare it with the current shortest path to each UV node
            for i in range(len(UV)):
                if tt_from_C[i] < M:  # Check if there is a feasible
                    new_path = Dij[1][C_index] + tt_from_C[i]
                    current_path = Dij[1][Dij[0].index(UV[i])]
                    if new_path <= current_path:
                        # New shortest path. Update Dij
                        Dij[1][Dij[0].index(UV[i])] = new_path
                        Dij[2][Dij[0].index(UV[i])] = C
            UV.remove(C)

        return Dij

    # ===============================   Farzane Added: September 18, 2023   =====================================
    def detAlternative(self, shortest_path_matrix, shortest_path_travel_time, malfunc_edges):
        # copy travel time to modification
        tvt = copy.deepcopy(self.tvt)

        # Given the order of malfunctioning edges, the shortest path matrix is modified.
        for m_index in range(len(malfunc_edges)):

            # Remove the edge from network by tvt
            for edge in malfunc_edges:
                tvt[edge[0]][edge[1]], tvt[edge[1]][edge[0]] = self.M, self.M

            # For each path i to j, check if the malfunctioned edge is on the path
            for i in range(len(shortest_path_matrix) - 1):
                for j in range(i + 1, len(shortest_path_matrix)):
                    indicator = shortest_path_matrix[i][j][-1]
                    condition = 0

                    # condition becomes 1 if malfunction edge is on the path
                    for index in range(len(indicator) - 1):
                        condition += int(indicator[index] == malfunc_edges[m_index][0] and indicator[index + 1] == \
                                    malfunc_edges[m_index][1])
                        condition += int(indicator[index] == malfunc_edges[m_index][1] and indicator[index + 1] == \
                                     malfunc_edges[m_index][0])

                    # remove malfunction edge from network and find another shortest path, add it to indicator.
                    if condition >= 1:
                        alt_dij = self.getDijTable(range(self.num), i, tvt)
                        alt_path, alt_path_tvt = self.getPath(alt_dij, j)

                        # update the SPMatrix and SPTimes
                        if len(shortest_path_matrix[i][j]) >= 2:
                            shortest_path_matrix[i][j].append(alt_path)
                            shortest_path_matrix[j][i].append(alt_path[::-1])
                            shortest_path_travel_time[j][i].append(alt_path_tvt)
                            shortest_path_travel_time[j][i].append(alt_path_tvt)
                        else:
                            shortest_path_matrix[i][j] = [shortest_path_matrix[i][j][0], alt_path]
                            shortest_path_matrix[j][i] = [shortest_path_matrix[j][i][0], alt_path[::-1]]
                            shortest_path_travel_time[i][j] = [shortest_path_travel_time[i][j][0], alt_path_tvt]
                            shortest_path_travel_time[j][i] = [shortest_path_travel_time[j][i][0], alt_path_tvt]
                # End of loop over SPMatrix columns
            # End of loop over SPMatrix rows
        # End of loop over malfunction edges
        return shortest_path_matrix, shortest_path_travel_time

    # ===============================   September 18, 2023   =====================================
    def getPath(self, dij, destination):
        p = [destination]
        traveler = destination
        traveltime = dij[1][dij[0].index(destination)]

        # find the path using dij
        while traveler != dij[0][0]:
            traveler_index = dij[0].index(traveler)
            p.append(dij[2][traveler_index])
            traveler = dij[2][traveler_index]

        return p[::-1], traveltime

    def getSpMatrix(self):
        ttm = [[self.M * 2 for j in range(self.num)] for j in range(self.num)]  # travel time matrix
        spm = [[None for j in range(self.num)] for j in range(self.num)]  # shortest path matrix
        for i in range(self.num - 1):
            dij = self.getDijTable(range(self.num), i, self.tvt)
            for j in range(i + 1, self.num):
                path, travel_time = self.getPath(dij, j)
                ttm[i][j] = [travel_time]
                ttm[j][i] = [travel_time]
                spm[i][j] = [path]
                spm[j][i] = [path[::-1]]

        return spm, ttm


if __name__ == '__main__':
    # randtrig = generalnw()
    # randtrig.gennwcord(30, "cord_40.pkl")
    # randtrig.readcord("network/cord3.pkl")
    # readnw("network1.pkl")
    # spctrig = specifcnw()
    # spctrig.savenw("network/network1.pkl")
    # Cordchange = manipulateCord("network/cord3.pkl")
    # rdtrig = readnw("network/cord3.pkl")
    # rdtrig.createnw("network/linknetwork1.pkl")
    # rdtrig.probset("network/network1_s.pkl")
    # with open("network/linknetwork1.pkl", 'rb') as f:
    # connect, tt = pickle.load(f)
    # print(connect)
    # print(tt)
    # with open('realnetwork.pkl', 'rb') as f:
    #   connect, tt = pickle.load(f)
    # f.close()
    # networkchange = manipulateNetwork("network/linknetwork1.pkl")
    # networkchange.del_link(2,5)
    # networkchange.save("network/linknetwork1_0.pkl")

    # ===============visualize network==================
    # visul = visualize("network/linknetwork1_0.pkl")
    # visul.showplt()
    # visul.save(f"network/linknetwork1_0.png")

    # =================================== Farzane Added ===================================
    # sp = ShortestPath("network/linknetwork1.pkl")
    # print(sp.con)
    # print(sp.tvt)
    # testing the functions in Shortest Path class
    # start = 1  # origin
    # finish = 8 # destination

    # Dij0 = sp.getDijTable(range(21), start)  # gets the nodes and the origin, return the Dij table
    # for i in range(21):
    #  finish = i
    # path, travel_time = sp.getPath(Dij0, finish)  # gets Dij and destination, returns the shortest path and distance

    # print(f'Path from {start} to {finish} is {path} with travel time {travel_time}\n')
    # print(f'Dij Table: {Dij0}')
    # =================================== Generate Shortest to all Nodes ============

    '''
    We give the algotithm the list of mal-functioned edges (malfunc_edge) and the time they malfunction (malfunc_time).
    For example: malfunc_edge = [(12, 4), (10, 17)] and malfunc_time = [10, 30] menas edge (12,4) becomes out of service
    at time 10 and edge (10, 17) at time 30.
    '''
    sp = ShortestPath("network/linknetwork1.pkl")
    spp, spt = sp.getSpMatrix()  # travel time matrix and corresponding shortest path matrix
    with open('network/linknetwork1_SPMatrix.pkl', 'wb') as f:
        pickle.dump([spp, spt], f)
    '''
    How it works?
    Given node i and the destination j, the shortest path is shortest_path[i][j] and travel 
    time is shortest_path_travel_time[i][j]
    '''

    # ============================= Farzane Added: September 18 2023 =============================
    spp_alt, spt_alt = sp.detAlternative(spp, spt, [[3, 16], [16, 12]])
    print(spp_alt[3][2], spt_alt[3][2])

    '''
    spp_alt has all the alternative paths from each node i to j. apt_alt has the first time that the edge must be passed
    before facing malfunction (SCT^m).
    The matrix will be saved .pkl file. 
    '''
    with open('network/linknetwork1_SPMatrixAlt.pkl', 'wb') as f:
        pickle.dump([spp_alt, spt_alt], f)
    # =================================== September 18 2023 ===================================

import copy
import time

import numpy as np
import random as random
import math
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class specifcnw:
    """
    Create specific network with known node to node shortest path
    :param :
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
        dm[3][0] = 2
        dm[3][1] = 2
        dm[3][2] = 1
        dm[4][0] = 1
        dm[4][1] = 2
        dm[4][2] = 2
        # print('dm00')
        # print('dm00',dm[0][0])

        dis = np.zeros((num, num))  # distance node to node
        tt = np.zeros((num, num))
        for i in range(1, num - 1):
            for j in range(1, num - 1):
                if i != j:
                    dis[i][j] = math.sqrt((x_cord[i] - x_cord[j]) ** 2 + (y_cord[i] - y_cord[j]) ** 2)
                    tt[i][j] = round(dis[i][j] * 10, 2)
        self.nwifo = [num, v_num, qnum, Vh, N, Na, Nd, Np, Ns, Ok, x_cord, y_cord, qs, sq, eq, tm, dp, cv, Y0, pr, dm,
                      dis, tt]

    def savenw(self, address: str):
        with open(address, 'wb') as f:
            pickle.dump(self.nwifo, f)

    def readnw(self, address: str):
        with open(address, 'rb') as f:
            a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23 = pickle.load(
                f)
        print(a23)
        print(a22)
        print(a21)


class generalnw:
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


class manipulateCord:
    def __init__(self, adr: str):
        with open(adr, 'rb') as f:
            self.x_cord, self.y_cord = pickle.load(f)
        f.close()
        self.y_cord[10] = 20
        self.x_cord[10] = 50
        with open(adr, 'wb') as f:
            pickle.dump([self.x_cord, self.y_cord], f)
        f.close()


class readnw:
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
        self.connect[7][15] = 1;
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


class manipulateNetwork:
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
        f.close()

    def showplt(self):
        for i in range(len(self.connect[0])):
            for j in range(len(self.connect[0])):
                plt.scatter(self.x_cord, self.y_cord, marker='o', color='blue')
                plt.text(self.x_cord[i] + 1.0, self.y_cord[i] + 1.0, "{i}".format(i=i))
                if j > i and self.connect[i][j] > 0.1:
                    # print(0)
                    plt.plot([self.x_cord[i], self.x_cord[j]], [self.y_cord[i], self.y_cord[j]], color='black')
                    plt.text((self.x_cord[i]+self.x_cord[j])/2, (self.y_cord[i]+self.y_cord[j])/2, s=f'{self.tt[i][j]}',
                             color='red')
        plt.show()
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

    def getDijTable(self, nodes_list, origin, tvt):
        # Putting the origin at the begining and the rest following
        UV = nodes_list
        UV.remove(origin)
        UV.insert(0, origin)
        C = origin
        M = self.M

        # Dij is saving all the information for the shortest paths
        # Dij_copy_0 and 1 are saving all the unvisited nodes (0 saves the node number and 1 saves the shortest path)
        # Dij_copy_0 and 1 are updated and every time one node is visited, it is removed form the list
        Dij = np.array([UV, [M for _ in UV], [0 for _ in UV]]).astype(int)
        Dij[1][0] = 0
        Dij_copy_0 = np.array(copy.copy(Dij[0]))
        Dij_copy_1 = np.array(copy.copy(Dij[1]))
        Dij_copy_index_Dij = np.array([np.where(Dij[0] == node)[0][0] for node in Dij_copy_0])

        while len(Dij_copy_0) > 1:
            # Step 1: Find the node from Dij_copy_1 with the minimum distance from origin, except for C
            C_index_in_copy = np.argmin(Dij_copy_1)
            C = Dij_copy_0[C_index_in_copy]
            C_index = np.where(Dij[0] == C)[0][0]

            Dij_copy_0 = np.delete(Dij_copy_0, C_index_in_copy)
            Dij_copy_1 = np.delete(Dij_copy_1, C_index_in_copy)
            Dij_copy_index_Dij = np.delete(Dij_copy_index_Dij, C_index_in_copy)

            # Step 2: Find the travel times from selected C to all unvisited nodes
            tt_from_C = [tvt[C][i] for i in Dij_copy_0]

            feasible = np.where(tt_from_C < M)[0]  # index of nodes that has tvt < M, referring to a node in Dij_copy_0

            # Step 3: Update the paths
            if len(feasible) != 0:
                # what if the lenght of feasible is 0?
                # in this case, we need to find a new C because it means this node has no other connections with others
                # and it is the dead-end
                for i in feasible:
                    new_path = Dij[1][C_index] + tt_from_C[i]
                    current_path = Dij[1][Dij_copy_index_Dij[i]]
                    if new_path < current_path:
                        # New shortest path. Update Dij
                        Dij[1][Dij_copy_index_Dij[i]] = new_path
                        Dij[2][Dij_copy_index_Dij[i]] = C
                        Dij_copy_1[i] = new_path
        return Dij

    def getPath(self, dij, destination):
        p = [destination]
        traveler = destination
        traveltime = dij[1][np.where(dij[0] == destination)[0][0]]

        # if there is a path between origin and destination, find it. O.W. let it go.
        if dij[1][np.where(dij[0] == destination)[0][0]] < self.M:
            # find the path using dij
            while traveler != int(dij[0][0]):
                traveler_index = int(np.where(dij[0] == traveler)[0][0])
                p.append(dij[2][traveler_index])
                traveler = dij[2][traveler_index]
            return p[::-1], traveltime
        else:
            return [0], self.M

    def detAlternative(self, shortest_path_matrix, shortest_path_travel_time, malfunc_edges, malfunc_time):
        # copy travel time to modification
        tvt = copy.deepcopy(self.tvt)
        con = self.con
        nodes = list(range(self.num))
        # Given the order of malfunctioning edges, the shortest path matrix is modified.
        for m_index in range(len(malfunc_edges)):
            # For each path i to j, check if the malfunctioned edge is on the path
            for i in range(self.num):
                for j in range(i, self.num):
                    # if i==j means we just need to add itself to the scenario
                    if i == j:
                        shortest_path_matrix[i][i].append(shortest_path_matrix[i][i][0])
                        shortest_path_travel_time[i][i].append(shortest_path_travel_time[i][i][0])
                    else:
                        indicator = shortest_path_matrix[i][j][-1]
                        condition = 0

                        # condition becomes 1 if malfunction edge is on the path
                        for index in range(len(indicator) - 1):
                            condition += int(indicator[index] == malfunc_edges[m_index][0] and indicator[index + 1] == \
                                        malfunc_edges[m_index][1])
                            condition += int(indicator[index] == malfunc_edges[m_index][1] and indicator[index + 1] == \
                                         malfunc_edges[m_index][0])

                        # Check whether we need to find an alternative path or not.
                        flag = 0
                        if condition >= 1:
                            cur_dij = self.getDijTable(nodes, i, tvt)
                            top_index = indicator.index(malfunc_edges[m_index][0])
                            bottom_index = indicator.index(malfunc_edges[m_index][1])
                            if top_index < bottom_index:
                                time_reach_m = cur_dij[1][np.where(cur_dij[0] == malfunc_edges[m_index][0])[0][0]]
                            else:
                                time_reach_m = cur_dij[1][np.where(cur_dij[0] == malfunc_edges[m_index][1])[0][0]]
                            if time_reach_m + tvt[i][j] > malfunc_time[m_index]:
                                # Yes, so remove the access on malfunctioned edge and find an alternative path.
                                flag = 1
                                con[malfunc_edges[m_index][0]][malfunc_edges[m_index][1]] = 0
                                con[malfunc_edges[m_index][1]][malfunc_edges[m_index][0]] = 0
                                tvt[malfunc_edges[m_index][0]][malfunc_edges[m_index][1]] = self.M
                                tvt[malfunc_edges[m_index][1]][malfunc_edges[m_index][0]] = self.M

                                alt_dij = self.getDijTable(nodes, i, tvt)
                                alt_path, alt_path_tvt = self.getPath(alt_dij, j)

                                # update the SPMatrix and SPTimes
                                if len(shortest_path_matrix[i][j]) >= 2:
                                    shortest_path_matrix[i][j].append(alt_path)
                                    shortest_path_matrix[j][i].append(alt_path[::-1])
                                    shortest_path_travel_time[i][j].append(alt_path_tvt)
                                    shortest_path_travel_time[j][i].append(alt_path_tvt)
                                else:
                                    shortest_path_matrix[i][j] = [shortest_path_matrix[i][j][0], alt_path]
                                    shortest_path_matrix[j][i] = [shortest_path_matrix[j][i][0], alt_path[::-1]]
                                    shortest_path_travel_time[i][j] = [shortest_path_travel_time[i][j][0], alt_path_tvt]
                                    shortest_path_travel_time[j][i] = [shortest_path_travel_time[j][i][0], alt_path_tvt]
                        if flag == 0:
                            if len(shortest_path_matrix[i][j]) >= 2:
                                shortest_path_matrix[i][j].append(shortest_path_matrix[i][j][-1])
                                shortest_path_matrix[j][i].append(shortest_path_matrix[j][i][-1])
                                shortest_path_travel_time[i][j].append(shortest_path_travel_time[i][j][-1])
                                shortest_path_travel_time[j][i].append(shortest_path_travel_time[j][i][-1])
                            else:
                                shortest_path_matrix[i][j] = [shortest_path_matrix[i][j][0], shortest_path_matrix[i][j][0]]
                                shortest_path_matrix[j][i] = [shortest_path_matrix[j][i][0], shortest_path_matrix[j][i][0]]
                                shortest_path_travel_time[i][j] = [shortest_path_travel_time[i][j][0], shortest_path_travel_time[i][j][0]]
                                shortest_path_travel_time[j][i] = [shortest_path_travel_time[j][i][0], shortest_path_travel_time[j][i][0]]
                # End of loop over SPMatrix columns
            # End of loop over SPMatrix rows
        # End of loop over malfunction edges
        return shortest_path_matrix, shortest_path_travel_time, self.tvt, self.con

    def getSpMatrix(self):
        ttm = [[self.M * 2 for j in range(self.num)] for j in range(self.num)]  # travel time matrix
        spm = [[j for j in range(self.num)] for j in range(self.num)]  # shortest path matrix

        for i in tqdm(range(self.num)):
            dij = self.getDijTable(list(range(self.num)), i, self.tvt)
            for j in range(i, self.num):
                if i == j:
                    ttm[i][i] = [self.M]
                    spm[i][i] = [i]
                else:
                    path, travel_time = self.getPath(dij, j)
                    ttm[i][j] = [travel_time]
                    ttm[j][i] = [travel_time]
                    spm[i][j] = [path]
                    spm[j][i] = [path[::-1]]
        return spm, ttm, self.tvt


if __name__ == '__main__':

    '''
    We give the algorithm the list of mal-functioned edges (malfunc_edge) and the time they malfunction (malfunc_time).
    For example: malfunc_edge = [(12, 4), (10, 17)] and malfunc_time = [10, 30] means edge (12,4) becomes out of service
    at time 10 and edge (10, 17) at time 30.
    '''
    scenario_times = [89]
    scenario_edges = [[16, 12]]

    with open('network/malfunction_road_info.pkl', 'wb') as f:
        pickle.dump([scenario_edges, scenario_times], f)
    f.close()

    real_case = False
    if real_case:
        sp = ShortestPath("realcaseNetwork/case1.pkl")
    else:
        sp = ShortestPath("network/linknetwork1.pkl")
    start = time.time()
    spp, spt, tvt = sp.getSpMatrix()  # travel time matrix and corresponding shortest path matrix
    middle = time.time()
    spp_alt, spt_alt, tvt, con = sp.detAlternative(spp, spt, scenario_edges, scenario_times)
    print(f'Shortest Path takes {middle-start} seconds. Finding Alternative path takes {time.time()-middle} seconds')

    with open('network/linknetwork1_SPMatrix.pkl', 'wb') as f:
        pickle.dump([spp, spt, tvt], f)
    f.close()
    with open('network/linknetwork1_SPMatrixAlt.pkl', 'wb') as f:
        pickle.dump([spp_alt, spt_alt, tvt], f)
    f.close()


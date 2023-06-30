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
        num = 7 # # of nodes
        v_num = 2 # # of drivers
        qnum = 3 # # of passenger types
        Vh = np.array([i for i in range(v_num)]) # Set of drivers
        N = np.array([i for i in range(num)]) # Set of nodes, range from 0 to n+1
        Na = np.array([1,2]) # Set of departure node of drivers
        Nd = np.array([5]) # Set of destination
        Np = np.array([3,4]) # Set of pickup node for passengers
        Ns = np.array([1,2,3,4,5]) # Set of nodes except for dummy node 0,n+1
        Ok = np.array([1,2]) # Set of departure node of drivers, same with Na
        x_cord = [0, 1.23, 6.85, 1.236, 5.65, 4.13,8] # X coordinate for nodes
        y_cord = [0, 1.42, 3.77, 3.56, 2.45, 2.98,8] # y coordinate for nodes
        qs = np.array([i for i in range(qnum)]) # Set of passenger types
        sq = np.array([1,1,1]) # service time for each types of passengers
        eq = np.array([1,2,3]) # number of passenger in each type of passengers
        tm = np.array([70,80]) # Maximum travel time for drivers
        dp = [10,10] #departure time for drivers
        cv = np.array([5,6]) #capacity for each driver
        Y0 = np.array([1,2]) #driver family number, initial number of people in each drivers' car
        pr = np.array([0,0,0,1,2,0,0]) # priority of passenger node
        dm = np.zeros((num, qnum)) # demand for passenger at picup node for passengers
        dm[3][0] = 2; dm[3][1] = 2; dm[3][2] = 1
        dm[4][0] = 1; dm[4][1] = 2; dm[4][2] = 2
        #print('dm00')
        #print('dm00',dm[0][0])

        dis = np.zeros((num,num)) #distance node to node
        tt = np.zeros((num,num))
        for i in range(1,num-1):
            for j in range(1,num-1):
                if(i != j):
                    dis[i][j] = math.sqrt((x_cord[i]-x_cord[j])**2 + (y_cord[i]-y_cord[j])**2)
                    tt[i][j] = round(dis[i][j]*10,2)
        self.nwifo = [num,v_num,qnum,Vh,N,Na,Nd,Np,Ns,Ok,x_cord,y_cord,qs,sq,eq,tm,dp,cv,Y0,pr,dm,dis,tt]

    def savenw(self, address: str):
        with open(address, 'wb') as f:
            pickle.dump(self.nwifo, f)

    def readnw(address:str):
        with open(address, 'rb') as f:
            a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23 = pickle.load(f)
        print(a23)
        print(a22)
        print(a21)

class generalnw():
    def __init__(self):
        print("start generalnw")
    def gennwcord(self, num:int, address:str):
        """
        randomly generate coordix based on number of node and save cord to address
        :param num: Number of nodes
        :param address: Address to save
        :return:
        """
        max_cord = 100
        cord = np.random.randint(0, max_cord, size=(num - 2, 2))  # random creates other 19 nodes location
        print(cord.shape)
        cord = np.append(cord, [[max_cord, max_cord]], axis=0)
        cord = np.append([[0, 0]], cord, axis=0)
        print(cord)
        x_cord = cord[:, 0]
        y_cord = cord[:, 1]
        with open(address, 'wb') as f:
            pickle.dump(cord, f)
        plt.scatter(x_cord, y_cord)
        plt.show()

class readnw():
    """
    Read cord file and transfer to network
    """
    def __init__(self, address:str):
        with open(address, 'rb') as f:
            self.cord = pickle.load(f)
        self.num = len(self.cord)
        self.x_cord = self.cord[:,0]
        self.y_cord = self.cord[:,1]
        self.maxdis = 5*np.max(self.x_cord)
        self.tt = np.full((self.num, self.num), self.maxdis)
        print(self.cord)
    def createnw(self, address:str):
        """
        Read cord file to transfer to network parameter based on manually editing.  Plot the network
        """
        num = self.num
        x_cord = self.x_cord
        y_cord = self.y_cord
        xmax = np.max(x_cord)
        ymax = np.max(y_cord)
        plt.scatter(x_cord, y_cord)
        #find shortest node for each node
        connect = np.zeros((num,num))
        dis = np.full((num,num), self.maxdis)

        for i in range(num):
            plt.text(x_cord[i], y_cord[i]+0.4, i)

        for i in range(num):
            for j in range(num):
                if j != i:
                    dis[i][j] = round(math.sqrt((x_cord[i]-x_cord[j])**2+(y_cord[i]-y_cord[j])**2),2)
        maxinx = dis.argmin(axis=1)
        #rint(maxinx)
        for i in range(num):
            connect[i][maxinx[i]] = 1
            connect[maxinx[i]][i] = 1
        # mannully add network connection ##################
        connect[9][18]=1;connect[3][18]=1;connect[12][16]=1;connect[2][19]=1;connect[2][5]=1;connect[10][17]=1;connect[7][13]=1;connect[8][15]=1;connect[4][11]=1
        ############################################
        for i in range(num):
            for j in range(num):
                if j > i:
                    if connect[i][j] == 1:
                        connect[j][i] = 1
        #print(connect)
        print(connect[7][13])
        for i in range(num):
            for j in range(num):
                if j>i and connect[i][j] > 0.1:
                    #x = np.linspace(x_cord[i], x_cord[j], 100)
                    #y = np.linspace(y_cord[i], y_cord[j], 100)
                    #plt.plot(x, y)
                    self.tt[i][j] = dis[i][j]
                    self.tt[j][i] = dis[i][j]
                    plt.plot([x_cord[i],x_cord[j]],[y_cord[i],y_cord[j]], color='black')
        print(self.tt)
        with open(address, 'wb') as f:
            pickle.dump([connect, self.tt], f)
        plt.show()
    def probset(self, nwaddress: str):
        """
        Setting other parameters for problem and save to network file
        parameter need to be decided:
        num # of nodes,
        v_num: # of drivers,
        q_num: # of passenger types,
        Vh: # of drivers,
        N: Set of nodes, range from 0 to n+1,
        Na: Set of departure node of drivers,
        Nd: Set of destination,
        Np; Set of pickup node for passengers,
        Ns: Set of nodes except for dummy node 0,n+1,
        Ok: Set of departure node of drivers, same with Na,
        qs: Set of passenger types,
        sq: service time for each types of passengers,
        eq: number of passenger in each type of passengers,
        tm: Maximum travel time for drivers,
        dp: departure time for drivers,
        cv: capacity for each driver,
        Y0: driver family number, initial number of people in each drivers' car,
        pr: priority of passenger node,
        dm: demand for passenger at picup node for passengers,
        x_cord: X coordinate for nodes,
        y_cord: Y coordinate for nodes,
        dis: Distance from node to node (shortest path),
        tt: Travel time from node to node (shortest path),
        :param address:
        :return: [num,v_num,q_num,Vh,N,Na,Nd,Np,Ns,Ok,x_cord,y_cord,qs,sq,eq,tm,dp,cv,Y0,pr,dm,dis,tt]
        """
        num = self.num
        v_num = 3
        q_num = 3
        Vh = np.array([i for i in range(v_num)])
        N = np.array([0,1,17,11,10,16,18,13,20])
        Na = np.array([16,18,13])
        Nd = np.array([10])
        Np = np.array([1,17,11])
        Ns = np.array([1,17,11,10,16,18,13])
        Ok = np.array([16,18,13])
        x_cord = self.x_cord
        y_cord = self.y_cord
        qs = np.array([i for i in range(q_num)])
        sq = np.array([1, 1, 1])
        eq = np.array([1, 2, 3])
        tm = np.array([70, 100,150])
        dp = [0,0,0]
        cv = np.array([5, 6, 5])
        Y0 = np.array([1, 2, 1])
        pr = np.zeros(num)
        pr[1]= 5; pr[11]= 3; pr[17]=1
        dm = np.zeros((num , q_num))
        dm[1][0] = 3;dm[1][1] = 2;dm[1][2] = 3
        dm[11][0] = 4;dm[11][1] = 2;dm[11][2] = 4
        dm[17][0] = 2;dm[17][1] = 3;dm[17][2] = 5;
        ## create travel time between each node
        rtt = np.zeros((num,num))
        rtt[1][16] = self.tt[1][9]+ self.tt[18][9]+self.tt[18][3]+self.tt[16][3]
        rtt[1][19] = self.tt[1][9]+ self.tt[18][9]+self.tt[18][3]+self.tt[16][3]+self.tt[16][19]
        rtt[1][18] = self.tt[1][9]+ self.tt[18][9]
        rtt[1][13] = self.tt[1][9]+ self.tt[18][9]+self.tt[18][12]+self.tt[4][12]+self.tt[4][11]+self.tt[7][11]+self.tt[7][13]
        rtt[1][15] = self.tt[1][9]+ self.tt[18][9]+self.tt[18][12]+self.tt[4][12]+self.tt[4][11]+self.tt[7][11]+self.tt[7][13]+self.tt[14][13]+self.tt[14][6]+self.tt[15][6]
        rtt[1][17] = self.tt[1][9]+ self.tt[18][9]+self.tt[18][3]+self.tt[16][3]+self.tt[16][19]+self.tt[2][19]+self.tt[2][10]+self.tt[10][17]
        rtt[1][11] = self.tt[1][9]+ self.tt[18][9]+self.tt[18][12]+self.tt[4][12]+self.tt[4][11]
        rtt[17][11] = self.tt[7][11]+self.tt[7][13]+self.tt[14][13]+self.tt[14][6]+self.tt[15][6]+self.tt[15][8]+self.tt[17][8]
        rtt[17][16] = self.tt[16][3]+self.tt[16][19]+self.tt[2][19]+self.tt[2][10]+self.tt[10][17]
        rtt[17][19] = self.tt[2][19]+self.tt[2][10]+self.tt[10][17]
        rtt[17][18] = self.tt[18][3]+self.tt[16][3]+self.tt[16][19]+self.tt[2][19]+self.tt[2][10]+self.tt[10][17]
        rtt[17][13] = self.tt[14][13]+self.tt[14][6]+self.tt[15][6]+self.tt[15][8]+self.tt[8][17]
        rtt[17][15] = self.tt[15][8]+self.tt[8][17]
        rtt[11][16] = self.tt[18][12]+self.tt[4][12]+self.tt[4][11]+ self.tt[18][3]+self.tt[16][3]
        rtt[11][19] = self.tt[18][12]+self.tt[4][12]+self.tt[4][11]+ self.tt[18][3]+self.tt[16][3]+self.tt[16][19]
        rtt[11][18] = self.tt[18][12]+self.tt[4][12]+self.tt[4][11]
        rtt[11][13] = self.tt[7][11]+self.tt[7][13]
        rtt[11][15] = self.tt[7][11]+self.tt[7][13]+self.tt[14][13]+self.tt[14][6]+self.tt[15][6]
        rtt[16][19] = self.tt[16][19]
        rtt[16][18] = self.tt[18][3]+self.tt[16][3]
        rtt[16][13] = self.tt[18][12]+self.tt[4][12]+self.tt[4][11]+self.tt[7][11]+self.tt[7][13]+self.tt[18][3]+self.tt[16][3]
        rtt[16][15] = self.tt[16][19]+self.tt[2][19]+self.tt[2][10]+self.tt[10][17]+self.tt[17][8]+self.tt[8][15]
        rtt[19][18] = self.tt[18][3]+self.tt[16][3]+self.tt[16][19]
        rtt[19][13] = self.tt[18][12]+self.tt[4][12]+self.tt[4][11]+self.tt[7][11]+self.tt[7][13]+self.tt[18][3]+self.tt[16][3]+self.tt[16][19]
        rtt[19][15] = self.tt[2][19]+self.tt[2][10]+self.tt[10][17]+self.tt[17][8]+self.tt[8][15]
        rtt[18][13] = self.tt[18][12]+self.tt[4][12]+self.tt[4][11]+self.tt[7][11]+self.tt[7][13]
        rtt[18][15] = self.tt[18][12]+self.tt[4][12]+self.tt[4][11]+self.tt[7][11]+self.tt[7][13]+self.tt[14][13]+self.tt[14][6]+self.tt[15][6]
        rtt[13][15] = self.tt[14][13]+self.tt[14][6]+self.tt[15][6]
        rtt[10][1] = self.tt[1][9]+ self.tt[18][9]+self.tt[18][3]+self.tt[16][3]+self.tt[16][19]+self.tt[2][19]+self.tt[2][10]
        rtt[10][17] = self.tt[10][17]
        rtt[10][11] = self.tt[18][12]+self.tt[4][12]+self.tt[4][11]+self.tt[18][3]+self.tt[16][3]+self.tt[16][19]+self.tt[2][19]+self.tt[2][10]
        rtt[10][16] = self.tt[16][19]+self.tt[2][19]+self.tt[2][10]
        rtt[10][19] = self.tt[2][19]+self.tt[2][10]
        rtt[10][18] = self.tt[18][3]+self.tt[16][3]+self.tt[16][19]+self.tt[2][19]+self.tt[2][10]
        rtt[10][13] = self.tt[14][13]+self.tt[14][6]+self.tt[15][6]+self.tt[10][17]+self.tt[17][8]+self.tt[8][15]
        rtt[10][15] = self.tt[10][17]+self.tt[17][8]+self.tt[8][15]
        for i in N:
            for j in N:
                if i != j and rtt[i][j]>0.1:
                    rtt[j][i] = rtt[i][j]
        rdis = np.copy(rtt)
        #print(rdis)
        nwinfo = [num,v_num,q_num,Vh,N,Na,Nd,Np,Ns,Ok,x_cord,y_cord,qs,sq,eq,tm,dp,cv,Y0,pr,dm,rdis,rtt]
        with open(nwaddress, 'wb') as f:
            pickle.dump(nwinfo, f)

class shortpath():
    '''gets the network of nodes with connections and travel times. Given a
    set of nodes, it returns the matrix stating the shortest path between nodes
    and corresponding travel time'''
    def __int__(self):
        pass
    def getshrstpath(self, nodes):
        with open("realnetwork.pkl", 'rb') as f:
            relation, ttime = pickle.load(f)
        f.close()
        connection = []
        traveltime = []
        for i in nodes:
            connection.append([relation[i][j] for j in nodes])
            traveltime.append([tt[i][j] for j in nodes])
        print(traveltime)






if __name__ == '__main__':
    #randtrig = generalnw()
    #randtrig.readcord("cord3.pkl")
    #readnw("network1.pkl")
    #spctrig = specifcnw()
    #spctrig.savenw("network1.pkl")
    #rdtrig = readnw("cord3.pkl")
    #rdtrig.createnw("realnetwork.pkl")
    #rdtrig.probset("network2_s.pkl")
    with open("realnetwork.pkl", 'rb') as f:
        connect, tt = pickle.load(f)
        #print(connect)
        #print(tt)
    f.close()
    sp = shortpath()
    sp.getshrstpath([0, 1, 2, 16, 9, 5])


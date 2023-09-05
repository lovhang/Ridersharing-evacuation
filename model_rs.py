
from docplex.mp.model import Model
from docplex.mp.conflict_refiner import ConflictRefiner
from docplex.mp.relaxer import Relaxer
import numpy as np
import math
import matplotlib.pyplot as plt
import random as random
import pickle


#read network from pkl
with open('network/realnetwork.pkl', 'rb') as f:
    connect, dist = pickle.load(f)
print(dist[2][5],dist[5][17], dist[2][17])
address = "network/network2_s.pkl"
with open(address, 'rb') as f:
    num, v_num, qnum, Vh, N, Na, Nd, Np, Ns, Ok, x_cord, y_cord, qs, sq, eq, tm, dp, cv, Y0, pr, dm, dis, tt = pickle.load(f)
#scenario change
#tt[16][8] = dis[16][12]+dis[12][4]+dis[4][11]+dis[11][7]+dis[7][13]+dis[14][13]+dis[14][6]+dis[6][15]+dis[15][8]
tt[10][8] = dis[10][2] + dis[2][5] + dis[5][17]
#tt[16][8] = temp168
#tt[8][16] = temp168
print(tt[10][8])
N = np.array([0,1,8,10,11,13,16,19,20])
N_0 = np.array([1,8,10,11,13,16,19])
N_1 = np.array([])
N_2 = np.array([10,11,13,16])
N_3 = np.array([])
N_S = np.array([1,8])
Vh = np.array([10,11,13,16,19])
sd = num-2 #super driver, num-1 is n+1, num-2 is super driver location
x_cord[sd]=60;y_cord[sd]=0 # preset the location of super driver
for i in N_0:
    tt[sd][i] = 10
    tt[i][sd] = 10
for i in N_0:
    tt[0][i]=1.0;tt[i][N[-1]]=1.0
N_dummy = np.array([0,20])
N_D = np.zeros(num) #destination
N_D[10]=8;N_D[11]=8;N_D[13]=8;N_D[16]=8;#;N_D[16]=17;N_D[18]=17;
dm = np.array([0.0 for i in range(0,num)])
dm[10]=2; dm[11]=1;dm[13]=2;dm[16]=2;
tm = np.array([0.0 for i in range(0,num)])
tm[10]=120;tm[11]=220;tm[13]=120;tm[16]=120;tm[19]=10000
cv = np.array([0.0 for i in range(0,num)])
cv[10]=3; cv[11]=5;cv[13]=5;cv[16]=2; cv[19]=100
mtw = 500; #maximum time window (planning period)
T_1=30;T_2=60;T_3=90
DP = np.array([mtw for i in range(0,num)]) #latest departure time, the time location is affected
DP[10] = T_1; DP[11] = 60
DT = np.array([mtw for i in range(0,num)]) #Detour time
DT[10] = 180; DT[11] = 180; DT[13] = 180; DT[16] = 180
tts = tt/100
M = 10000
w_s1 = 1000
w_s2 = 10
w_2 = 0.0001
mals1 = [[10,8],[16,8]]

####value store here####
x_var = np.zeros((num,num,num))
y_var = np.zeros((num,num))
r_var = np.zeros((num,num))
class model:
    def __init__(self):
        self.mdl = Model(name='Evacuation')
        self.x_indices = [(i,j,k) for i in N for j in N for k in Vh if i!=j]
        self.x = self.mdl.binary_var_dict(self.x_indices, name='x')
        self.yr_indices = [(i,k) for i in N for k in Vh]
        self.y = self.mdl.continuous_var_dict(self.yr_indices, name='y')
        self.r = self.mdl.continuous_var_dict(self.yr_indices, name='r')
    def createmodel(self):
        self.mdl.minimize(self.mdl.sum(self.x[0,j,k] for j in N_2 for k in Vh)+w_2*self.mdl.sum(self.r[i,k]+self.y[i,k] for i in N for k in Vh)+w_s1*self.x[0,sd,sd]+w_s2*self.mdl.sum(self.x[i,j,sd]*tt[i][j] for i in N for j in N if i!=j))
        # constriant 1.1
        for i in N_2:
            self.mdl.add_constraint(self.x[0,i,i]-self.x[N_D[i],N[-1],i] <= 0 )
        # constraint 1.2 and 1.3
        for i in N_1:
            self.mdl.add_constraint(self.x[0,i,i] == 1)
            self.mdl.add_constraint(self.x[N_D[i], N[-1], i] == 1)
        # constraint 1.4
        for k in Vh:
            for i in N_3:
                self.mdl.add_constraint(self.x[0,i,k] == 0)
        # constriant 4
        for k in Vh:
            for i in N_0:
                if k != i:
                    self.mdl.add_constraint(self.x[0,i,k] == 0)
        # constraint 4.1 for super driver, no other car is able to visit 19
        for _ in Vh:
            if _ != sd:
                self.mdl.add_constraint(self.mdl.sum(self.x[i,sd,_] for i in N if i != sd) ==0)
        #constraint 6
        for i in N_1:
            self.mdl.add_constraint(self.mdl.sum(self.x[i,j,k] for j in N if j!=i for k in Vh) ==1)
        for i in N_2:
            self.mdl.add_constraint(self.mdl.sum(self.x[i,j,k] for j in N if j!=i for k in Vh) ==1)
        for i in N_3:
            self.mdl.add_constraint(self.mdl.sum(self.x[i,j,k] for j in N if j!=i for k in Vh) ==1)
        # constraint 7
        for i in N_0:
            for k in Vh:
                self.mdl.add_constraint(self.mdl.sum(self.x[i,j,k] for j in N if j!=i) - self.mdl.sum(self.x[j,i,k] for j in N if j!= i) == 0)
        # constraint 8
        for k in Vh:
            for i in N_2:
                self.mdl.add_constraint(self.mdl.sum(self.x[j,i,k] for j in N if j != i) - self.mdl.sum(self.x[j,N_D[i],k] for j in N if j != N_D[i]) <= 0)
            for i in N_3:
                self.mdl.add_constraint(self.mdl.sum(self.x[j,i,k] for j in N if j != i) - self.mdl.sum(self.x[j,N_D[i],k] for j in N if j != N_D[i]) <= 0)
        # constraint 9
        self.mdl.add_constraints(self.r[i,k]-self.r[N_D[i],k] <= 0 for i in N_2 for k in Vh)
        # constraint 10
        self.mdl.add_constraints(self.r[j,k]-self.r[i,k]-M*self.x[i,j,k] >= tt[i][j] -M for i in N for j in N if j != i for k in N_1)
        self.mdl.add_constraints(self.r[j,k]-self.r[i,k]-M*self.x[i,j,k] >= tt[i][j] -M for i in N for j in N if j != i for k in N_2)
        self.mdl.add_constraints(self.r[j,k]-self.r[i,k]-M*self.x[i,j,k] >= tt[i][j] -M for i in N for j in N if j != i for k in N_3)
        # constraint 11
        self.mdl.add_constraints(self.r[j,sd]-self.r[i,sd]-M*self.x[i,j,sd] >= tts[i][j] -M for i in N for j in N if j != i )
        # constraint 12
        self.mdl.add_constraints(self.y[j,k]-self.y[i,k]-M*self.x[i,j,k] >= dm[j] - M for i in N for j in N if j != i for k in Vh)
        # constraint 13
        self.mdl.add_constraints(self.r[i,k]-self.r[N_D[i],k] <= 0 for i in N_1 for k in Vh)
        self.mdl.add_constraints(self.r[i,k]-self.r[N_D[i],k] <= 0 for i in N_2 for k in Vh)
        self.mdl.add_constraints(self.r[i,k]-self.r[N_D[i],k] <= 0 for i in N_3 for k in Vh)
        # constraint 15,16
        for k in Vh:
            for i in N_1:
                self.mdl.add_constraint(self.r[i,k] <= DP[i])
                self.mdl.add_constraint(self.r[N[-1],k] - self.r[i,k] <= DT[i])
            for i in N_2:
                self.mdl.add_constraint(self.r[i,k] <= DP[i])
                self.mdl.add_constraint(self.r[N[-1],k] - self.r[i,k] <= DT[i])
            for i in N_3:
                self.mdl.add_constraint(self.r[i,k] <= DP[i])
                self.mdl.add_constraint(self.r[N[-1],k] - self.r[i,k] <= DT[i])
        #self.mdl.add_constraints(self.r[N[-1],k] <= tm[k] for k in Vh)
        self.mdl.add_constraints(self.y[N[-1],k] <= cv[k] for k in Vh)
        self.mdl.add_constraints(self.mdl.sum(self.x[N[-1],j,k] for j in N_0) == 0 for k in Vh)
        self.mdl.add_constraints(self.mdl.sum(self.x[i,0,k] for i in N_0)==0 for k in Vh)
        ###check constraints###
        #self.mdl.add_constraint(self.x[0,19,19]==1)
        #self.mdl.add_constraint(self.x[19,10,19]==1)
        #self.mdl.add_constraint(self.x[10,16,19]==1)
        #self.mdl.add_constraint(self.x[16,11,19]==1)
        #self.mdl.add_constraint(self.x[11,13,19]==1)
        #self.mdl.add_constraint(self.x[13,8,19]==1)
        #self.mdl.add_constraint(self.x[8,1,19]==1)
        #self.mdl.add_constraint(self.x[1,20,19]==1)
        ###Scenario based constraints###
        for _ in Vh:
            self.mdl.add_constraint(self.r[10,_]>=25)
        for _ in mals1:
            n1=_[0]; n2=_[1]
            self.mdl.add_constraints(self.r[n2,k]+M*self.x[n1,n2,k] <= M+T_2 for k in Vh)
            self.mdl.add_constraints(self.r[n1,k]+M*self.x[n2,n1,k] <= M+T_2 for k in Vh)
        self.mdl.export_as_lp("evacuation_model")
    def solve(self):
        solution = self.mdl.solve(log_output=True)
        if solution == None:
            print(self.mdl.solve_details)
            cr = ConflictRefiner()
            cref = cr.refine_conflict(self.mdl, display=True)
            rx = Relaxer()
            rs = rx.relax(self.mdl)
            rx.print_information()
            rs.display()
            self.mdl.end()
        else:
            print("Model solved")
            print("objective value: "+ str(solution.get_objective_value()))
            for i,j,k in self.x_indices:
                if solution.get_value(self.x[i,j,k])>0.9:
                    x_var[i][j][k] = 1.0
                    print(f'Value of x[{i},{j},{k}] = 1')
                    y_var[i][k] = round(solution.get_value(self.y[i,k]))
                    r_var[i][k] = round(solution.get_value(self.r[i,k]),2)
                    print(f'Value of r[{i},{k}]:', r_var[i][k])
            for i,k in self.yr_indices:
             #   ytempvar = solution.get_value(self.y[i,k])
                if y_var[i][k]>0.1:
               #     y_var[i][k] = round(ytempvar)
                    print(f'Value of y[{i},{k}]:', y_var[i][k])
            for i,k in self.yr_indices:
                if r_var[i][k] > 0.1:
             #   rtempvar = solution.get_value(self.r[i,k])
              #  r_var[i][k] = round(rtempvar,2)
                    print(f'Value of r[{i},{k}]:', r_var[i][k])
            self.mdl.end()
class visualize():
    def __int__(self):
        self.carcl = np.array([None for _ in num], dtype = 'object')
        for _ in Vh:
            r = random.random()
            b = random.random()
            g = random.random()
            self.carcl[_] = (r, g, b)
    def shownetwork(self):
        for i in N_1:
            plt.plot(x_cord[i],y_cord[i], marker = 'o', color = 'r')
            plt.text(x_cord[i]+0.5,y_cord[i]+0.5, "{i}({j})".format(i=i,j=int(N_D[i])))
        for i in N_2:
            plt.plot(x_cord[i],y_cord[i], marker = 'o', color = 'y')
            plt.text(x_cord[i]+0.5,y_cord[i]+0.5, "{i}({j})".format(i=i,j=int(N_D[i])))
        for i in N_3:
            plt.plot(x_cord[i],y_cord[i], marker = 'o', color ='g')
            plt.text(x_cord[i]+0.5,y_cord[i]+0.5, "{i}({j})".format(i=i,j=int(N_D[i])))
        for i in N_S:
            plt.plot(x_cord[i],y_cord[i], marker ='o', color ='b')
            plt.text(x_cord[i]+0.5,y_cord[i]+0.5, "{i}({j})".format(i=i,j=int(N_D[i])))
        for i in N_dummy:
            plt.plot(x_cord[i],y_cord[i], marker ='o', color ='k')
            plt.text(x_cord[i]+0.5,y_cord[i]+0.5, "{i}({j})".format(i=i,j=int(N_D[i])))
        plt.plot(x_cord[sd],y_cord[sd], marker ='*', color ='m')
        plt.text(x_cord[sd]+0.5,y_cord[sd]+0.5, sd)
    def showroutes(self):
        for k in Vh:
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            for i in N_0:
                for j in N_0:
                    if j != i and x_var[i,j,k] == 1:
                        x = [x_cord[i], x_cord[j]]
                        y = [y_cord[i], y_cord[j]]
                        plt.plot(x,y, c = color)
    def show(self):
        plt.show()

#exp = model()
#exp.createmodel()
#exp.solve()
vis = visualize()
vis.shownetwork()
#vis.showroutes()
vis.show()


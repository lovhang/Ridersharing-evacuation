
from docplex.mp.model import Model
from docplex.mp import conflict_refiner
from docplex.mp.relaxer import Relaxer
import numpy as np
import math
import matplotlib.pyplot as plt
import random as random
import pickle


#read network from pkl and transfer it to t_ij^m
num = 21
scenario = 3
scset= [i for i in range(scenario)]
with open('network/linknetwork1_0.pkl', 'rb') as f:
    x_cord, y_cord, dist, connect, ttlink = pickle.load(f)
f.close()
# scenario = 3
ttm = np.array([None for i in range(3)])
spm = np.array([None for i in range(3)])
for i in scset:
    with open('network/linknetwork1_{i}_sp.pkl'.format(i=i), 'rb') as f:
        ttm[i] , spm[i] = pickle.load(f)
    f.close()
bigM = 10000 # set bigM to the ij that do not have alternative path so that x_ij^vm no going to select this path
tt = np.array([[[bigM for k in scset] for j in range(num)]for i in range(num)])
sp = np.array([[[None for k in scset] for j in range(num)]for i in range(num)])
for i in range(1,num-1):
    for j in range(1,num-1):
        if j != i:
            for k in scset:
                tt[i][j][k] = ttm[k][i][j]
                sp[i][j][k] = spm[k][i][j]
for i in range(0,num):
    for k in scset:
        tt[0][i][k] = 1.0;tt[i][0][k]=1.0;tt[num-1][i][k]=1.0;tt[i][num-1][k]=1.0
        tt[i][i][k] = 1.0
print(tt[3][8])
#print(sp[3][8])
#==================== start to create network =======================

#N = np.array([0,1,8,10,11,13,16,19,20])
#N_0 = np.array([1,8,10,11,13,16,19])
#N_1 = np.array([])
#N_2 = np.array([10,11,13,16])
#N_3 = np.array([])
#N_S = np.array([1,8])
#Vh = np.array([10,11,13,16,19])

#==================== case 2 ==================
N = np.array([0,1,2,3,4,5,6,8,10,19,20])
N_0 = np.array([1,2,3,4,5,6,8,10,19])
N_1 = np.array([])
N_2 = np.array([2,3,4,5,6,10])
N_3 = np.array([])
N_S = np.array([1,8])
Vh = np.array([2,3,4,5,6,10,19])
sd = num-2 #super driver, num-1 is n+1, num-2 is super driver location

N_dummy = np.array([0,20])
N_D = np.zeros(num) #destination
for i in range(0,num):
    N_D[i]=8
#N_D[10]=8;N_D[11]=8;N_D[13]=8;N_D[16]=8;#;N_D[16]=17;N_D[18]=17;
dm = np.array([0.0 for i in range(0,num)])
for i in N_2:
    dm[i] = 1.0
dm[10]=2; dm[11]=1;dm[13]=2;dm[16]=2;
#tm = np.array([0.0 for i in range(0,num)])
#tm[10]=220;tm[11]=220;tm[13]=120;tm[16]=120;tm[19]=10000
cv = np.array([2.0 for i in range(0,num)])
cv[10]=2; cv[11]=5;cv[13]=5;cv[16]=3; cv[19]=100
 #maximum time window (planning period)

sec_T = [30,150,1000]
LDP = np.array([bigM for i in range(0,num)]) #latest departure time, the time location is affected
#lDP[10] = mtw; lDP[11] = 60
EDP = np.array([0 for i in range(0,num)])
EDP[10]=35
DT = np.array([bigM for i in range(0,num)]) #Detour time
DT[10] = 280; DT[11] = 180; DT[13] = 180; DT[16] = 180; DT[19] = 10000
tts = tt/100
M = 20000
w_s1 = 10000
w_s2 = 10
w_2 = 0.0001
####value store here####
x_var = np.zeros((num,num,num,scenario))
y_var = np.zeros((num,num))
r_var = np.zeros((num,num))
class model:
    def __init__(self):
        self.mdl = Model(name='Evacuation')
        print(M)
        self.x_indices = [(i,j,k,m) for i in N for j in N if i!=j for k in Vh for m in scset ]
        self.x = self.mdl.binary_var_dict(self.x_indices, name='x')
        self.yr_indices = [(i,k) for i in N for k in Vh]
        self.y = self.mdl.continuous_var_dict(self.yr_indices, name='y')
        self.r = self.mdl.continuous_var_dict(self.yr_indices, name='r')
    def createmodel(self):
        self.mdl.minimize(self.mdl.sum(self.x[0,j,k,m] for j in N_2 for k in Vh for m in scset)+w_2*self.mdl.sum(self.r[i,k]+self.y[i,k] for i in N for k in Vh)+w_s1*self.mdl.sum(self.x[0,sd,sd,m] for m in scset)+w_s2*self.mdl.sum(self.x[i,j,sd,m]*tt[i][j][m] for i in N for j in N if i!=j for m in scset))
        # constriant 2
        for i in N_2:
            self.mdl.add_constraint(self.mdl.sum(self.x[0,i,i,m] for m in scset)-self.mdl.sum(self.x[N_D[i],N[-1],i,m] for m in scset) <= 0 )
        # constraint 3 and 2.1
        for i in N_1:
            self.mdl.add_constraint(self.mdl.sum(self.x[0,i,i,m]for m in scset) == 1)
            self.mdl.add_constraint(self.mdl.sum(self.x[N_D[i], N[-1], i, m] for m in scset) == 1)
        # constraint 4
        for k in Vh:
            for i in N_3:
                self.mdl.add_constraint(self.x[0,i,k] == 0)
        # constriant 5
        for k in Vh:
            for i in N_0:
                if k != i:
                    self.mdl.add_constraint(self.mdl.sum(self.x[0,i,k, m] for m in scset) == 0)
        # constraint 6 for super driver, no other car is able to visit 19
        for _ in Vh:
            if _ != sd:
                self.mdl.add_constraint(self.mdl.sum(self.x[i,sd,_, m] for i in N for m in scset if i != sd) ==0)
        #constraint 7 for at most one alternative path to be chosen
        self.mdl.add_constraints(self.mdl.sum(self.x[i,j,k,m] for m in scset) <= 1 for i in N for j in N if j!=i for k in Vh)
        #constraint 8
        for i in N_1:
            self.mdl.add_constraint(self.mdl.sum(self.x[i,j,k,m] for j in N if j!=i for k in Vh for m in scset) == 1)
        for i in N_2:
            self.mdl.add_constraint(self.mdl.sum(self.x[i,j,k,m] for j in N if j!=i for k in Vh for m in scset) == 1)
        for i in N_3:
            self.mdl.add_constraint(self.mdl.sum(self.x[i,j,k,m] for j in N if j!=i for k in Vh for m in scset) == 1)
        # constraint 9
        for i in N_0:
            for k in Vh:
                self.mdl.add_constraint(self.mdl.sum(self.x[i,j,k,m] for j in N if j!=i for m in scset) - self.mdl.sum(self.x[j,i,k,m] for j in N if j!= i for m in scset) == 0)
        # constraint 10
        for k in Vh:
            for i in N_2:
                self.mdl.add_constraint(self.mdl.sum(self.x[j,i,k,m] for j in N if j != i for m in scset) - self.mdl.sum(self.x[j,N_D[i],k,m] for j in N if j != N_D[i] for m in scset) <= 0)
            for i in N_3:
                self.mdl.add_constraint(self.mdl.sum(self.x[j,i,k,m] for j in N if j != i for m in scset) - self.mdl.sum(self.x[j,N_D[i],k,m] for j in N if j != N_D[i] for m in scset) <= 0)
        # constraint 9
        self.mdl.add_constraints(self.r[i,k]-self.r[N_D[i],k] <= 0 for i in N_2 for k in Vh)
        # constraint 11
        self.mdl.add_constraints(self.r[j,k]-self.r[i,k]-M*self.x[i,j,k,m] >= tt[i][j][m] -M for i in N for j in N if j != i for k in N_1 for m in scset)
        self.mdl.add_constraints(self.r[j,k]-self.r[i,k]-M*self.x[i,j,k,m] >= tt[i][j][m] -M for i in N for j in N if j != i for k in N_2 for m in scset)
        self.mdl.add_constraints(self.r[j,k]-self.r[i,k]-M*self.x[i,j,k,m] >= tt[i][j][m] -M for i in N for j in N if j != i for k in N_3 for m in scset)
        # constraint 12
        self.mdl.add_constraints(self.r[j,k]+M*self.x[i,j,k,m] <= M+sec_T[m] for i in N for j in N if j != i for k in Vh for m in scset )
        # constraint 13
        self.mdl.add_constraints(self.r[j,sd]-self.r[i,sd]-M*self.x[i,j,sd,m] >= tts[i][j][m] -M for i in N for j in N if j != i for m in scset )
        # constraint 14
        self.mdl.add_constraints(self.y[j,k]-self.y[i,k]-M*self.mdl.sum(self.x[i,j,k,m] for m in scset) >= dm[j] - M for i in N for j in N if j != i for k in Vh)
        # constraint 15
        self.mdl.add_constraints(self.r[i,k]-self.r[N_D[i],k] <= 0 for i in N_1 for k in Vh)
        self.mdl.add_constraints(self.r[i,k]-self.r[N_D[i],k] <= 0 for i in N_2 for k in Vh)
        self.mdl.add_constraints(self.r[i,k]-self.r[N_D[i],k] <= 0 for i in N_3 for k in Vh)
        # constraint 16,17,18
        for k in Vh:
            for i in N_1:
                self.mdl.add_constraint(self.r[i,k] <= LDP[i])
                self.mdl.add_constraint(self.r[i, k] >= EDP[i])
                self.mdl.add_constraint(self.r[N[-1],k] - self.r[i,k] <= DT[i])
            for i in N_2:
                self.mdl.add_constraint(self.r[i,k] <= LDP[i])
                self.mdl.add_constraint(self.r[i, k] >= EDP[i])
                self.mdl.add_constraint(self.r[N[-1],k] - self.r[i,k] <= DT[i])
            for i in N_3:
                self.mdl.add_constraint(self.r[i,k] <= LDP[i])
                self.mdl.add_constraint(self.r[i, k] >= EDP[i])
                self.mdl.add_constraint(self.r[N[-1],k] - self.r[i,k] <= DT[i])
        #self.mdl.add_constraints(self.r[N[-1],k] <= tm[k] for k in Vh)
        #constraint 19
        self.mdl.add_constraints(self.y[N[-1],k] <= cv[k] for k in Vh)
        #constraint 20,21
        self.mdl.add_constraints(self.mdl.sum(self.x[N[-1],j,k,m] for j in N_0 for m in scset) == 0 for k in Vh)
        self.mdl.add_constraints(self.mdl.sum(self.x[i,0,k,m] for i in N_0 for m in scset)==0 for k in Vh)
        ###check constraints###

        #self.mdl.add_constraint(self.mdl.sum(self.x[0,19,19,m] for m in scset)==0)
        #self.mdl.add_constraint(self.mdl.sum(self.x[0, 2, 2, m] for m in scset) == 1)
        self.mdl.export_as_lp("evacuation_model")
    def solve(self):
        solution = self.mdl.solve(log_output=True)
        if solution == None:
            print(self.mdl.solve_details)
            c_r = conflict_refiner
            cf = c_r.ConflictRefiner().refine_conflict(self.mdl, display=True)
            crr = c_r.ConflictRefinerResult(cf)
            crr.display()
            rx = Relaxer()
            rs = rx.relax(self.mdl)
            rx.print_information()
            rs.display()
            self.mdl.end()
        else:
            print("Model solved")
            print("objective value: "+ str(solution.get_objective_value()))
            for i,j,k,m in self.x_indices:
                if solution.get_value(self.x[i,j,k,m])>0.9:
                    x_var[i][j][k][m] = 1.0
                    print(f'Value of x[{i},{j},{k},{m}] = 1')
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
                    for m in scset:
                        if j != i and x_var[i,j,k,m] == 1:
                            x = [x_cord[i], x_cord[j]]
                            y = [y_cord[i], y_cord[j]]
                            plt.plot(x,y, c = color)
                            print('route {i} to {j} for vehicle {k}'.format(i=i, j=j, k=k))
                            print(sp[i][j][m])
    def show(self):
        plt.show()

exp = model()
exp.createmodel()
exp.solve()
vis = visualize()
vis.shownetwork()
vis.showroutes()
vis.show()


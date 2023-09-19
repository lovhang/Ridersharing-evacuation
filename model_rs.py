
from docplex.mp.model import Model
from docplex.mp import conflict_refiner
from docplex.mp.relaxer import Relaxer
import numpy as np
import math
import matplotlib.pyplot as plt
import random as random
import pickle
import itertools
import time

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

#print(tt[3][8])
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
sd = num-2 # super driver,num-1 is n+1, num-2 is super driver location
N = np.array([0,1,2,3,4,5,6,8,9,sd,20]) # 0,n+1,N_1,N_2,N_3,N_S, super
N_0 = np.array([1,2,3,4,5,6,8,9,sd])# N_1,N_2,N_3
N_1 = np.array([]) #N_1 driver
N_2 = np.array([2,3,4,5,6,9]) #N_2 : flexible driver
N_3 = np.array([]) #N_3 evacuee
N_S = np.array([1,8]) #shelter node,
Vh = np.array([2,3,4,5,6,9,sd]) # Driver index (N_1,N_2,N_3,super)
N_dummy = np.array([0,num-1])
N_e = [] # evacuee list, including N_1,N_2,N_3
for item in N_1:
    N_e.append(item)
for item in N_2:
    N_e.append(item)
for item in N_3:
    N_e.append(item)
#=============parameter ======================
N_D = np.zeros(num) #destination
for i in range(0,num):
    N_D[i]=8
#N_D[10]=8;N_D[11]=8;N_D[13]=8;N_D[16]=8;#;N_D[16]=17;N_D[18]=17;
dm = np.array([0.0 for i in range(0,num)]) # demand, passenger number of node i
for i in N_2:
    dm[i] = 1.0
dm[10]=2; dm[11]=1;dm[13]=2;dm[16]=2;

cv = np.array([2.0 for i in range(0,num)]) # capacity of node i
cv[10]=2; cv[11]=5;cv[13]=5;cv[16]=3; cv[sd]=100
 #maximum time window (planning period)
T_end = 1000
sec_T = [30,100,T_end]
LDP = np.array([T_end for i in range(0,num)]) #latest departure time, the time location is affected
#lDP[10] = mtw; lDP[11] = 60

EDP = np.array([0 for i in range(0,num)]) # Earlist departure time
#EDP[10]=0

DT = np.array([T_end for i in range(0,num)]) #Detour time
DT[2]=130;DT[3]=150;DT[10] = 280; DT[11] = 180; DT[13] = 180; DT[16] = 180; DT[19] = 10000
#print(tt[3][8])
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
        #print(M)
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
class state:
    def __init__(self, S,R,T,dt,Q,route, last):
        #print(input)
        #self.v = input
        self.S = S # Set of included node
        self.R = R # Set of evacuee node haven't been delivered
        self.T = T# [max(tt[0][self.v][0], EDP[self.v])] # Set of leaving time
        self.dt = dt # Set of maximum detour time for i in R
        self.Q = Q#[dm[self.v]] # Set of leaving number of people
        self.route = route
        self.last = last
        #for _ in N_0:
         #   if _ != sd and _ != self.v:
          #      self.C.add(_)
class dynapro:
    def __init__(self):
        print("initiate")
        self.solutionpoll = []
        self.Ncan = []
        self.candidateState = []
        self.ite = 0
        for item in N_1:
            self.Ncan.append(item)
        for item in N_2:
            self.Ncan.append(item)
        for _ in self.Ncan:
            S = [0,_]
            R = [_]
            et = max(tt[0][_][0], EDP[_]) #leaving time
            T = [et]
            dt = [et + DT[_]]
            Q = [dm[_]]
            route = [[N[0],_]]
            last = _
            s1 = state(S, R, T, dt, Q, route, last)
            self.solutionpoll.append(s1)
    def action(self, inputstate: state, des:int ):
        S = inputstate.S.copy()
        R = inputstate.R.copy()
        T = inputstate.T.copy()
        dt = inputstate.dt.copy()
        Q = inputstate.Q.copy()
        route = inputstate.route.copy()
        l = inputstate.last
        if des in N_S: # for des is the destination node
            if N_D[S[0]] == des: # if des is the destination of driver
                ifRnotd = False
                for item in R:
                    if N_D[item] != des:
                        ifRnotd = True
                        print("visit destination of driver while there still evacuees whos destination is other location")
                if ifRnotd == True:
                    return False
                else:
                    for i in range(len(tt[l][des])):
                        T_j = T[-1] + tt[l][des][i]  #
                        Q_j = Q[-1]
                        dtmin = min(dt)
                        if T_j <= min(sec_T[i], LDP[des], dtmin):
                            R = []
                            dt = []
                            Q_j = 0
                            S.append(des)
                            S.append(N[-1])
                            T.append(T_j)
                            Q.append(Q_j)
                            route.append(sp[l][des][i])
                            l = N[-1]
                            return state(S, R, T, dt, Q,route, l)
                    return False
            else:
                for i in range(len(tt[l][des])): # if i is the destination of evacuees
                    T_j = T[-1] + tt[l][des][i] #
                    Q_j = Q[-1]
                    dtmin = min(dt)
                    if T_j <= min(sec_T[i],LDP[des],dtmin ):
                        for j in range(len(R) - 1, -1, -1):
                            if N_D[R[j]] == des:
                                Q_j -= dm[R[j]]
                                del R[j]
                                del dt[j]

                        S.append(des)
                        T.append(T_j)
                        Q.append(Q_j)
                        route.append(sp[l][des][i])
                        l = des
                        return state(S,R,T,dt,Q,route,l)
                return False
        if des in N_e: # if des is node of evacuees
            try:
                if des in S:
                    raise ValueError("demand node {i} already visited before".format(i=des))
                else:
                    for i in range(len(tt[l][des])):
                        T_j = max(EDP[des],T[-1] + tt[l][des][i]) #
                        Q_j = Q[-1] + dm[des]
                        dtmin = min(dt)
                        if T_j <= min(sec_T[i],LDP[des],dtmin ) and Q_j <= cv[S[0]]:
                            R.append(des)
                            S.append(des)
                            T.append(T_j)
                            dt.append(T_j + DT[des])
                            Q.append(Q_j)
                            route.append(sp[l][des][i])
                            l = des
                            return state(S,R,T,dt,Q,route,l)
                    return False
            except ValueError as e:
                print(e)
        else:
            print('des not in N_e and N_S')
            return False
    def rundp(self, input:list):
        tempslopol = input
        returnstate = []
        if len(tempslopol)>=1:
            self.ite += 1
            #print(self.ite)
            for item in tempslopol:
                #print('S before', item.S)
                #print('last', item.last)
                if item.last == N[-1]:
                    self.candidateState.append(item)
                else:
                    can_node = []
                    S = item.S
                    for _ in N_e:
                        if _ not in S:
                            can_node.append(_)
                    for _ in S:
                        if N_D[_] in N_S:
                            can_node.append(int(N_D[_]))
                            break
                    for des in can_node:
                        #print('des', des)
                        tpstate = self.action(item,des)
                        if tpstate != False:
                            returnstate.append(tpstate)
            return self.rundp(returnstate)
        else:
            #print("end iteration")
            return None
class dymaster:
    def __init__(self, input:list):
        self.a = [[] for i in range(num)] # v,k,i
        self.route = [[] for i in range(num)] # v,k
        self.route2 = [[] for i in range(num)]
        for item in input:
            temp = [0.0 for i in range(num)]
            self.route[item.S[1]].append(item.S)
            self.route[item.S[1]].append(item.route)
            for _ in N_e:
                if _ in item.S:
                    temp[_] = 1.0
            self.a[item.S[1]].append(temp)
        self.mdl = Model(name='masterprob')
        self.maxlen = 0
        for item in self.a:
            lent = len(item)
            if lent > self.maxlen:
                self.maxlen = lent
        self.x_indices = [(v,k) for v in Vh for k in range(self.maxlen)]
        self.x = self.mdl.binary_var_dict(self.x_indices, name='x')
        self.result = []

    def masterprob(self):
        self.mdl.minimize(self.mdl.sum(self.x[v,k] for v in Vh for k in range(self.maxlen)))
        for i in N_e:
            temp = 0
            for v in Vh:
                for k in range(len(self.a[v])):
                    temp = temp + self.a[v][k][i]*self.x[v,k]
            self.mdl.add_constraint(temp == 1.0)
        for v in Vh:
            self.mdl.add_constraint(self.mdl.sum(self.x[v,k] for k in range(len(self.a[v]))) <= 1.0)
        self.mdl.export_as_lp("masterprob_model")
    def solve(self):
        solution = self.mdl.solve(log_output=True)

        for v in Vh:
            for k in range(self.maxlen):
                if solution.get_value(self.x[v,k]) > 0.9:
                    self.result.append(self.route[v][k])

        print(self.result)





class visualize:
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



def main(triger:int):
    #============run model on cplex==============
    if triger == 0:
        exp = model()
        exp.createmodel()
        exp.solve()
        #vis = visualize()
        #vis.shownetwork()
        #vis.showroutes()
        #vis.show()
    #============run model on dynamic programming==========
    elif triger == 1:
        print("start dynamic programming")
        ts = dynapro()
        print(ts.solutionpoll)
        start = time.time()
        ts.rundp(ts.solutionpoll)
        end = time.time()
        for item in ts.candidateState:
            print(item.S)
            #print(item.T)
            #print(item.route)
        #print(len(ts.candidateState))
        dyma = dymaster(ts.candidateState)
        dyma.masterprob()
        dyma.solve()
        print('time for generating routes: ', (end-start) * 10**3, "ms")





if __name__=="__main__":
    main(1)


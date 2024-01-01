from docplex.mp.model import Model
from docplex.mp import conflict_refiner
from docplex.mp.relaxer import Relaxer
import numpy as np
import matplotlib.pyplot as plt
import random as random
import pickle
import time
from tqdm import tqdm

# read network from pkl and transfer it to t_ij^m
num=0 #initiate num
sd = 0 # initiate sd
sc_set = [] # initaie sc_set
Xiaohang = 0

#with open('realcaseNetwork/1000demand_SPM.pkl', 'rb') as f:
    #x_cord, y_cord, tt, sp = pickle.load(f)
#f.close()
with open('case/casenum1011_x_y_ttn_spn_type_dtime_count.pkl', 'rb') as f:
     x_cord, y_cord, tt, sp, type, depart_time, passenger_count = pickle.load(f)
     f.close()
#=========add dummy node 0 and n+1=============
x_min = x_cord[0]
x_max = x_cord[-1]
y_min = y_cord[0]
y_max = y_cord[-1]
num = len(x_cord)
#print(num)
#sd = num-2
scenario = 2
sc_set = [i for i in range(scenario)]
bigM = 10000  # set bigM to the ij that do not have alternative path so that x_ij^vm no going to select this path
#print(np.shape(tt))
N = np.array([i for i in range(0,num)])  # 0,n+1,N_1,N_2,N_3,N_S, super
N_0 = N[1:-1]  # shelters,N_1,N_2,N_3, super driver
N_S= np.array(N[-2:-9:-1])
sd = N[-9]
#N_1 = np.array(N[1:10])  # N_1 driver
n_1 = []
n_2 = []
n_3 = []
for i in range(len(type)-1): #last one is super driver
    if type[i] == 1:
        n_1.append(i+1)
    elif type[i] == 2:
        n_2.append(i+1)
    elif type[i] == 0:
        n_3.append(i+1)
    else:
        print("exceptional type!")
        break
N_1 = np.array(n_1)
N_2 = np.array(n_2)
N_3 = np.array(n_3)
N_D = np.array([0 for i in range(0,num)])
N_dummy = np.array([0, num - 1])
print(len(N))
print(len(N_1))
print(len(N_2))
print(len(N_3))
print(N_D)
print(N_dummy)
for i in range(0, len(N_D)):
    N_D[i] = random.choice(N_S)
#print(N_D)
ScenTimes = [360,1000]
Vh = []  # Driver index (N_1,N_2,N_3,super)
for item in N_1:
    Vh.append(item)
for item in N_2:
    Vh.append(item)
Vh.append(sd)
Vh = np.sort(Vh)
N_e = []  # evacuee list, including N_1,N_2,N_3
N_23 = [] # evacuee list, including N_2,N_3
for item in N_1:
    N_e.append(item)
for item in N_2:
    N_e.append(item)
    N_23.append(item)
for item in N_3:
    N_e.append(item)
    N_23.append(item)
N_e = np.sort(N_e)
N_23 = np.sort(N_23)
#Vh = np.array([])
dm = np.array([0.0 for i in N])
#dm_seed = random.choices(range(1,3),k=num)
for i in range(len(passenger_count)):
    dm[i+1] = passenger_count[i]

cv = np.array([0.0 for i in N])
for i in range(0,num):
    cv[i] = random.randint(3,6)
#cv_seed = random.choices(range(3,5),k=num)
T_end = 1000
#sec_T = [500,1000]
#cenTimes.append(T_end)

# Tolerance
LDP = np.array([T_end for i in range(0, num)])  # latest departure time, the time location is affected
EDP = np.array([0 for i in range(0, num)])  # Earliest departure time
time_window = np.random.uniform(30,60,num) # time window uniformly distributed
for i in range(len(depart_time)):
    EDP[i+1] = depart_time[i]
    LDP[i+1] = EDP[i+1] + time_window[i]
#for i in range(0,num):
 #   temp = random.uniform(0,720)
  #  EDP[i] = temp
   # LDP[i] = temp + time_window[i]

DT = np.array([0 for i in range(0, num)])  # Detour time
#dtrate = 1.5 #detour rate
heuristic_ratio = 1.0 # should less than detour ratio
for i in N_e:
    for k in sc_set:
        if EDP[i] + tt[i][N_D[i]][k] <= ScenTimes[k]:
            DT[i] = round(tt[i][N_D[i]][k] * random.uniform(1.1, 1.5)) #detour rate
min_flex_interval = 20 # minmum flexible time interval for per passenger picked up.
#for i in N_1:
    #print(EDP[i], LDP[i], DT[i], tt[i][N_D[i]])
Largest_state_num = 1000000


tts = np.divide(tt, 100)  # Super driver travel time

M = 20000
Mhalf = 10000
w_s1 = 10000
w_s2 = 10
w_2 = 0.0001

# === decision values store here ===
x_var = np.zeros((num, num, num, scenario))  # x(i, j, v, m)
y_var = np.zeros((num, num))  # q(i, v)
r_var = np.zeros((num, num))  # r(i, v)


class model:
    def __init__(self):
        self.mdl = Model(name='Evacuation')
        # print(M)
        self.x_indices = [(i, j, k, m) for i in N for j in N if i != j for k in Vh for m in sc_set]
        self.x = self.mdl.binary_var_dict(self.x_indices, name='x')
        self.yr_indices = [(i, k) for i in N for k in Vh]
        self.y = self.mdl.continuous_var_dict(self.yr_indices, name='y')
        self.r = self.mdl.continuous_var_dict(self.yr_indices, name='r')
    def createmodel(self):
        self.mdl.minimize(self.mdl.sum(self.x[0, j, k, m] for j in N_2 for k in Vh for m in sc_set) +
                          w_2 * self.mdl.sum(self.r[i, k] + self.y[i, k] for i in N for k in Vh) +
                          w_s1 * self.mdl.sum(self.x[0, sd, sd, m] for m in sc_set) +
                          w_s2 * self.mdl.sum(
            self.x[i, j, sd, m] * tt[i][j][m] for i in N for j in N if i != j for m in sc_set))
        # constriant 2
        for i in N_2:
            self.mdl.add_constraint(self.mdl.sum(self.x[0, i, i, m] for m in sc_set) -
                                    self.mdl.sum(self.x[N_D[i], N[-1], i, m] for m in sc_set) <= 0)
        # constraint 3 and 2.1
        for i in N_1:
            self.mdl.add_constraint(self.mdl.sum(self.x[0, i, i, m] for m in sc_set) == 1)
            self.mdl.add_constraint(self.mdl.sum(self.x[N_D[i], N[-1], i, m] for m in sc_set) == 1)

        # constraint 4
        for k in Vh:
            for i in N_3:
                for m in sc_set:
                    self.mdl.add_constraint(self.x[0, i, k, m] == 0)

        # constriant 5
        for k in Vh:
            for i in N_0:
                if k != i:
                    self.mdl.add_constraint(self.mdl.sum(self.x[0, i, k, m] for m in sc_set) == 0)
        # constraint 6 for super driver, no other car is able to visit 19
        for _ in Vh:
            if _ != sd:
                self.mdl.add_constraint(self.mdl.sum(self.x[i, sd, _, m] for i in N for m in sc_set if i != sd) == 0)

        # constraint 7 for at most one alternative path to be chosen
        self.mdl.add_constraints(
            self.mdl.sum(self.x[i, j, k, m] for m in sc_set) <= 1 for i in N for j in N if j != i for k in Vh)
        # constraint 8
        for i in N_1:
            self.mdl.add_constraint(
                self.mdl.sum(self.x[i, j, k, m] for j in N if j != i for k in Vh for m in sc_set) == 1)
        for i in N_2:
            self.mdl.add_constraint(
                self.mdl.sum(self.x[i, j, k, m] for j in N if j != i for k in Vh for m in sc_set) == 1)
        for i in N_3:
            self.mdl.add_constraint(
                self.mdl.sum(self.x[i, j, k, m] for j in N if j != i for k in Vh for m in sc_set) == 1)
        # constraint 9
        for i in N_0:
            for k in Vh:
                self.mdl.add_constraint(
                    self.mdl.sum(self.x[i, j, k, m] for j in N if j != i for m in sc_set) - self.mdl.sum(
                        self.x[j, i, k, m] for j in N if j != i for m in sc_set) == 0)
        # constraint 10
        for k in Vh:
            for i in N_2:
                self.mdl.add_constraint(
                    self.mdl.sum(self.x[j, i, k, m] for j in N if j != i for m in sc_set) - self.mdl.sum(
                        self.x[j, N_D[i], k, m] for j in N if j != N_D[i] for m in sc_set) <= 0)
            for i in N_3:
                self.mdl.add_constraint(
                    self.mdl.sum(self.x[j, i, k, m] for j in N if j != i for m in sc_set) - self.mdl.sum(
                        self.x[j, N_D[i], k, m] for j in N if j != N_D[i] for m in sc_set) <= 0)
        # constraint 9
        self.mdl.add_constraints(self.r[i, k] - self.r[N_D[i], k] <= 0
                                 for i in N_2 for k in Vh)
        # constraint 11
        self.mdl.add_constraints(
            self.r[j, k] - self.r[i, k] - Mhalf * self.x[i, j, k, m] >= tt[i][j][m] - M
            for i in N for j in N if j != i for k in N_1 for m in sc_set)
        self.mdl.add_constraints(
            self.r[j, k] - self.r[i, k] - Mhalf * self.x[i, j, k, m] >= tt[i][j][m] - M
            for i in N for j in N if j != i for k in N_2 for m in sc_set)
        '''self.mdl.add_constraints(
            self.r[j, k] - self.r[i, k] - M * self.x[i, j, k, m] >= tt[i][j][m] - M for i in N for j in N if j != i for
            k in N_3 for m in sc_set)'''
        # constraint 12
        self.mdl.add_constraints(
            self.r[j, k] + M * self.x[i, j, k, m] <= M + ScenTimes[m]
            for i in N for j in N if j != i for k in Vh for m in sc_set)
        # constraint 13
        self.mdl.add_constraints(
            self.r[j, sd] - self.r[i, sd] - M * self.x[i, j, sd, m] >= tts[i][j][m] - M
            for i in N for j in N if j != i for m in sc_set)
        # constraint 14
        self.mdl.add_constraints(
            self.y[j, k] - self.y[i, k] - M * self.mdl.sum(self.x[i, j, k, m] for m in sc_set) >= dm[j] - M
            for i in N for j in N if j != i for k in Vh)
        # constraint 15
        self.mdl.add_constraints(self.r[i, k] - self.r[N_D[i], k] <= 0 for i in N_1 for k in Vh)
        self.mdl.add_constraints(self.r[i, k] - self.r[N_D[i], k] <= 0 for i in N_2 for k in Vh)
        self.mdl.add_constraints(self.r[i, k] - self.r[N_D[i], k] <= 0 for i in N_3 for k in Vh)
        # constraint 16,17,18
        for k in Vh:
            for i in N_1:
                self.mdl.add_constraint(self.r[i, k] <= LDP[i])
                self.mdl.add_constraint(self.r[i, k] >= EDP[i])
                self.mdl.add_constraint(self.r[N[-1], k] - self.r[i, k] <= DT[i])
            for i in N_2:
                self.mdl.add_constraint(self.r[i, k] <= LDP[i])
                self.mdl.add_constraint(self.r[i, k] >= EDP[i])
                self.mdl.add_constraint(self.r[N[-1], k] - self.r[i, k] <= DT[i])
            for i in N_3:
                self.mdl.add_constraint(self.r[i, k] <= LDP[i])
                self.mdl.add_constraint(self.r[i, k] >= EDP[i])
                self.mdl.add_constraint(self.r[N[-1], k] - self.r[i, k] <= DT[i])
        # self.mdl.add_constraints(self.r[N[-1],k] <= tm[k] for k in Vh)
        # constraint 19
        self.mdl.add_constraints(self.y[N[-1], k] <= cv[k] for k in Vh)
        # constraint 20,21
        self.mdl.add_constraints(self.mdl.sum(self.x[N[-1], j, k, m] for j in N_0 for m in sc_set) == 0 for k in Vh)
        self.mdl.add_constraints(self.mdl.sum(self.x[i, 0, k, m] for i in N_0 for m in sc_set) == 0 for k in Vh)
        ###check constraints###

        # self.mdl.add_constraint(self.mdl.sum(self.x[0,19,19,m] for m in sc_set)==0)
        # self.mdl.add_constraint(self.mdl.sum(self.x[0, 2, 2, m] for m in sc_set) == 1)
        self.mdl.export_as_lp("evacuation_model")

    def solve(self):
        start = time.time()
        solution = self.mdl.solve(log_output=True)
        sol_time = time.time()-start
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
            print("objective value: " + str(solution.get_objective_value()))
            for i, j, k, m in self.x_indices:
                if solution.get_value(self.x[i, j, k, m]) > 0.9:
                    x_var[i][j][k][m] = 1.0
                    print(f'Value of x[{i},{j},{k},{m}] = 1')
                    y_var[i][k] = round(solution.get_value(self.y[i, k]))
                    r_var[i][k] = round(solution.get_value(self.r[i, k]), 2)
                    print(f'Value of r[{i},{k}]:', r_var[i][k])
            for i, k in self.yr_indices:
                #   ytempvar = solution.get_value(self.y[i,k])
                if y_var[i][k] > 0.1:
                    #     y_var[i][k] = round(ytempvar)
                    print(f'Value of y[{i},{k}]:', y_var[i][k])
            for i, k in self.yr_indices:
                if r_var[i][k] > 0.1:
                    #   rtempvar = solution.get_value(self.r[i,k])
                    #  r_var[i][k] = round(rtempvar,2)
                    print(f'Value of r[{i},{k}]:', r_var[i][k])
            #with open(f'solutions/case_{case_number}_solutions.pkl', 'wb') as f:
             #   pickle.dump([x_var, y_var, r_var, sol_time], f)
            #f.close()
            self.mdl.end()


class state:
    def __init__(self, S, R, T, dt, Q, route, last, alternative, e_num):
        # print(input)
        # self.v = input
        self.S = S  # Set of included node
        self.R = R  # Set of evacuee node haven't been delivered
        self.T = T  # [max(tt[0][self.v][0], EDP[self.v])] # Set of leaving time
        self.dt = dt  # Set of maximum detour time for i in R
        self.Q = Q  # [dm[self.v]] # Set of leaving number of people
        self.route = route # corresponding routes, transportation nodes
        self.last = last # last node of current path
        self.alternative = alternative
        self.e_num = e_num # number of evacuees already visited before
        # for _ in N_0:
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
            S = [0, _]
            R = [_]
            et = max(tt[0][_][0], EDP[_])  # leaving time
            #print(et)
            T = [et]
            dt = [et + DT[_]]
            Q = [dm[_]]
            route = []
            last = _
            alternative = []
            e_num = 1 # number of evacuees picked before
            #flex_time = LDP[_] - et # flexible time
            s1 = state(S, R, T, dt, Q, route, last, alternative, e_num)
            self.solutionpoll.append(s1)
        self.state_num = len(self.solutionpoll)
    def action(self, inputstate: state, des: int):
        S = inputstate.S.copy()
        R = inputstate.R.copy()
        T = inputstate.T.copy()
        dt = inputstate.dt.copy()
        Q = inputstate.Q.copy()
        route = inputstate.route.copy()
        l = inputstate.last
        alternative = inputstate.alternative.copy()
        e_num = inputstate.e_num
        #print(N_S)
        if des in N_S:  # for des is the destination node
            if N_D[S[1]] == des:  # if des is the destination of driver
                ifRnotd = False
                for item in R:
                    if N_D[item] != des:
                        ifRnotd = True
                        #print(
                         #   "visit destination of driver while there still evacuees whos destination is other location")
                if ifRnotd == True:
                    return None
                else:
                    temp_alt = 0
                    dtmin = min(dt)
                    #earlist_ar_time = min(dtmin)
                    for i in range(len(tt[l][des])):
                        T_j = T[-1] + tt[l][des][i]  #
                        Q_j = Q[-1]
                        if T_j+min_flex_interval <= min(ScenTimes[i],dtmin): #deleted DT[des] here, not sure
                            R = []
                            dt = []
                            Q_j = 0
                            S.append(des)
                            #S.append(N[-1])
                            T.append(T_j)
                            Q.append(Q_j)
                            route.append(sp[l][des][i])
                            l = N[-1]
                            alternative.append(temp_alt)
                            #flex_time = min(flex_time, min(ScenTimes[i],dtmin)-T_j)
                            return state(S, R, T, dt, Q, route, l, alternative, e_num)
                        temp_alt += 1
                    return None
            else:
                temp_alt = 0
                for i in range(len(tt[l][des])):  # if i is the destination of evacuees
                    T_j = T[-1] + tt[l][des][i]  #
                    Q_j = Q[-1]
                    dtmin = min(dt)
                    latest_ar_time = min(ScenTimes[i], dtmin)
                    if T_j + min_flex_interval <= latest_ar_time: #deleted DT[des] here, not sure
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
                        alternative.append(temp_alt)
                        #flex_time = min(flex_time, latest_ar_time - T_j)
                        return state(S, R, T, dt, Q, route, l,alternative, e_num)
                    temp_alt += 1
                return None
        elif des in N_e:  # if des is node of evacuees
    #=================elimination rule 2: no repeat visit for shelter=================================================
            if N_D[des] in S:
                return None
    #=================end of rule 2=================================
            else:
                temp_alt = 0
                for i in range(len(tt[l][des])):
                    T_j = max(EDP[des], T[-1] + tt[l][des][i])  #
                    Q_j = Q[-1] + dm[des]
                    dtmin = min(dt)
#=======================elimination rule 1: each action will check detour time for all passenger on car===========================================
                    feasi = True # check if there node that is visited before is not able arrivle its destination by visiting node 'des'
                    #addtime = 0.0
                    for j in range(0,len(R)):
                        tempf = False
                        for k in range(len(tt[des][N_D[j]])):
                            tempt = T_j + tt[des][N_D[j]][k]
                            if tempt + min_flex_interval< dt[j] and tempt +min_flex_interval <= ScenTimes[k] and tempt+min_flex_interval <= dt[0]:
                                tempf = True

                                break
                        if tempf == False:
                            feasi = False
                            break
#=======================end of elimination rule 1======================================================
                    if feasi == False:
                        return None
                    else:
#========================elimination rule 5 =====================================
                        des_set = [N_D[des]]
                        for item in R:
                            if N_D[item] not in des_set:
                                des_set.append(N_D[item])
                        des_TT = 0
                        if len(des_set) == 1:
                            for k in sc_set:
                                des_TT += tt[des][des_set[0]][k]
                            des_TT = des_TT/scenario
                        else:
                            des_TT=0
                            for l in sc_set:
                                des_TT = tt[des_set[-1]][des_set[0]][l]
                                for k in range(len(des_set)-1):
                                    des_TT = tt[des_set[k]][des_set[k+1]][l]
                            des_TT = des_TT/scenario
                        if T_j + des_TT*heuristic_ratio+min_flex_interval> dt[0]:
                            return None
#=======================end of elimination rule 5======================================================
                        latest_ar_time = min(ScenTimes[i], LDP[des], dtmin)
                        if T_j + min_flex_interval <= latest_ar_time and Q_j <= cv[S[1]]:
                            R.append(des)
                            S.append(des)
                            T.append(T_j)
                            dt.append(T_j + DT[des])
                            Q.append(Q_j)
                            route.append(sp[l][des][i])
                            l = des
                            alternative.append(temp_alt)
                            e_num += 1
                            #flex_time = min(flex_time, latest_ar_time)
                            return state(S, R, T, dt, Q, route, l, alternative, e_num)
                    temp_alt += 1
                return None
        else:
            print('des not in N_e and N_S')
            return False
    def rundp(self, input):
        try:
            tempslopol = input
            returnstate = []
            if len(tempslopol) >= 1:
                self.ite += 1
                print('====iteration: {i}===='.format(i=self.ite))
                print('====current states num: {num} ,start to extend state===='.format(num=len(self.solutionpoll)))
                for item in tqdm(tempslopol):
                    # print('S before', item.S)
                    # print('last', item.last)
                    can_node = []
                    S = item.S
                    R = item.R
                    e_num = item.e_num
                    #print(item.S)
                    max_p_num = 10
                    for _ in N_23:
                        # ====================rule 4=========================
                        # if e_num <= max_p_num # limit total number of evacuees to be visited
                        # ====================end rule 4 ====================
                        if e_num <= max_p_num and _ not in S:
                            can_node.append(_)
                    for _ in R:
                        #if N_D[_] not in S:
                        can_node.append(N_D[_])
                    #for _ in S:
                     #   if N_D[_] in N_S:
                      #      can_node.append(int(N_D[_]))
                       #     break
                    #print(can_node)
                    #print(item.S)
                    for des in can_node:
                            # print('des', des)
                        tpstate = self.action(item, des)
                        if tpstate != None:
                            if tpstate.last == N[-1]:
                                self.candidateState.append(tpstate)
                            else:
                                returnstate.append(tpstate)
                #self.solutionpoll = returnstate.copy()
                print("number of state before elinimation1: {num_state}".format(num_state = len(returnstate)))
                #print(returnstate)
    #===================elimination rule: keep routes has larger capacity with identical node set visit before=============================
                #tempstate = self.eliminate(returnstate)
                #print("number of state after elinimation1: {num_state}".format(num_state=len(tempstate)))
                #returnstate = tempstate
                #for item in tempstate:
                 #   print(item.S)
    #===================end of elimination rule======================
                self.solutionpoll = returnstate
                state_num = len(self.solutionpoll)
                if state_num > Largest_state_num:
                    raise Exception("large state number")
                self.state_num = self.state_num + state_num
                return self.rundp(self.solutionpoll)
            else:
                print("======end iteration, candidate route:======")

                #for item in self.candidateState:
                 #   print(item.S)
                return None
        except:
            print("number of states too large")
    def eliminate(self, input):
        inputstate = input.copy()
        uniqueState = []
        #best_cap = []
        print("========start to elinimate state==========")
        for i in range(len(inputstate)):
            state1 = inputstate[i]
            s1 = state1.S
            de = False # whether s1 belong to one of uniqueState
            for j in range(len(uniqueState)):
                state2 = uniqueState[j]
                s2 = state2.S
                if (np.array_equal(np.sort(s1),np.sort(s2)) == True):
                    new_cap = cv[s1[1]]
                    old_cap = cv[s2[1]]
                    new_T = state1.T[-1]
                    old_T = state2.T[-1]
                    de = True
                    if(new_cap > old_cap and new_T < old_T):
                        del uniqueState[j]
                        uniqueState.append(state1)
                    break
            if (de == False):
                uniqueState.append(state1)
        return uniqueState

class dymaster:
    def __init__(self, input: list):
        self.a = [[] for i in range(num)]  # v,k,i
        self.route = [[] for i in range(num)]  # v,k all alternative routes for v
        self.route2 = [[] for i in range(num)] # all underlying routes for alternative routs for v
        self.route_num = [0 for i in range(num)]
        self.alternate = [[] for _ in range(num)]
        for item in input:
            temp = [0.0 for i in range(num)]
            self.route[item.S[1]].append(item.S)
            self.route2[item.S[1]].append(item.route)
            self.alternate[item.S[1]].append(item.alternative)
            for _ in N_e:
                if _ in item.S:
                    temp[_] = 1.0
            self.a[item.S[1]].append(temp)
        for i in range(num):
            self.route_num[i] = len(self.route[i])
        #for _ in Vh:
         #   print("vehicle {i} has {num} route generated".format(i = _, num = self.route_num[_]))
        self.mdl = Model(name='masterprob')
        self.maxlen = 0
        for item in self.a:
            lent = len(item)
            if lent > self.maxlen:
                self.maxlen = lent
        self.x_indices = [(v, k) for v in Vh for k in range(self.maxlen)]
        self.x = self.mdl.binary_var_dict(self.x_indices, name='x')
        self.y = self.mdl.binary_var_dict([i for i in N],name = 'y')
        self.result = [] # result of master problem
        self.result2 = [] #underlying routs for  result of master problem
        self.result3 = []  # alternative path for each underlying routes
        self.yresult = [sd] # result of super driver
        #with open('realcaseNetwork/route1_forcheck.pkl', 'wb') as f:
         #   pickle.dump(self.route, f)
        #f.close()
    def masterprob(self):
        wy = 500 #weight for super driver
        self.mdl.minimize(self.mdl.sum(self.x[v, k] for v in Vh for k in range(self.maxlen)) + wy*self.mdl.sum(self.y[i] for i in N))
        for i in N_e:
            #rn = 0 # check if there is a route visit this node
            temp = 0
            for v in Vh:
                for k in range(len(self.a[v])):
                    #rn += self.a[v][k][i]
                    temp = temp + self.a[v][k][i] * self.x[v, k]
            #if rn > 0.1:
                #print(rn)
            temp = temp +self.y[i]
            self.mdl.add_constraint(temp == 1.0)
            #print(temp)
        for v in Vh:
            self.mdl.add_constraint(self.mdl.sum(self.x[v, k] for k in range(len(self.a[v]))) <= 1.0)
        #self.mdl.export_as_lp("masterprob_model")

    def solve(self):
        solution = self.mdl.solve(log_output=True)
        print("objective value: " + str(solution.get_objective_value()))
        num_car = 0
        for v in Vh:
            for k in range(self.maxlen):
                if solution.get_value(self.x[v, k]) > 0.9:
                    self.result.append(self.route[v][k])
                    self.result2.append(self.route2[v][k])
                    self.result3.append(self.alternate[v][k])
                    num_car += 1
        for i in N:
            if solution.get_value(self.y[i]) > 0.9:
                self.yresult.append(i)
        #with open(f'solutions/case_{case_number}_solutions_dp.pkl', 'wb') as f:
         #   pickle.dump([self.result, dm, cv, N, N_D, N_S], f)
        #f.close()
        with open('realcaseNetwork/result_1.pkl', 'wb') as f:
            pickle.dump([self.result,self.result2], f)
            f.close()
        print("number of driver: ", len(N_1))
        print("number of flexible driver: ", len(N_2))
        print("number of passenger", len(N_3))
        print("number of car in use: {num}".format(num = num_car))
        print(self.result[1])
        print(self.result2[1])
        print(self.result3[1])
        print('========super driver=========')
        print(self.yresult)
        print('========number of people picked by super driver=========')
        print(len(self.yresult))
        num_altpath = [0 for i in range(scenario)]
        for route in self.result3:
            for link in route:
                num_altpath[link] += 1
        for i in range(len(num_altpath)):
            print("number of alternative path {i} :".format(i=i), num_altpath[i])
    def plotmap(self):
        for i in N_1:
            plt.plot(x_cord[i], y_cord[i], marker='o', color='r')
            #plt.text(x_cord[i] + 0.5, y_cord[i] + 0.5, "{i}({j})".format(i=i, j=int(N_D[i])))
        for i in N_2:
            plt.plot(x_cord[i], y_cord[i], marker='o', color='y')
            #plt.text(x_cord[i] + 0.5, y_cord[i] + 0.5, "{i}({j})".format(i=i, j=int(N_D[i])))
        for i in N_3:
            plt.plot(x_cord[i], y_cord[i], marker='o', color='g')
            #plt.text(x_cord[i] + 0.5, y_cord[i] + 0.5, "{i}({j})".format(i=i, j=int(N_D[i])))
        for i in N_S:
            plt.plot(x_cord[i], y_cord[i], marker='o', color='b')
            #plt.text(x_cord[i] + 0.5, y_cord[i] + 0.5, "{i}({j})".format(i=i, j=int(N_D[i])))
        for i in N_dummy:
            plt.plot(x_cord[i], y_cord[i], marker='o', color='k')
            #plt.text(x_cord[i] + 0.5, y_cord[i] + 0.5, "{i}({j})".format(i=i, j=int(N_D[i])))
        plt.plot(x_cord[sd], y_cord[sd], marker='*', color='m')
        #plt.text(x_cord[sd] + 0.5, y_cord[sd] + 0.5, str(sd))
    def plotroute(self):
        for item in self.result:
            r = random.random()
            b = random.random()
            g = random.random()
            #color = (r, g, b)
            xcordset = []
            ycordset = []
            for i in range(1,len(item)-1):
                xcordset.append(x_cord[item[i]])
                ycordset.append(y_cord[item[i]])
            plt.plot(xcordset,ycordset, c = 'black')
        sdx = [] # super driver routes x coordinate
        sdy = [] # super driver routes y coordinate
        for i in self.yresult:
            sdx.append(x_cord[i])
            sdy.append(y_cord[i])
        plt.plot(sdx,sdy, c = 'm')
        plt.show()
    def plotaltroute(self):
        for i in range(0, len(self.result)):
            color = (random.random(), random.random(), random.random())
            for j in range(1, len(self.result[i]) - 2):
                node1 = self.result[i][j]
                node2 = self.result[i][j+1]
                if self.result3[i][j-1] == 0:
                    plt.plot(np.array([x_cord[node1],x_cord[node2]]),np.array([y_cord[node1],y_cord[node2]]), c = color, ls = 'solid' )
                else:
                    plt.plot(np.array([x_cord[node1],x_cord[node2]]),np.array([y_cord[node1],y_cord[node2]]), c = color, ls ='dotted' )
        plt.show()
    def endmodel(self):
        self.mdl.end()


class visualize:
    def __int__(self):
        self.carcl = np.array([None for _ in range(num)], dtype='object')
        for _ in Vh:
            r = random.random()
            b = random.random()
            g = random.random()
            self.carcl[_] = (r, g, b)

    def shownetwork(self):
        for i in N_1:
            plt.plot(x_cord[i], y_cord[i], marker='o', color='r')
            plt.text(x_cord[i] + 0.5, y_cord[i] + 0.5, "{i}({j})".format(i=i, j=int(N_D[i])))
        for i in N_2:
            plt.plot(x_cord[i], y_cord[i], marker='o', color='y')
            plt.text(x_cord[i] + 0.5, y_cord[i] + 0.5, "{i}({j})".format(i=i, j=int(N_D[i])))
        for i in N_3:
            plt.plot(x_cord[i], y_cord[i], marker='o', color='g')
            plt.text(x_cord[i] + 0.5, y_cord[i] + 0.5, "{i}({j})".format(i=i, j=int(N_D[i])))
        for i in N_S:
            plt.plot(x_cord[i], y_cord[i], marker='o', color='b')
            plt.text(x_cord[i] + 0.5, y_cord[i] + 0.5, "{i}({j})".format(i=i, j=int(N_D[i])))
        for i in N_dummy:
            plt.plot(x_cord[i], y_cord[i], marker='o', color='k')
            plt.text(x_cord[i] + 0.5, y_cord[i] + 0.5, "{i}({j})".format(i=i, j=int(N_D[i])))
        plt.plot(x_cord[sd], y_cord[sd], marker='*', color='m')
        plt.text(x_cord[sd] + 0.5, y_cord[sd] + 0.5, str(sd))

    def showroutes(self):
        for k in Vh:
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            for i in N_0:
                for j in N_0:
                    for m in sc_set:
                        if j != i and x_var[i, j, k, m] == 1:
                            x = [x_cord[i], x_cord[j]]
                            y = [y_cord[i], y_cord[j]]
                            plt.plot(x, y, c=color)
                            print(f'route {i} to {j} for vehicle {k}')


    def show(self):
        plt.show()


def main(triger: int):
    # ============run model on cplex==============
    if triger == 0:
        exp = model()
        exp.createmodel()
        exp.solve()
        # vis = visualize()
        # vis.shownetwork()
        # vis.showroutes()
        # vis.show()
    # ============run model on dynamic programming==========
    elif triger == 1:
        print("start dynamic programming")
        ts = dynapro()
        start = time.time()
        ts.rundp(ts.solutionpoll)
        print("generate {state_num} states totally".format(state_num = ts.state_num))
        end = time.time()
        ite = 0
        for item in ts.candidateState:
            #print(ite)
            ite += 1
        dyma = dymaster(ts.candidateState)
        dyma.masterprob()
        dyma.solve()
        dyma.plotmap()
        dyma.plotaltroute()
        print('time for generating routes: ', (end - start) * 10 ** 3, "ms")
        dyma.endmodel()


if __name__ == "__main__":
    print('test')
    main(1)
    #vis = visualize()
    #vis.shownetwork()
    #vis.showroutes()
    #vis.show()

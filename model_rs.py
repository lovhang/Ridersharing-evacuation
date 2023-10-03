from docplex.mp.model import Model
from docplex.mp import conflict_refiner
from docplex.mp.relaxer import Relaxer
import numpy as np
import matplotlib.pyplot as plt
import random as random
import pickle
import time

# read network from pkl and transfer it to t_ij^m
num = 21
sd = num - 2  # super driver, num-1 is n+1, num-2 is super driver location
scenario = 3
sc_set = [i for i in range(scenario)]
Xiaohang = 0
if Xiaohang == 1:
    with open('network/linknetwork1.pkl', 'rb') as f:
        x_cord, y_cord, dist, connect, ttlink = pickle.load(f)
    f.close()
    # scenario = 3
    ttm = np.array([None for i in range(3)])
    spm = np.array([None for i in range(3)])
    for i in sc_set:
        with open(f'network/linknetwork1_{i}_SPMatrix.pkl', 'rb') as f:
            spm[i], ttm[i] = pickle.load(f)
        f.close()

    bigM = 10000  # set bigM to the ij that do not have alternative path so that x_ij^vm no going to select this path
    tt = np.array([[[bigM for k in sc_set] for j in range(num)] for i in range(num)])
    sp = np.array([[[None for k in sc_set] for j in range(num)] for i in range(num)])
    for i in range(1, num - 1):
        for j in range(1, num - 1):
            if j != i:
                for k in sc_set:
                    tt[i][j][k] = ttm[k][i][j][0]
                    sp[i][j][k] = spm[k][i][j]
else:
    with open('network/linknetwork1.pkl', 'rb') as f:
        x_cord, y_cord, dis, con, tvt = pickle.load(f)
    f.close()
    with open(f'network/linknetwork1_SPMatrixAlt.pkl', 'rb') as f:
        sp, tt, tvt, SceEdges, ScenTimes = pickle.load(f)
    f.close()
    for i in range(0, num):
        for k in sc_set:
            tt[0][i][k] = 1.0  # Zero node to all nodes for all scenarios
            tt[i][0][k] = 1.0  # All nodes to 0 for all scenarios
            tt[sd][i][k] = 1.0  # superdriver to all nodes for all scenarios
            tt[i][sd][k] = 1.0  # all nodes to superdriver for all scenarios
            tt[i][i][k] = 1.0  # node to itself for all scenarios

# =============== Case study Generation =======================
# set of all case studies of N
N_cases = [np.array([0, 1, 4, 5, 6, 8, 9, 10, 16, sd, 20]),
           np.array([0, 1, 2, 7, 8, 10, 13, 17, 18, sd, 20]),
           np.array([0, 1, 3, 4, 6, 8, 9, 10, 14, 15, 17, sd, 20]),
           np.array([0, 1, 4, 5, 6, 7, 8, 9, 13, 12, 15, 16, 18, sd, 20])]


case_count = len(N_cases)

N_1_cases = [np.array([5, 16]),
             np.array([2, 7, 10]),
             np.array([4, 6, 9]),
             np.array([5, 6, 7, 12, 15])]

N_2_cases = [np.array([4, 6]),
             np.array([13]),
             np.array([14, 17]),
             np.array([9, 13, 18])]

N_3_cases = [np.array([9, 10]),
             np.array([18, 17]),
             np.array([3, 10, 15]),
             np.array([4, 16])]  # Carless evacuees

N_S = np.array([1, 8])  # Shelters nodes

N_0_cases = [N_cases[i][1:-1] for i in range(case_count)]  # N1, N2, N3

# Destinations shelter for nodes
N_D_cases = [np.array([0 for _ in range(num)]) for _ in range(case_count)]
for case in range(case_count):
    for node in range(num):
        if node not in (0, 1, 8, sd, num-1):
            N_D_cases[case][node] = random.choice(N_S)
        if node in (1, 8):
            N_D_cases[case][node] = num-1

vh_cases = [np.union1d(N_1_cases[i], np.union1d(N_2_cases[i], sd)) for i in range(case_count)]  # N_1 + N_2

# Dummy nodes 0 and N
N_dummy = np.array([0, num - 1])

# Evacuees list
Ne_cases = [np.union1d(N_1_cases[i], np.union1d(N_2_cases[i], N_3_cases[i])) for i in range(case_count)]

# Demand or passenger counts of nodes
dm_cases = [np.array([0 for _ in range(num)]) for _ in range(case_count)]
dm_list = [1, 2, 3]
random.seed(124)
for case in range(case_count):
    for node in range(num):
        if node not in (0, 1, 8, sd, 20):
            dm_cases[case][node] = random.choice(dm_list)

# Capacity of nodes (for drivers)
cv_cases = [np.array([0 for _ in range(num)]) for _ in range(case_count)]
cv_list = [2, 3, 5, 4]
random.seed(123)
for case in range(case_count):
    for node in range(num):
        if node in np.union1d(N_1_cases[case], N_2_cases[case]):
            cv_cases[case][node] = random.choice(cv_list)
        if node == sd:
            cv_cases[case][node] = 100

# Planning Time horizon (window)
T_end_cases = [1000, 1400, 1000, 800]
# =============== End of Case study Generation =======================

# ===== change case number from 1 to case_count to run codes for different case studies ====
case_number = 3

N = N_cases[case_number]  # 0,n+1,N_1,N_2,N_3,N_S, super
N_0 = N_0_cases[case_number]  # N_1,N_2,N_3
N_1 = N_1_cases[case_number]  # N_1 driver
N_2 = N_2_cases[case_number]  # N_2 : flexible driver
N_3 = N_3_cases[case_number]  # N_3 evacuee
N_D = N_D_cases[case_number]
Vh = vh_cases[case_number]  # Driver index (N_1,N_2,N_3,super)
N_e = Ne_cases[case_number]  # evacuee list, including N_1,N_2,N_3
dm = dm_cases[case_number]
cv = cv_cases[case_number]
T_end = T_end_cases[case_number]
ScenTimes.append(T_end)

# Tolerance
LDP = np.array([T_end for i in range(0, num)])  # latest departure time, the time location is affected
EDP = np.array([0 for i in range(0, num)])  # Earliest departure time

DT = np.array([T_end for i in range(0, num)])  # Detour time
DT[2] = 130
DT[3] = 150
DT[10] = 280
DT[11] = 180
DT[13] = 180
DT[16] = 180
DT[19] = 10000
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
            with open(f'solutions/case_{case_number}_solutions.pkl', 'wb') as f:
                pickle.dump([x_var, y_var, r_var, sol_time], f)
            f.close()
            self.mdl.end()


class state:
    def __init__(self, S, R, T, dt, Q, route, last):
        # print(input)
        # self.v = input
        self.S = S  # Set of included node
        self.R = R  # Set of evacuee node haven't been delivered
        self.T = T  # [max(tt[0][self.v][0], EDP[self.v])] # Set of leaving time
        self.dt = dt  # Set of maximum detour time for i in R
        self.Q = Q  # [dm[self.v]] # Set of leaving number of people
        self.route = route
        self.last = last
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
            print(et)
            T = [et]
            dt = [et + DT[_]]
            Q = [dm[_]]
            route = [[N[0], _]]
            last = _
            s1 = state(S, R, T, dt, Q, route, last)
            self.solutionpoll.append(s1)

    def action(self, inputstate: state, des: int):
        S = inputstate.S.copy()
        R = inputstate.R.copy()
        T = inputstate.T.copy()
        dt = inputstate.dt.copy()
        Q = inputstate.Q.copy()
        route = inputstate.route.copy()
        l = inputstate.last
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
                    for i in range(len(tt[l][des])):
                        T_j = T[-1] + tt[l][des][i]  #
                        Q_j = Q[-1]
                        dtmin = min(dt)
                        if T_j <= min(ScenTimes[i], LDP[des], dtmin):
                            R = []
                            dt = []
                            Q_j = 0
                            S.append(des)
                            S.append(N[-1])
                            T.append(T_j)
                            Q.append(Q_j)
                            route.append(sp[l][des][i])
                            l = N[-1]
                            return state(S, R, T, dt, Q, route, l)
                    return None
            else:
                for i in range(len(tt[l][des])):  # if i is the destination of evacuees
                    T_j = T[-1] + tt[l][des][i]  #
                    Q_j = Q[-1]
                    dtmin = min(dt)
                    if T_j <= min(ScenTimes[i], LDP[des], dtmin):
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
                        return state(S, R, T, dt, Q, route, l)
                return None
        if des in N_e:  # if des is node of evacuees
            try:
                if des in S:
                    raise ValueError("demand node {i} already visited before".format(i=des))
                else:
                    for i in range(len(tt[l][des])):
                        T_j = max(EDP[des], T[-1] + tt[l][des][i])  #
                        Q_j = Q[-1] + dm[des]
                        dtmin = min(dt)
                        if T_j <= min(ScenTimes[i], LDP[des], dtmin) and Q_j <= cv[S[1]]:
                            R.append(des)
                            S.append(des)
                            T.append(T_j)
                            dt.append(T_j + DT[des])
                            Q.append(Q_j)
                            route.append(sp[l][des][i])
                            l = des
                            return state(S, R, T, dt, Q, route, l)
                    return None
            except ValueError as e:
                print(e)
        else:
            print('des not in N_e and N_S')
            return False

    def rundp(self, input: list):
        tempslopol = input
        returnstate = []
        if len(tempslopol) >= 1:
            self.ite += 1
            # print(self.ite)
            for item in tempslopol:
                # print('S before', item.S)
                # print('last', item.last)

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
                        # print('des', des)
                    tpstate = self.action(item, des)
                    if tpstate != None:
                        if tpstate.last == N[-1]:
                            self.candidateState.append(tpstate)
                        else:
                            returnstate.append(tpstate)
            self.solutionpoll = returnstate.copy()
            print('====iteration==== {i}'.format(i = self.ite))
            print(self.solutionpoll)
            return self.rundp(self.solutionpoll)
        else:
            # print("end iteration")
            return None


class dymaster:
    def __init__(self, input: list):
        self.a = [[] for i in range(num)]  # v,k,i
        self.route = [[] for i in range(num)]  # v,k
        self.route2 = [[] for i in range(num)]
        print('input')
        for item in input:
            print(item.S)
            temp = [0.0 for i in range(num)]
            self.route[item.S[1]].append(item.S)
            self.route2[item.S[1]].append(item.route)
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
        self.x_indices = [(v, k) for v in Vh for k in range(self.maxlen)]
        self.x = self.mdl.binary_var_dict(self.x_indices, name='x')
        self.y = self.mdl.binary_var(name = 'y')
        self.result = []
        self.result2 = []

    def masterprob(self):
        wy = 500 #weight for super driver
        self.mdl.minimize(self.mdl.sum(self.x[v, k] for v in Vh for k in range(self.maxlen)) + wy*self.y)
        for i in N_e:
            rn = 0  # check if there is a route visit this node
            temp = 0
            for v in Vh:
                for k in range(len(self.a[v])):
                    rn += self.a[v][k][i]
                    temp = temp + self.a[v][k][i] * self.x[v, k]
            if rn > 0.1:
                print(rn)
                temp = temp + self.y
                self.mdl.add_constraint(temp == 1.0)
                print(temp)
        for v in Vh:
            self.mdl.add_constraint(self.mdl.sum(self.x[v, k] for k in range(len(self.a[v]))) <= 1.0)
        self.mdl.export_as_lp("masterprob_model")

    def solve(self):
        solution = self.mdl.solve(log_output=True)
        print("objective value: " + str(solution.get_objective_value()))
        for v in Vh:
            for k in range(self.maxlen):
                if solution.get_value(self.x[v, k]) > 0.9:
                    self.result.append(self.route[v][k])
                    self.result2.append(self.route2[v][k])
        with open(f'solutions/case_{case_number}_solutions_dp.pkl', 'wb') as f:
            pickle.dump([self.result, dm, cv, N, N_D, N_S], f)
        f.close()
        print(self.result)


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
        print(ts.solutionpoll)
        start = time.time()
        ts.rundp(ts.solutionpoll)
        end = time.time()
        ite = 0
        for item in ts.candidateState:
            print(ite)
            ite += 1
            print(item.S)
            # print(item.T)
            # print(item.route)
        # print(len(ts.candidateState))
        dyma = dymaster(ts.candidateState)
        dyma.masterprob()
        dyma.solve()
        print('time for generating routes: ', (end - start) * 10 ** 3, "ms")


if __name__ == "__main__":
    main(1)
'''    vis = visualize()
    vis.shownetwork()
    vis.showroutes()
    vis.show()'''

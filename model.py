import itertools

from docplex.mp.model import Model
from docplex.mp.conflict_refiner import ConflictRefiner
from docplex.mp.relaxer import Relaxer
import numpy as np
import math
import matplotlib.pyplot as plt
import random as random
import pickle


#read network from pkl
address = "network/network2_s.pkl"
with open(address, 'rb') as f:
    num, v_num, qnum, Vh, N, Na, Nd, Np, Ns, Ok, x_cord, y_cord, qs, sq, eq, tm, dp, cv, Y0, pr, dm, dis, tt = pickle.load(f)
print(num)
#print(cv)
#print(dm)
#print(dis)
#print(tt)
#manually change parameter for tweak########################
tm = [150,180,150]
#Parameter for modeling#####################################
max_num = 10000
max_dist = 1000
w1 = 10000
w2 = 0.01
#Varible that store result of decision variables
x_var = np.zeros((num,num,v_num))
y_var = np.zeros((num,v_num))
z_var = np.zeros((num,v_num,qnum))
r_var = np.zeros((num,v_num))

class model:
    def __init__(self):
        self.mdl = Model(name='Evacuation')
        print(type(self.mdl))
        self.x_indices = [(i,j,k) for i in N for j in N for k in Vh if i!=j]
        self.x = self.mdl.binary_var_dict(self.x_indices, name='x')
        self.yr_indices = [(i,k) for i in N for k in Vh]
        self.y = self.mdl.continuous_var_dict(self.yr_indices, name='y')
        self.z_indices = [(i,k,q) for i in N for k in Vh for q in qs]
        self.z = self.mdl.integer_var_dict(self.z_indices, name='z')
        self.r = self.mdl.continuous_var_dict(self.yr_indices, name='r')
    def createmodel(self):
        self.mdl.maximize(w1*self.mdl.sum(pr[i]*self.z[i,k,q]*eq[q] for i in Np for k in Vh for q in qs) - w2*self.mdl.sum(self.r[i,k]+self.y[i,k] for i in N for k in Vh))
        #self.mdl.maximize(self.mdl.sum(self.x[i,j,k]*tt[i][j] for i in N for j in N for k in Vh if i!=j))
        for _ in Vh:
            self.mdl.add_constraint(self.x[0,Ok[_],_] == 1)
            self.mdl.add_constraint(self.mdl.sum(self.x[j,N[-1],_] for j in Nd) == 1)
        for j in Ns:
            for _ in Vh:
                self.mdl.add_constraint(self.mdl.sum(self.x[i,j,_] for i in N if i != j) - self.mdl.sum(self.x[j,k,_] for k in N if k!=j) ==0)
        for i in N:
            for j in N:
                if j!=i:
                    for _ in Vh:
                        self.mdl.add_constraint(self.r[j,_] - self.r[i,_]-self.mdl.sum(self.z[j,_,k]*sq[k] for k in qs) - max_num*self.x[i,j,_] >= tt[i][j] -max_num)
        for i in N:
            for j in N:
                if j!=i:
                    for _ in Vh:
                        self.mdl.add_constraint(self.y[j,_] - self.y[i,_]-self.mdl.sum(self.z[j,_,k]*eq[k] for k in qs) - max_num*self.x[i,j,_] >= -max_num)
        for i in N:
            for _ in Vh:
                self.mdl.add_constraint(self.mdl.sum(self.z[i,_,q] for q in qs)-max_num*self.mdl.sum(self.x[i,j,_] for j in N if j != i) <=0)
        for i in N:
            for q in qs:
                self.mdl.add_constraint(self.mdl.sum(self.z[i,k,q] for k in Vh) <= dm[i][q])
        for _ in Vh:
            self.mdl.add_constraint(self.r[N[-1],_] <= tm[_])
            self.mdl.add_constraint(self.y[N[-1],_] <= cv[_])
            self.mdl.add_constraint(self.r[0,_] == dp[_])
            self.mdl.add_constraint(self.y[0,_] == Y0[_])
        for _ in Vh:
            self.mdl.add_constraint(self.mdl.sum(self.x[j,0,_] for j in N if j!=0) == 0)
            self.mdl.add_constraint(self.mdl.sum(self.x[0,j,_] for j in N if j!=0) == 1)
            self.mdl.add_constraint(self.mdl.sum(self.x[N[-1],j,_] for j in N if j!=N[-1]) == 0)
            self.mdl.add_constraint(self.mdl.sum(self.x[j,N[-1],_] for j in N if j!=N[-1]) == 1)
        #self.mdl.add_constraint(self.z[4,1,2] == 1)
        self.mdl.export_as_lp("evacuation_model")
    def solvep(self, soladdress:str):
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
            return False
        else:
            print("Model solved")
            print("objective value: "+ str(solution.get_objective_value()))
            for i,j,k in self.x_indices:
                if solution.get_value(self.x[i,j,k])>0.9:
                    x_var[i][j][k] = 1.0
                    print(f'Value of x[{i},{j},{k}] = 1')
            for i,k in self.yr_indices:
                ytempvar = solution.get_value(self.y[i,k])
                if ytempvar>0.1:
                    y_var[i][k] = round(ytempvar)
                    print(f'Value of y[{i},{k}]:', ytempvar)
            for i,k,q in self.z_indices:
                ztempvar = solution.get_value(self.z[i,k,q])
                if ztempvar >0.1:
                    z_var[i][k][q] = round(ztempvar)
                    print(f'Value of z[{i},{k},{q}]:', ztempvar)
            for i,k in self.yr_indices:
                rtempvar = solution.get_value(self.r[i,k])
                r_var[i][k] = round(rtempvar,2)
                print(f'Value of r[{i},{k}]:', rtempvar)
            with open(soladdress,'wb') as f:
                pickle.dump([x_var,y_var,z_var,r_var],f)
            self.mdl.end()
            return True
    def endp(self):
        self.mdl.end()

class visualize():
    def __init__(self):
        color = np.array(["blue" for _ in range(num)], dtype = 'object')
        self.carcl = np.array([None for _ in Vh], dtype = 'object')
        mk = np.array(["o" for _ in range(num)], dtype = 'object' )
        color[Nd] = "green"
        color[Np] = "red"
        color[0] = "grey"
        color[N[-1]] = "grey"
        mk[Np] = "^"
        mk[Nd] = "s"
        for _ in Vh:
            r = random.random()
            b = random.random()
            g = random.random()
            self.carcl[_] = (r, g, b)
        #print(self.carcl)
        for i in N:
            plt.plot(x_cord[i],y_cord[i],marker = mk[i], c = color[i])
            plt.text(x_cord[i],y_cord[i]+0.2, i)
    def adddist(self):
        for i in Ns:
            for j in Ns:
                if i != j and tt[i][j]<max_num-1 :
                    x = [x_cord[i],x_cord[j]]
                    y = [y_cord[i],y_cord[j]]
                    m = [(x_cord[i]+x_cord[j])/2, (y_cord[i]+y_cord[j])/2]
                    plt.plot(x,y)
                    plt.text(m[0],m[1],tt[i][j])
    def plotroutes(self):
        for k in Vh:
            for i in Ns:
                for q in qs:
                    if z_var[i,k,q] >0.1:
                        plt.text(x_cord[i],y_cord[i]+0.8*(q+1), f'z{k,q}={z_var[i,k,q]}')
                for j in Ns:
                    if j != i and x_var[i,j,k] == 1:
                        x = [x_cord[i], x_cord[j]]
                        y = [y_cord[i], y_cord[j]]
                        plt.plot(x,y, c = self.carcl[k])
    def plotmore(self):
        for k in Vh:
            for i in Ns:
                if y_var[i][k]>0.1:
                    plt.text(x_cord[i],y_cord[i]+0.6*(k+1), f'y{i,k}={y_var[i][k]}')
                if r_var[i][k]>0.1:
                    plt.text(x_cord[i],y_cord[i]-0.6*(k+1), f'r{i,k}={r_var[i][k]}')
    def show(self):
        plt.show()



if __name__ == '__main__':
    print("start model")
    ex = model()
    ex.createmodel()
    if ex.solvep("solution/nw2_1.pkl") == True:
        vslz1 = visualize()
        vslz1.plotroutes()
        #vslz1.adddist()
        vslz1.show()
    #vslz2 = visualize()
    #vslz2.show()
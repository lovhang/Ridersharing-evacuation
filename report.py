import numpy as np
import matplotlib.pyplot as plt
import random as random
import pickle


with open('network/linknetwork1.pkl', 'rb') as f:
    x_cord, y_cord, dis, con, tvt = pickle.load(f)
f.close()

with open(f'network/linknetwork1_SPMatrixAlt.pkl', 'rb') as f:
    sp, tt, tvt, SceEdges, ScenTimes = pickle.load(f)
f.close()
case_number = 3
with open(f'solutions/case_{case_number}_solutions_dp.pkl', 'rb') as f:
    solutions, dm, cv, N, N_D, N_S, EDT = pickle.load(f)
f.close()
num = len(cv)


#  Calculating the cumulative number of passengers in each vehicle
passengerCount = []
elapsedTime = []
earliestDepart = []
waitingTime = []
for sol in solutions:

    # to save the total passengers in a vehicle (solution)
    passCounts = []
    passCount = 0

    # to save the elapsed time over nodes of one vehicle (solution)
    passTimes = [0 for _ in range(len(sol))]
    passTime = 0

    # To save the waiting time of vehicle (solution) at each node

    for i in range(1, len(sol)):
        # Total number of passengers in the vehicle at each node
        passCount += dm[sol[i-1]]
        passCounts.append(passCount)

        # Time the vehicle reaches the next node
        # not considering different scenarios for now
        passTime += tt[sol[i-1]][sol[i]][0]
        passTimes[i] = passTime

    # For each solution, generate an array showing the EDT of each node
    departTime = [EDT[index] for index in sol]

    # calculate Waiting Time
    waitTime = [0 for _ in sol]
    for index in range(len(sol)):
        wait = departTime[index] - passTimes[index]
        if wait > 0:
            waitTime[index] = wait

    earliestDepart.append(departTime)
    elapsedTime.append(passTimes)
    waitingTime.append(waitTime)
    passengerCount.append(passCounts)
'''for i in range(len(solutions)):
    print(earliestDepart[i], elapsedTime[i], waitingTime[i])'''
fig_size = 15
lWidth = 0.5 * fig_size
lWidthDash = 0.2 * fig_size

class visualize:
    def __int__(self):
        pass
    def shownetwork(self):
        plt.figure(figsize=(fig_size, fig_size), dpi=200)
        ax = plt.axes(projection='3d')
        for i in N:
            if i in N_S:
                ax.plot3D(x_cord[i], y_cord[i], 0, marker='D', color='gray', markersize=6)
                ax.text(x_cord[i] + 0.5, y_cord[i] + 0.5, 0, f"{i}")
            elif i == num-2:
                ax.plot3D(x_cord[i], y_cord[i], 0, marker='*', color='gray', markersize=6)
                ax.text(x_cord[i] + 0.5, y_cord[i] + 0.5, 0, f"{i})")
            else:
                ax.plot3D(x_cord[i], y_cord[i], 0, marker='o', color='gray', markersize=6)
                ax.text(x_cord[i] + 0.5, y_cord[i] + 0.5, 0, f"{i}")

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Time')
        self.ax = ax
    def showroutes(self):
        # Only plot a few vehicles
        visits = (0, 4)
        ax = self.ax
        for solI in visits:
            color = (random.random(), random.random(), random.random())

            for j in range(len(solutions[solI]) - 1):
                # The cordinations here are intended for plotting the routes.
                x = [x_cord[solutions[solI][j]], x_cord[solutions[solI][j + 1]]]
                y = [y_cord[solutions[solI][j]], y_cord[solutions[solI][j + 1]]]

                # Z values are considering the waiting time
                reachZ = elapsedTime[solI][j]
                startZ = reachZ + waitingTime[solI][j]
                endZ = elapsedTime[solI][j + 1]
                z = [startZ, endZ]


                if j == 0 or j == len(solutions[solI])-2:   # Plot the line with dashed style
                    # Set label for the vehicle and plot the dummy node
                    if j == len(solutions[solI])-2:
                        ax.plot3D(x, y, z, c=color, linewidth=lWidthDash, alpha=0.4, linestyle=':',
                                  label=f'Vehicle {solutions[solI][1]}')
                        ax.plot3D(x[1], y[1], z[1], 'rD', markersize=lWidth)

                    else:  # Plot the dashed line from 0 node to the vehicle
                        ax.plot3D(x, y, z, c=color, linewidth=lWidthDash, alpha=0.4, linestyle=':')
                    ax.plot3D(x, y, 0, c=color, alpha=0.3)

                # Plot the solid line for the normal trip route
                else:
                    ax.plot3D(x, y, z, c=color, linewidth=lWidth, alpha=0.4)  # Normal 3D Route
                    # Waiting 3D Route
                    ax.plot3D([x[0], x[0]], [y[0], y[0]], [reachZ, startZ],
                              c='red', linewidth=lWidth, alpha=0.4)
                    ax.plot3D(x, y, 0, c=color, alpha=0.3)  # 2D route

                # Generate the incremental passengers size marker
                z_rate = 0
                for _ in range(passengerCount[solI][j]):
                    ax.plot3D(x_cord[solutions[solI][j]], y_cord[solutions[solI][j]], z[0] + 6 * z_rate,
                              marker='^', color=color, markersize=fig_size, markeredgecolor='black')
                    z_rate += 0.5

    def showplot(self):
        plt.legend()
        plt.savefig(f'figures/Case_{case_number}_3D_plot.jpg', bbox_inches='tight')
        plt.show()




if __name__ == '__main__':
    vis = visualize()
    vis.shownetwork()
    vis.showroutes()
    vis.showplot()

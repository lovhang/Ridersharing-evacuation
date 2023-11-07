import numpy as np
import matplotlib.pyplot as plt
import random as random
import pickle

colors = ['#4c8a04', '#047a8a', '#9d12b3', '#b6bf08', '#1a08bf', '#5817ad']

class Visualize:
    def __init__(self, solution_file: str, matrix_file: str, condition: str):
        self.case_number = 3

        if condition == 'With Malfunction':
            with open('network/malfunction_road_info.pkl', 'rb') as handle:
                self.scen_edge, self.scen_time = pickle.load(handle)
            handle.close()

        with open('network/linknetwork1.pkl', 'rb') as f:
            self.x_cord, self.y_cord, dis, con, tvt = pickle.load(f)
        f.close()

        with open(matrix_file, 'rb') as f:
            self.sp, self.tt, self.tvt = pickle.load(f)
        f.close()

        with open(solution_file, 'rb') as f:
            self.solutions, self.dm, self.cv, self.N, self.N_1, self.N_2, self.N_3, self.N_S, self.EDT = pickle.load(f)
        f.close()

        # First generate the whole paths between nodes of each solution
        # example: solution is [0, 3, 16, 1, 20]
        # whole path: [0, 3, 16, 12, 9, 1, 20], This can be plotted on the network

        self.detailed_solutions = []
        for solI in self.solutions:
            path = []
            for j in range(1, len(solI) - 2):
                for node in self.sp[solI[j]][solI[j + 1]][-1]:
                    if len(path) >= 1:
                        if node != path[-1]:
                            path.append(node)
                    else:
                        path.append(node)
            path.append(solI[-1])
            self.detailed_solutions.append(path)
        print(self.detailed_solutions)
        #  Calculating the cumulative number of passengers in each vehicle
        self.passengerCount = []
        self.elapsedTime = []
        self.earliestDepart = []
        self.waitingTime = []
        self.travelTime = []
        for solution_index in range(len(self.detailed_solutions)):
            sol = self.detailed_solutions[solution_index]
            # to save the total passengers in a vehicle (solution)
            passCounts = []
            passCount = 0

            # to save the elapsed time over nodes of one vehicle (solution)
            passTimes = [0 for _ in sol]
            passTime = 0

            # To save the waiting time of vehicle (solution) at each node

            # For each solution, generate an array showing the EDT of each node
            departTimes = [self.EDT[node] for node in sol]

            # calculate Waiting Time
            waitTime = [0 for _ in sol]

            # travel time
            travelTime = [0 for _ in sol]
            for i in range(len(sol)):
                # Total number of passengers in the vehicle at each node
                if sol[i] in self.solutions[solution_index]:
                    passCount += self.dm[sol[i]]
                    passCounts.append(passCount)
                else:
                    passCounts.append(passCount)

                # Time the vehicle reaches the next node
                # not considering different scenarios for now
                # calculate waiting for node in self.solution, not all
                if sol[i] in self.solutions[solution_index]:
                    if i == 0:
                        passTime += departTimes[i]
                        wait = 0
                    elif i == len(sol)-1:
                        wait = 0
                    else:
                        travelTime[i] = self.tt[sol[i - 1]][sol[i]][-1]
                        passTime += travelTime[i] + wait
                        wait = max(departTimes[i] - passTime, 0)
                else:
                    travelTime[i] = self.tt[sol[i - 1]][sol[i]][-1]
                    passTime += travelTime[i] + wait
                    wait = 0

                passTimes[i] = passTime
                waitTime[i] = wait

            self.earliestDepart.append(departTimes)
            self.elapsedTime.append(passTimes)
            self.waitingTime.append(waitTime)
            self.passengerCount.append(passCounts)
            self.travelTime.append(travelTime)

        self.fig_size = 15
        self.lWidth = 0.3 * self.fig_size
        self.lWidthDash = 0.1 * self.fig_size
        self.condition = condition
    def shownetwork(self):
        plt.figure(figsize=(self.fig_size, self.fig_size), dpi=200)
        ax = plt.axes(projection='3d')
        ax.view_init(20, -30)
        labels = ['Shelter', 'Super Driver', 'Driver', 'Flexible Driver', 'Carless Evacuee', '']
        counts = [0, 0, 0, 0, 0, 0]
        for i in self.N:
            if i in self.N_S:
                marker = 'D'
                node_color = 'gray'
                indicator = 0
            elif i == self.N[-2]:
                marker = '*'
                node_color = 'black'
                indicator = 1
            elif i in self.N_1:
                marker = 'o'
                node_color = 'green'
                indicator = 2
            elif i in self.N_2:
                marker = 'o'
                node_color = 'lightgreen'
                indicator = 3
            elif i in self.N_3:
                marker = 'o'
                node_color = 'orange'
                indicator = 4
            else:
                marker = 'o'
                node_color = 'black'
                indicator = 5

            if counts[indicator] == 0:
                ax.plot3D(self.x_cord[i], self.y_cord[i], 0,
                          marker=marker, color=node_color, markersize=6, label=labels[indicator])
            else:
                ax.plot3D(self.x_cord[i], self.y_cord[i], 0,
                          marker=marker, color=node_color, markersize=6)
            counts[indicator] += 1
            ax.text(self.x_cord[i] + 0.5, self.y_cord[i] + 0.5, 0, f"{i}")

        ax.set_title(f'Ride Sharing {self.condition}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Time')
        ax.set_zlim([0, 500])
        ax.legend()
        self.ax = ax
    def showroutes(self):
        ax = self.ax
        alpha = 1
        for solI in range(2):  # plot only a few solutions
            color = colors[solI]
            solution = self.detailed_solutions[solI]
            for j in range(len(solution)-1):
                # The cordinations here are intended for plotting the routes.
                x = [self.x_cord[solution[j]], self.x_cord[solution[j + 1]]]
                y = [self.y_cord[solution[j]], self.y_cord[solution[j + 1]]]

                # Z values are considering the waiting time
                reachZ = self.elapsedTime[solI][j]
                startZ = reachZ + self.waitingTime[solI][j]
                endZ = self.elapsedTime[solI][j + 1]
                z = [startZ, endZ]

                if j == len(solution)-2:
                    ax.plot3D(x, y, z, c=color, linewidth=self.lWidthDash, alpha=alpha, linestyle=':',
                              label=f'Vehicle {solution[0]}')
                    ax.plot3D(x[1], y[1], z[1], 'rD', markersize=self.lWidth)

                # Plot the solid line for the normal trip route
                else:
                    ax.plot3D(x, y, z, c=color, linewidth=self.lWidth, alpha=alpha)  # Normal 3D Route
                    # Waiting 3D Route
                    ax.plot3D([x[0], x[0]], [y[0], y[0]], [reachZ, startZ],
                              c='red', linewidth=self.lWidth, alpha=0.5*alpha)
                    ax.plot3D(x, y, 0, c=color, alpha=0.6*alpha)  # 2D route

                # Generate the incremental passengers size marker
                z_rate = 0
                if solution[j] in self.solutions[solI]:
                    for _ in range(self.passengerCount[solI][j]):
                        # add markers only for nodes that are in self.solution
                        ax.plot3D(self.x_cord[solution[j]], self.y_cord[solution[j]], z[0] + 6 * z_rate,
                                  marker='^', color=color, markersize=self.fig_size, markeredgecolor='black')
                        z_rate += 0.5

    def showmalfunction(self):
        ax = self.ax
        for index in range(len(self.scen_edge)):
            road = self.scen_edge[index]
            time = self.scen_time[index]
            x = [self.x_cord[road[0]], self.x_cord[road[1]]]
            y = [self.y_cord[road[0]], self.y_cord[road[1]]]
            ax.plot3D(x, y, 0, c='red')  # 2D route
            ax.text(np.mean(x), np.mean(y), 0, s=f'Time: {time}')

    def showplot(self):
        plt.legend()
        plt.savefig(f'figures/Case_{self.case_number}_3D_plot_{self.condition}.jpg', bbox_inches='tight')
        #plt.show()


if __name__ == '__main__':
    print('With Malfunction')
    solution_file = "solutions/case_3_solutions_dp.pkl"
    matrix_file = "network/linknetwork1_SPMatrixAlt.pkl"
    vis = Visualize(solution_file, matrix_file, 'With Malfunction')
    vis.shownetwork()
    vis.showmalfunction()
    vis.showroutes()
    vis.showplot()

    print('Without Malfunction')
    solution_file = "solutions/case_3_solutions_dp_nomal.pkl"
    matrix_file = "network/linknetwork1_SPMatrix.pkl"
    vis = Visualize(solution_file, matrix_file, 'Without Malfunction')
    vis.shownetwork()
    vis.showroutes()
    vis.showplot()

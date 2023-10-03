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
with open(f'case_{case_number}_solutions_dp.pkl', 'rb') as f:
    solutions, dm, cv, N, N_D, N_S = pickle.load(f)
f.close()
num = len(cv)

#  Calculating the cumulative number of passengers in each vehicle
passenger = []
elapsedTime = []
for sol in solutions:
    # to save the total passengers in a vehicle (solution)
    passTemp1 = []
    buffer1 = 0

    # to save the elapsed time over nodes of one vehicle (solution)
    passTemp2 = [0 for _ in range(len(sol))]
    buffer2 = 0

    for i in range(1, len(sol)):
        buffer1 += dm[sol[i]]
        passTemp1.append(buffer1)

        buffer2 += tt[sol[i-1]][sol[i]][0]  # not considering different scenarios for now
        passTemp2[i] = buffer2

    passenger.append(passTemp1)
    elapsedTime.append(passTemp2)

fig_size = 15

class visualize:
    def __int__(self):
        pass
    def shownetwork(self):
        plt.figure(figsize=(fig_size, fig_size))
        ax = plt.axes(projection='3d')
        for i in N:
            if i in N_S:
                ax.plot3D(x_cord[i], y_cord[i], 0, marker='D', color='gray', markersize=6)
                ax.text(x_cord[i] + 0.5, y_cord[i] + 0.5, 0, f"{i}")
            elif i == num-1:
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
        visLimit = 1
        ax = self.ax
        for solI in range(len(solutions)):
            color = (random.random(), random.random(), random.random())

            for j in range(len(solutions[solI]) - 1):
                x = [x_cord[solutions[solI][j]], x_cord[solutions[solI][j + 1]]]
                y = [y_cord[solutions[solI][j]], y_cord[solutions[solI][j + 1]]]
                z = [elapsedTime[solI][j], elapsedTime[solI][j + 1]]

                if j == 0 or j == len(solutions[solI])-2:   # Plot the line with dashed style
                    if j == len(solutions[solI])-2:  # Set label for the vehicle and plot the dummy node
                        ax.plot3D(x, y, z, c=color, linewidth=fig_size, alpha=0.4, linestyle=':',
                                  label=f'Vehicle {solutions[solI][1]}')
                        ax.plot3D(x[1], y[1], z[1], 'rD', markersize=fig_size)

                    else:  # Plot the dashed line from 0 node to the vehicle
                        ax.plot3D(x, y, z, c=color, linewidth=fig_size, alpha=0.4, linestyle=':')
                    ax.plot3D(x, y, 0, c=color, alpha=0.3)

                else:  # Plot the solid line for the normal trip route
                    ax.plot3D(x, y, z, c=color, linewidth=fig_size, alpha=0.4)
                    ax.plot3D(x, y, 0, c=color, alpha=0.3)

                # Generate the incremental passengers size marker
                z_rate = 0
                for _ in range(passenger[solI][j]):
                    ax.plot3D(x_cord[solutions[solI][j]], y_cord[solutions[solI][j]], elapsedTime[solI][j] + 12 * z_rate,
                              marker='^', color=color, markersize=fig_size, markeredgecolor='black')
                    z_rate += 1
            # Only plot a few vehicles
            if solI == visLimit:
                break
    def showplot(self):
        plt.legend()
        plt.savefig(f'figures/Case_{case_number}_3D_plot.jpg', bbox_inches='tight')
        plt.show()




if __name__ == '__main__':
    vis = visualize()
    vis.shownetwork()
    vis.showroutes()
    vis.showplot()

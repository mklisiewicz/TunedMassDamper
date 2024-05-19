import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
import chardet
import pprint
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.optimize import curve_fit


directory = r'C:\Users\mmate\Szkoła\Drgania Mechaniczne\Krzywa rezonansowa\LD'
undamped_plot_data = []
damped_plot_data = []
raw_data = []
rejected_files = []
omega0 = 17.75
f=1
mu=0.05

def theoreticalCurve(gamma, f=1, mu=0.05):
    def calculateAcceleration(alpha, gamma):
        top = (2 * gamma * alpha) ** 2 + (f ** 2 - alpha ** 2) ** 2
        bottom = (2 * gamma * alpha) ** 2 * (1 - alpha ** 2 - mu * alpha ** 2) ** 2 + ((1 - alpha ** 2) * (f ** 2 - alpha ** 2) - mu * f ** 2 * alpha ** 2) ** 2
        acceleration = alpha ** 2 * np.sqrt(top / bottom)
        return acceleration*2

    alpha_values = np.linspace(0.4, 1.7, 500)
    acceleration_values = calculateAcceleration(alpha_values, gamma)

    return alpha_values, acceleration_values

def isDamped(name):
    file_number = name[-9:-7]
    file_number = int(file_number)
    if file_number > 2 and file_number < 40:
        return True
    return False

def findValue(data, value):
    for i in range(len(data)):
        if data[i][0] == value:
            return i
    return -1

def splitData(data):
    index=2
    data_lin = data[:index]
    data_cub = data[index:]
    return data_lin, data_cub

def extractArrays(data):
    frequency, acceleration = [i[0] for i in data], [i[1] for i in data]
    return frequency, acceleration

def cubicInterpolation(data):
    frequency, acceleration = extractArrays(data)
    interp = interp1d(frequency, acceleration, kind='cubic', fill_value='extrapolate', bounds_error=False)
    X_pred = np.linspace(min(frequency), max(frequency), 1000)
    Y_pred = interp(X_pred)
    return X_pred, Y_pred, frequency, acceleration

def findIntersection(x_vals, y_vals):
    interp1 = interp1d(x_vals[0], y_vals[0], kind='cubic', fill_value='extrapolate', bounds_error=False)
    interp2 = interp1d(x_vals[1], y_vals[1], kind='cubic', fill_value='extrapolate', bounds_error=False)

    def diff_func(x):
        return interp1(x) - interp2(x)

    # Use fsolve to find the roots of the difference function
    x_intersect = fsolve(diff_func, [0.9, 1.15])

    # Calculate the y-values of the intersection points
    y_intersect = [interp1(x) for x in x_intersect]

    # Create the intersection points
    intersections = list(zip(x_intersect, y_intersect))
    print(intersections)
    # Find the points p and q
    p = min(intersections, key=lambda pair: abs(pair[0] - 0.8))
    #q = min(intersections, key=lambda pair: abs(pair[0] - 1.15))
    q=[1.0612, 6.169]
    return p, q

def preparePlot(undamped_data, damped_data):

    plt.figure()
    plt.xlim(0.3, 1.7)
    plt.ylim(0, 38)
    xlim = plt.xlim()
    ylim = plt.ylim()
    x_err = 0.022  
    y_err = 1  
    x_text = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    y_text = ylim[1] - 0.05 * (ylim[1] - ylim[0])

    undamped_x, undamped_y, undamped_x_pts, undamped_y_pts = cubicInterpolation(undamped_data)
    damped_x, damped_y, damped_x_pts, damped_y_pts = cubicInterpolation(damped_data)

    x_vals = [undamped_x, damped_x]
    y_vals = [undamped_y, damped_y]
    
    p, q = findIntersection(x_vals, y_vals)
    
    theoretical_x, theoretical_y = theoreticalCurve(0.1, 0.96)
    theoretical_x_0, theoretical_y_0 = theoreticalCurve(0.42, 0.96)
    #theoretical_x, theoretical_y = theoreticalCurve(0.1, 1)
    #theoretical_x_0, theoretical_y_0 = theoreticalCurve(0.42, 1)
    theoretical_x_vals = [theoretical_x, theoretical_x_0]
    theoretical_y_vals = [theoretical_y, theoretical_y_0]
    #p_t, q_t = findIntersection(theoretical_x_vals, theoretical_y_vals)
    #plt.errorbar(p_t[0], p_t[1], xerr=x_err, yerr=y_err, color='black')
    #plt.errorbar(q_t[0], q_t[1], xerr=x_err, yerr=y_err, color='black')
    #plt.annotate('P', (p_t[0], p_t[1]), textcoords="offset points", xytext=(-10,10), ha='center', fontsize=12, color='black', zorder=3)
    #plt.annotate('Q', (q_t[0], q_t[1]), textcoords="offset points", xytext=(-10,-10), ha='center', fontsize=12, color='black', zorder=3)
    #plt.text(x_text, y_text, f'P ({p_t[0]:.2f}, {p_t[1]:.2f})\nQ ({q_t[0]:.2f}, {q_t[1]:.2f})', ha='left', va='top', fontsize=12, color='black', zorder=3)

    

    
    #plt.errorbar(p[0], p[1], xerr=x_err, yerr=y_err, color='black')
    #plt.errorbar(q[0], q[1], xerr=x_err, yerr=y_err, color='black')
    #plt.annotate('P', (p[0], p[1]), textcoords="offset points", xytext=(-10,10), ha='center', fontsize=12, color='black', zorder=3)
    #plt.annotate('Q', (q[0], q[1]), textcoords="offset points", xytext=(-10,-10), ha='center', fontsize=12, color='black', zorder=3)


    #plt.text(x_text, y_text, f'P ({p[0]:.2f}, {p[1]:.2f})\nQ ({q[0]:.2f}, {q[1]:.2f})', ha='left', va='top', fontsize=12, color='black', zorder=3)
    plt.plot(undamped_x_pts, undamped_y_pts, 'r.', markersize=10, label='Wyniki pomiarów bez eliminatora')
    plt.plot(undamped_x, undamped_y, 'b-', label='Interpolacja krzywej rezonansowej bez eliminatora')
    plt.plot(damped_x_pts, damped_y_pts, 'g.', markersize=10, label='Wyniki pomiarów z eliminatorem')
    plt.plot(damped_x, damped_y, 'purple', label='Interpolacja krzywej rezonansowej z eliminatorem')


    plt.plot(theoretical_x, theoretical_y, 'g--', label='Teoretyczna krzywa rezonansowa dla $\gamma=0.1$ f=0.96')
    plt.plot(theoretical_x_0, theoretical_y_0, 'y--', label='Teoretyczna krzywa rezonansowa dla $\gamma=0.42$ f=0.96')

    plt.xlabel(r'$\frac{{\nu}}{{\omega}_0}$')
    ax = plt.gca()  
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.annotate(r'$\frac{{\nu}}{{\omega}_{10}}$', xy=(1, 0), xycoords='axes fraction', xytext=(0, -10), textcoords='offset points', ha='left', va='top', fontsize=16)
    ax.annotate(r'$a [\frac{m}{s^2}]$', xy=(0, 1), xycoords='axes fraction', xytext=(-20, 0), textcoords='offset points', ha='right', va='top', fontsize=12)
    #ax.annotate(r'$\frac{x_1}{x_{st}}$', xy=(0, 1), xycoords='axes fraction', xytext=(-20, 0), textcoords='offset points', ha='right', va='top', fontsize=12)

    plt.axvline(x=1, color='black', linestyle='--')
    ax.annotate(r'${\nu}={\omega}_{10}=35,5{\pi}\frac{rad}{s}$', xy=(1, 0), xycoords=('data', 'axes fraction'), xytext=(0, -20), textcoords='offset points', ha='center', va='top', color='black', fontsize=12)
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth=0.5)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right')
    plt.show()

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    if filename.endswith(".csv"):
        file_data = pd.read_csv(file_path, delimiter='\t', decimal=',', encoding=result['encoding'])
        file_data.columns = ["t", "a_1", "a_2", "f"]
        average_acceleration = file_data["a_1"].mean()
        print(average_acceleration)
        file_data = file_data.tail(1000)

        max_acceleration = file_data["a_1"].max()
        
        accel = max_acceleration - average_acceleration
        frequency = file_data["f"].mean()
        if frequency > 5:
            if isDamped(filename):
                damped_plot_data.append([frequency/18.5, accel, filename])
            else:
                undamped_plot_data.append([frequency/omega0, accel, filename])
        else:
            rejected_files.append(filename)

damped_plot_data.sort(key=lambda x: x[0])
undamped_plot_data.sort(key=lambda x: x[0])

pp=pprint.PrettyPrinter(indent=4)
pp.pprint(damped_plot_data)
pp.pprint(undamped_plot_data)
pp.pprint(rejected_files)

 
PQsquare_A = 1/(2+mu)*(1+f**2+mu*f**2)
PQsquare_B = (1-2*f**2+(1+mu)**2*f**4)**0.5
P_x = np.sqrt(PQsquare_A - PQsquare_B)
Q_x =np.sqrt(PQsquare_A + PQsquare_B)

h=(2*mu/(1+mu))**0.5
PQ_y=(2-h)/(h*(1+mu)-mu)

print(P_x, PQ_y, Q_x, PQ_y)

preparePlot(undamped_plot_data, damped_plot_data)
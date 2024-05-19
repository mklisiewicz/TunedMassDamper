import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
import chardet
import pprint
from scipy.interpolate import interp1d
from scipy.optimize import fsolve


directory = r'C:\Users\mmate\Documents\Szkoła\Drgania Mechaniczne\Laboratorium Drgań Mechanicznych\Krzywa rezonansowa\LD'
undamped_plot_data = []
damped_plot_data = []
raw_data = []
rejected_files = []
omega0 = 17
omega1 = 17.42




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

def moving_average(y, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(y, window, 'same')

def preparePlot(undamped_data, damped_data):
    un_freq_arr, un_accel_arr = [i[0] for i in undamped_data], [i[1] for i in undamped_data]
    interp_un = interp1d(un_freq_arr, un_accel_arr, kind='linear', fill_value='extrapolate', bounds_error=False)
    X_pred_un = np.linspace(min(un_freq_arr)-0.4, max(un_freq_arr)+0.4, 1000)
    Y_pred_un = interp_un(X_pred_un)
    Y_smooth_un = moving_average(Y_pred_un, 30)

    freq_arr, accel_arr = [i[0] for i in damped_data], [i[1] for i in damped_data]
    interp = interp1d(freq_arr, accel_arr, kind='linear', fill_value='extrapolate', bounds_error=False)
    X_pred = np.linspace(min(freq_arr)-0.4, max(freq_arr)+0.4, 1000)
    Y_pred = interp(X_pred)
    Y_smooth = moving_average(Y_pred, 30)

    def find_intersection(x):
        return interp_un(x) - interp(x)

    x_intersections = fsolve(find_intersection, X_pred)
    p = min(x_intersections, key=lambda x: abs(x - 1))
    q = min(x_intersections, key=lambda x: abs(x - 1.15))
    print(x_intersections)

    plt.figure()
    plt.xlim(0.3, 1.7)
    plt.ylim(0, 38)
    xlim = plt.xlim()
    ylim = plt.ylim()

    x_err = 0.022  
    y_err = 1  
    plt.errorbar(p, interp(p), xerr=x_err, yerr=y_err, color='black')
    plt.errorbar(q, interp(q), xerr=x_err, yerr=y_err, color='black')
    plt.annotate('P', (p, interp_un(p)), textcoords="offset points", xytext=(-10,10), ha='center', fontsize=12, color='black', zorder=3)
    plt.annotate('Q', (q, interp_un(q)), textcoords="offset points", xytext=(-10,-10), ha='center', fontsize=12, color='black', zorder=3)
    
    x_text = xlim[0] + 0.01 * (xlim[1] - xlim[0])
    plt.text(x_text, 30, f'P ({p:.2f}, {interp_un(p):.2f})\nQ ({q:.2f}, {interp_un(q):.2f})', ha='left', va='top', fontsize=12, color='black', zorder=3)

    plt.plot(un_freq_arr, un_accel_arr, 'r.', markersize=10, label='Wyniki pomiarów bez eliminatora')
    plt.plot(X_pred_un, Y_smooth_un, 'b-', label='Interpolacja krzywej rezonansowej bez eliminatora')
    plt.plot(freq_arr, accel_arr, 'g.', markersize=10, label='Wyniki pomiarów z eliminatorem')
    plt.plot(X_pred, Y_smooth, 'purple', label='Interpolacja krzywej rezonansowej z eliminatorem')
    plt.xlabel(r'$\frac{{\nu}}{{\omega}_0}$')
    ax = plt.gca()  # get current axes

    # Remove the original labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add the labels at the end of the axes
    ax.annotate(r'$\frac{{\nu}}{{\omega}_0}$', xy=(1, 0), xycoords='axes fraction', xytext=(0, -10), textcoords='offset points', ha='left', va='top', fontsize=16)
    ax.annotate(r'$a [\frac{m}{s^2}]$', xy=(0, 1), xycoords='axes fraction', xytext=(-20, 0), textcoords='offset points', ha='right', va='top', fontsize=12)

    plt.axvline(x=1, color='black', linestyle='--')
    ax.annotate(r'${\nu}={\omega}_10=34{\pi}\frac{rad}{s}$', xy=(1, 0), xycoords=('data', 'axes fraction'), xytext=(0, -20), textcoords='offset points', ha='center', va='top', color='black', fontsize=12)
    
    plt.axvline(x=omega1/omega0, color='black', linestyle='--')
    ax.annotate(r'${\nu}={\omega}_0=34,84{\pi}\frac{rad}{s}$', xy=(omega1/omega0, 0), xycoords=('data', 'axes fraction'), xytext=(0, -20), textcoords='offset points', ha='center', va='top', color='black', fontsize=12)


    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth=0.5)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5)
    
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
    plt.show()

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    if filename.endswith(".csv"):
        file_data = pd.read_csv(file_path, delimiter='\t', decimal=',', encoding=result['encoding'])
        file_data.columns = ["t", "a_1", "a_2", "f"]
        average_acceleration = file_data["a_1"].mean()
        file_data = file_data.tail(1000)

        max_acceleration = file_data["a_1"].max()
        
        accel = max_acceleration - average_acceleration
        frequency = file_data["f"].mean()
        if frequency > 5:
            if isDamped(filename):
                damped_plot_data.append([frequency/omega0, accel, filename])
            else:
                undamped_plot_data.append([frequency/omega0, accel, filename])
        else:
            rejected_files.append(filename)

damped_plot_data.sort(key=lambda x: x[0])
undamped_plot_data.sort(key=lambda x: x[0])
'''
pp=pprint.PrettyPrinter(indent=4)
pp.pprint(damped_plot_data)
pp.pprint(undamped_plot_data)
pp.pprint(rejected_files)
'''
#mu=0.2
#f=math.sqrt(1/(1+mu))

m=1
c=1


#gamma = c/(2*m*omega0)
#alpha = nu/omega0
gamma=0.1
#f=math.sqrt(1/(1+mu))
mu=0.184
nu_range = np.linspace(0, 2, 1000)
f=0.9
def theoreticCurve(alpha):
    mu=0.05
    f=1
    def calculateAcceleration(alpha):
        top=(2*gamma*alpha)**2+(f**2-alpha**2)**2
        bottom=(2*gamma*alpha)**2*(1-alpha**2-mu*alpha**2)**2+((1-alpha**2)*(f**2-alpha**2)-mu*f**2*alpha**2)**2
        acceleration=alpha**2*math.sqrt(top/bottom)
        return acceleration
    plt.figure()
    plt.xlim(0.2, 1.8)
    plt.ylim(0, 38)
    plt.plot(nu_range, [theoreticCurve(i) for i in nu_range])
    plt.show()

mu=0.05
f=1
PQsquare_A = 1/(2+mu)*(1+f**2+mu*f**2)
PQsquare_B = (1-2*f**2+(1+mu)**2*f**4)**0.5
P_x = PQsquare_A - PQsquare_B
Q_x = PQsquare_A + PQsquare_B
P_x = math.sqrt(P_x)
Q_x = math.sqrt(Q_x)
print(P_x, Q_x)


    

preparePlot(undamped_plot_data, damped_plot_data)
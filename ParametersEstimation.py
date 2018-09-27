
# coding: utf-8

# In[52]:


import numpy as np
from scipy.integrate import odeint
import scipy.optimize as optimize
from math import exp,log, sin
import matplotlib.pyplot as plt

Vg = 1
Rm = 21
a1 = 30
C1 = 2000
Ub = 72/10
C2 = 144#/10
C3 = 1000
Vp = 3/10
Vi = 11/10
E = 0.2/10
U0 = 4
Um = 94
beta = 1.77/10
C4 = 80
Rg = 18
alpha = 0.29/10
C5 = 26#/10
ti = 6/10
Gin = 1.35/10
di = 0.06/10

time_file = open("time_linescape.txt", "r")
glucose_file = open("glucose_data.txt", "r")
insulin_file = open("insulin_data.txt", "r")

time = time_file.read().split('\n')
glucose_data = glucose_file.read().split('\n')
insulin_data = insulin_file.read().split('\n')

     

    
k = 0;
for i in time:
    time[k] = int(time[k])
    k = k + 1
    
k = 0;
for i in glucose_data:
    glucose_data[k] = float(glucose_data[k])
    k = k + 1
k = 0;
for i in time:
    insulin_data[k] = float(insulin_data[k])
    k = k + 1    
    
glucose_data_148 = []
insulin_data_148 = []
glucose_small_data_148 = []
insulin_small_data_148 = []

for i in range(30, 76):
    glucose_data_148.append(glucose_data[i])
    
for i in range(30, 76):
    insulin_data_148.append(insulin_data[i])  
    
for i in range(30, 76):
    glucose_small_data_148.append(glucose_data[i])     
    
for i in range(30, 76):
    insulin_small_data_148.append(insulin_data[i] * 2.1) 
    
    
def f1(G):
    return Rm/(1 + exp((C1 - G/Vg)/a1))
def f2(G):
    return Ub*(1 - exp(-G/(C2*Vg)))
def f3(G):
    return G/(C3*Vg)
def f4(I):
    return U0 + (Um - U0)/(1 + exp(-beta*log(I/C4 * (1/Vi + 1/(E*ti)))))
def f5(I):
    return Rg/(1 + exp(alpha*(I/Vp - C5)))

def model(y, t, A1, A2, B1, B2):
    y1, y2, y3, y4, y5, y6 = y
    dGdt = Gin - f2(y1) - (f3(y1) * f4(y2)) + f5(y5)
    dIdt = f1(y3) - di * y2
    dgdt = y4
    dMdt = -A1 * y4 - B1 * y3 + B1 * y1
    didt = y6
    dNdt = - A2 * y6 - B2 * y5 + B2 * y2
    return [dGdt, dIdt, dgdt, dMdt, didt, dNdt]

#t = np.linspace(0,1170,1170)
#y0 = [95, 10, 40, 1, 250, 1]
#t = np.linspace(0,1170,1170)
#g_solution, y2, y3, y4, y5, y6 = odeint(model, y0, time, mxstep=5000, args = (A1, A2, B1, B2,)).T

t = np.linspace(0,92, 46)
def f(params):
    A1, A2, B1, B2 = params
    #solution, y2 = odeint(oscil_diff_solution, [148, 56, 148, 1, 112, 1], t, args = (omega,)).T
    solution = odeint(model,  [148, 56, 148, 1, 112, 1], t, args = (A1, A2, B1, B2,), mxstep=5000).T
    n1 = np.sqrt(np.sum( (glucose_data_148 - solution[0,:])**2  ))
    n2 = np.sqrt(np.sum( (insulin_data_148 - solution[1,:])**2  ))
    n3 = np.sqrt(np.sum( (glucose_small_data_148 - solution[3,:])**2  ))
    n4 = np.sqrt(np.sum( (insulin_small_data_148 - solution[5,:])**2  ))
    print(np.sqrt(n1**2 + n2**2 + n3**2 + n4**2))
    return np.sqrt(n1**2 + n2**2 + n3**2 + n4**2)

initial_guess = [1, 1, 1000, 0.0001]
result = optimize.minimize(f, initial_guess, method='Nelder-Mead',options={'disp': True, 'ftol':0.05, 'maxiter':1000, 'maxfev':1000})
if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)
    
'''fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(t, glucose_data, color = 'r', lw = 2)
ax.plot(t, g_solution, color = 'b', lw = 2)
ax.grid('on')
plt.show()
'''


'''
A1 = -1.57667748e-02
A2 = 8.21611867e-02
B1 = 9.43687086e+02
B2 = -2.98736506e-04

y0 = [148, 56, 148, 1, 112, 1]
t = np.linspace(0,92, 46)
g_solution, i_solution, y3, y4, y5, y6 = odeint(model, y0, t, mxstep=5000).T
'''
'''
k = 0
for i in i_solution:
    i_solution[k] = i_solution[k] * 100
    k = k + 1
'''
'''

fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(t, g_solution, color = 'b', lw = 2)
ax.plot(t, i_solution, color = 'k', lw = 2)
ax.plot(t, glucose_data_148, color = 'g', lw = 2)
#ax.plot(t, insulin_small_data_148, color = 'yellow', lw = 2)
ax.plot(t, insulin_data_148, color = 'gray', lw = 2)
ax.grid('on')
plt.show()
  
'''


# In[ ]:






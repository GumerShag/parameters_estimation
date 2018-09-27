
# coding: utf-8

# In[93]:


import numpy as np
from scipy.integrate import odeint
from math import exp,log,sqrt
import matplotlib.pyplot as plt


def oscil_solution():
    w = sqrt(3)
    amp = 1
    t = np.linspace(0, 20, 100)
    y = amp*np.sin(w * t)
    return y

def oscil_diff_solution(y, t, omega):
    y1, y2 = y
    return [y2, - omega **2 * y1]

y_real = oscil_solution();

def f(params):
    omega = params
    solution, y2 = odeint(oscil_diff_solution, [0, 3], t, args = (omega,)).T
    #print(solution.reshape(-1, solution.shape[0])[0])
    #print(np.sqrt(np.sum((y_real - solution)**2)))
    return np.sqrt(np.sum((y_real - solution.reshape(-1, solution.shape[0])[0])**2))

t = np.linspace(0, 20, 100)

initial_guess = [1.]
result = optimize.minimize(f, initial_guess, method='Nelder-Mead',options={'disp': True, 'ftol':0.05, 'maxiter':100, 'maxfev':100})
if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)
    

y0 = [0, 2]
omega =1.73623047
t = np.linspace(0, 20, 100)
y1, y2 = odeint(oscil_real, y0, t, full_output=False).T

fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(t, oscil_solution(), color = 'r', lw = 2)
ax.plot(t, y1, color = 'b', lw = 2)
ax.grid('on')
plt.show()





# In[78]:


import scipy.optimize as optimize

def real_solution(x):
    return 5.* np.exp(-0.3 * x)

def cur_solution(k, x):
    return 5.* np.exp(-k * x)

x = np.linspace(0, 10, 20)
y_real = real_solution(x)

def right_path(y, t, k):
    return -k * y


def f(params):
    k = params
    solution = odeint(right_path, 5, x, args = (k,) )
    #print(solution.reshape(-1, solution.shape[0])[0])
    #print(np.sqrt(np.sum((y_real - solution)**2)))
    return np.sqrt(np.sum((y_real - solution.reshape(-1, solution.shape[0])[0])**2))

#k = 0.2
#y0 = [0]
#solution = odeint(right_path, 5, x, args = (k,))
#print(solution)

initial_guess = [1.]
result = optimize.minimize(f, initial_guess, method='Nelder-Mead',options={'disp': True, 'ftol':0.05, 'maxiter':100, 'maxfev':100})
if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)

    
fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(x, real_solution(x), color = 'r', lw = 2)
ax.plot(x, cur_solution(result.x, x), color = 'b', lw = 2)
ax.grid('on')

print(np.sqrt(np.sum( (real_solution(x) - cur_solution(result.x, x))**2 )))
plt.show()


# In[114]:


import scipy.optimize as optimize

def f_solution(x):
    return 3.* np.exp(3. * x) + 1 * np.exp(-x * 3.)

def g_solution(x):
    return 3.* np.exp(3. * x) - 1 * np.exp(-x * 3.)

x = np.linspace(0, 1, 20)


def f_right_path(y, x, a, b):
    f, g = y
    return [g*a, f*b]


def f(params):
    a, b = params
    solution = odeint(f_right_path, [4, 2], x, args = (a,b,) ).T
    #print(solution.shape)
    n1 = np.sqrt(np.sum( (f_solution(x) - solution[0,:])**2  ))
    n2 = np.sqrt(np.sum( (g_solution(x) - solution[1,:])**2  ))
    
    return np.sqrt(n1**2 + n2**2)

#solution = odeint(f_right_path, [4, 2], x ).T
#print(solution.shape)
#k = 0.2
#y0 = [0]
#solution = odeint(right_path, 5, x, args = (k,))
#print(solution)

initial_guess = [1., 1.]
result = optimize.minimize(f, initial_guess, method='Nelder-Mead',options={'disp': True, 'ftol':0.05, 'maxiter':100, 'maxfev':100})
if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)

   
fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(x, f_solution(x), color = 'r', lw = 2)
ax.plot(x, g_solution(x), color = 'g', lw = 2)
ax.plot(x, solution[0, :], color = 'k', lw = 8, alpha = 0.5)
ax.plot(x, solution[1, :], color = 'k', lw = 8, alpha = 0.5)
#ax.plot(x, cur_solution(result.x, x), color = 'b', lw = 2), alpha = 0.3
ax.grid('on')

print(np.sqrt(np.sum( (real_solution(x) - cur_solution(result.x, x))**2 )))
plt.show()


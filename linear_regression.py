import numpy as np
import matplotlib.pyplot as plt

X=np.array([1.1, 2.5, 2.5, 3.2, 5.5, 7.7, 8.9, 9.2])
#| 17  | 21  | 30  | 27  | 60  | 85  | 88  | 95  |
Y=np.array([17, 21, 30, 27, 60, 85, 88, 95])

#prediction function
def predic(x,w,b):
    return w*x+b

def gradient(f,X,h):
    #df has the same length(size) as vector X
    df = np.zeros(X.size)
    #i mean dimention, axis
    for i in range(X.size):
        #difference at point x_i
        #a1, a2 is a vector like X,
        #only differentiate at one axis i, and keep other axis orinal point
        a1 = X.copy()
        a2 = X.copy()
        a1[i] = X[i] - h
        a2[i] = X[i] + h
        df[i] = ( f(a2) - f(a1) )/(2*h)
    return df   

def steepest_descent(f,start,step,precision):
    f_old = float('Inf')
    x = np.array(start)
    steps = []
    f_new = f(x)
    while abs(f_old-f_new)>precision:
    #while np.linalg.norm(ad.gh(f)[0](x))>precision: # an alternative stopping rule
        f_old = f_new # store value at the current point
        d = -gradient(f,x,0.01) # search direction
        x = x+d*step # take a step
        f_new = f(x) # compute function value at the new point
        steps.append(list(x)) # save step
    return x,f_new,steps

def linear_regression():
    F = []
    for i in range(0, np.size(X)):
        fi = lambda p: (Y[i]-predic(X[i],p[0],p[1]))**2
        F.append(fi)
    loss = lambda p: sum( fi(p) for fi in F)/8
    #print(loss([11,2.5]))
    start = [0.0,0.0]
    step_size = 0.001
    precision = 0.00001
    (x_value,f_value,steps) = steepest_descent(loss,start,step_size,precision)
    return x_value


value = linear_regression()
print('w=%f, b=%f'%(value[0],value[1]))
plt.scatter(X, Y, color = 'red')
plt.plot(X , predic(X,value[0],value[1]), color ='yellow')
plt.show()

#testing weights
# w = np.array([1,1,1,1,1,1,1,1])
# b = np.array([1,1,1,1,1,1,1,1])
# print(predic(X,w,b))

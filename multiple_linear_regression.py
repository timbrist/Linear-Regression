import numpy as np
import matplotlib.pyplot as plt

X=np.array([
    [1.1, 2.5, 2.5, 3.2, 5.5, 7.7, 8.9, 9.2],
    [3,     4, 5,   6,   7,   8,   9,   10] ])
#| 17  | 21  | 30  | 27  | 60  | 85  | 88  | 95  |
Y=np.array([17, 21, 30, 27, 60, 85, 88, 95])

def plot_3d_function(f, X,Y):
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(X[0], X[1],Y, color = 'red')
    ax.plot(X[0], X[1], f(X), color ='yellow')
    plt.show()


#prediction function
def predict(x,w,b):
    return np.dot(w,x)+b

#mean square error, risk function, cost function
def loss_function(predict,x,y):
    xi = x.transpose()
    num_rows, num_cols = xi.shape
    L = []
    for i in range(0, num_rows):
        l = lambda w,b: (y[i] - predict(xi[i], w, b))**2
        L.append(l)
    loss = lambda p : sum(  i( p[0], p[1] ) for i in L  ) / num_rows
    return loss

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
    x = start
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

def linear_regression(x,y):
    loss = loss_function(predict, x, y)
    #print(loss([11,2.5]))
    xi = x.transpose()
    num_rows, num_cols = xi.shape
    w = np.zeros(num_cols)
    b = 0.0
    #b = b.reshape((num_rows,1))
    
    start = np.array([w,b])
    step_size = 0.001
    precision = 0.00001
    (x_value,f_value,steps) = steepest_descent(loss,start,step_size,precision)
    return x_value


value = linear_regression(X,Y)
print(value)
# plt.scatter(X, Y, color = 'red')
# plt.plot(X , predict(X,value[0],value[1]), color ='yellow')
# plt.show()

new_predict = lambda x: predict(x,value[0],value[1])
plot_3d_function(new_predict,X,Y)

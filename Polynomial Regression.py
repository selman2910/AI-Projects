import numpy as np
import matplotlib.pyplot as plt

def cost(x,y,w,b):
    m=x.shape[0]
    j=0
    for i in range(m):
        j+=(np.dot(w,x[i])+b-y[i])**2
    j/=(2*m)
    return j

def gradient(x,y,w,b):
    m,n=x.shape
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
        dj_db+=np.dot(w,x[i])+b-y[i]
        for i2 in range(n):
            dj_dw[i2]+=(np.dot(w,x[i])+b-y[i])*x[i,i2]
    dj_dw/=m
    dj_db/=m
    return dj_dw,dj_db

def gradient_descent(x,y,w,b,alpha,itnum):
    n=x.shape[1]
    t_w=np.zeros(n)
    t_b=0
    cost_history = []
    for i in range(itnum):
        dw,db=gradient(x,y,w,b)
        t_w=w-alpha*dw
        t_b=b-alpha*db
        w=t_w
        b=t_b
        cost_history.append(cost(x, y, w, b))
    return w,b, cost_history

def z_score_normalization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    
    # Initialize x_norm as a copy of x to avoid modifying the original array
    x_norm = np.copy(x)
    
    # Create a mask for columns with non-zero sigma
    non_zero_sigma_mask = sigma != 0
    
    # Apply normalization only to the columns where sigma is not zero
    x_norm[:, non_zero_sigma_mask] = (x[:, non_zero_sigma_mask] - mu[non_zero_sigma_mask]) / sigma[non_zero_sigma_mask]
    
    # For columns where sigma is zero, the values are all the same.
    # After subtracting the mean, they all become 0. So we can set them to 0.
    zero_sigma_mask = ~non_zero_sigma_mask
    x_norm[:, zero_sigma_mask] = 0
    
    return x_norm, mu, sigma

def f(w,b,x1,x2, mu, sigma):
    x=np.array([x1,x2])
    x_sq=x**2
    x_int=np.array([x1*x2])
    x_p=np.hstack([x,x_sq,x_int])
    
    # Apply the same normalization as the training data
    x_p_norm = (x_p - mu) / sigma
    
    s=np.dot(w,x_p_norm)+b
    return s

def main():
    x=np.array([[2,9],[4,8],[6,6],[8,5],[10,4]])
    y=np.array([85,90,75,60,50])
    x_squared=x**2
    m,n=x.shape
    x_interaction=np.zeros((m,1))
    for i in range(m):
        c=1
        for i2 in range(n):
            c*=x[i,i2]
        x_interaction[i]=c
    X_poly = np.hstack([x, x**2, x_interaction])
    X_poly, mu, sigma =z_score_normalization(X_poly)
    n1=X_poly.shape[1]
    w=np.zeros(n1)
    b=0
    w,b, cost_history =gradient_descent(X_poly,y,w,b,0.01,1000)
    print(f"{w}\n{b}\n{cost(X_poly,y,w,b)}")
    print(f(w,b,7,7, mu, sigma))

    # Plot cost versus iteration
    plt.plot(cost_history)
    plt.title("Cost vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

main()
import numpy as np
import matplotlib.pyplot as plt
def main():
    file_path = r""
    data = np.loadtxt(file_path, skiprows=1)
    x=data[:,0:2]
    y=data[:,2]
    x,mu,sigma=z_score_normalization(x)
    x_1=np.array([40,90],dtype=float)
    nn=sigma!=0
    n1=~nn
    x_1[nn]=(x_1[nn]-mu[nn])/sigma[nn]
    x_1[n1]=0.0
    w=np.zeros(x.shape[1])
    b=0
    # run gradient descent and collect history for plotting (same behavior as gradient_descent)
    w,b,cost_history,w_history = gradient_descent_with_history(x, y, w, b, 0.01, 1000)
    cost1=cost(x,y,w,b)
    sonuc=sigmoid(x_1,w,b)
    if sonuc<0.5:
        sonuc=0
    else:
        sonuc=1
    print(cost1)
    print(sonuc)
    # save plots: cost, weights, sigmoid and decision boundary (feature space is normalized)
    try:
        plot_cost_and_weights(cost_history, w_history, out_prefix="training")
        plot_decision_boundary(x, y, w, b)
    except Exception as e:
        print(f"Plotting failed: {e}")
def sigmoid(x,w,b):
    a=np.dot(w,x)+b
    z=1/(1+(np.exp(-a)))
    return z
def cost(x,y,w,b):
    m=x.shape[0]
    j=0
    for i in range(m):
        a=sigmoid(x[i],w,b)
        j+=-y[i]*np.log(a)-(1-y[i])*np.log(1-a)
    j/=m
    return j
def z_score_normalization(x):
    mu=np.mean(x,axis=0)
    sigma=np.std(x,axis=0)
    non_zero_sigma=sigma!=0
    x=x.copy()
    x[:,non_zero_sigma]=(x[:,non_zero_sigma]-mu[non_zero_sigma])/sigma[non_zero_sigma]
    zero_sigma=~non_zero_sigma
    x[:,zero_sigma]=0
    return x,mu,sigma
def gradient(x,y,w,b):
    m,n=x.shape
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
        for i2 in range(n):
            dj_dw[i2]+=(sigmoid(x[i],w,b)-y[i])*x[i,i2]
        dj_db+=(sigmoid(x[i],w,b)-y[i])
    dj_dw/=m
    dj_db/=m
    return dj_dw,dj_db
def gradient_descent(x,y,w,b,alpha,itnum):
    n=x.shape[1]
    t_w=np.zeros(n)
    t_b=0
    for i in range(itnum):
        dj_dw,dj_db=gradient(x,y,w,b)
        t_w=w-alpha*dj_dw
        t_b=b-alpha*dj_db
        w=t_w
        b=t_b
    return w,b

def gradient_descent_with_history(x, y, w, b, alpha, itnum):
    """Run gradient descent like `gradient_descent` but record cost and weight history.
    This does not modify any existing functions; it's an additive helper for plotting.
    Returns: (w_final, b_final, cost_history, w_history)
    """
    w = w.copy().astype(float)
    b = float(b)
    cost_history = []
    w_history = [w.copy()]
    for i in range(itnum):
        dj_dw, dj_db = gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost_history.append(cost(x, y, w, b))
        w_history.append(w.copy())
    return w, b, np.array(cost_history), np.array(w_history)

def plot_cost_and_weights(cost_history, w_history, out_prefix="plot"):
    # Single figure: cost (left y-axis) and weights (right y-axis) share same x (iterations)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    it = np.arange(1, len(cost_history) + 1)
    ax1.plot(it, cost_history, '-b', label='cost')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, which='both', axis='both')

    ax2 = ax1.twinx()
    # plot each weight on the right axis
    for j in range(w_history.shape[1]):
        ax2.plot(np.arange(len(w_history)), w_history[:, j], label=f'w[{j}]', linestyle='--')
    ax2.set_ylabel('Weights', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Cost (left) and Weights (right) over iterations')
    plt.tight_layout()
    plt.show()



def plot_decision_boundary(x, y, w, b, out_file="decision_boundary.png"):
    """Plot 2D data (normalized feature space) and the linear decision boundary for w,b.
    Expects x to be (m,2).
    """
    if x.shape[1] != 2:
        print("Decision boundary plot requires 2 features; skipping.")
        return

    plt.figure()
    # scatter the data
    pos = y == 1
    neg = y == 0
    plt.scatter(x[pos, 0], x[pos, 1], c='b', marker='o', label='y=1')
    plt.scatter(x[neg, 0], x[neg, 1], c='r', marker='x', label='y=0')

    # decision boundary: w0*x1 + w1*x2 + b = 0 => x2 = -(w0/w1)*x1 - b/w1
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    xs = np.linspace(x_min, x_max, 200)
    if abs(w[1]) > 1e-12:
        ys = -(w[0] * xs + b) / w[1]
        plt.plot(xs, ys, 'g-', label='decision boundary')
    else:
        # vertical line at x = -b/w0
        if abs(w[0]) > 1e-12:
            xv = -b / w[0]
            plt.axvline(x=xv, color='g', label='decision boundary')

    plt.xlabel('Feature 1 (normalized)')
    plt.ylabel('Feature 2 (normalized)')
    plt.title('Decision boundary (normalized feature space)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_decision_and_sigmoid(x, y, w, b):
    """Combined plot: decision boundary and sigmoid overlaid in one figure using twin y-axis.
    Left x/y: feature scatter and decision boundary. Right y-axis: sigmoid probability plotted
    vs x1 (with x2 fixed at mean) so probabilities correspond to horizontal variation.
    """
    if x.shape[1] != 2:
        print("Decision+sigmoid plot requires 2 features; skipping.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # scatter
    pos = y == 1
    neg = y == 0
    ax1.scatter(x[pos, 0], x[pos, 1], c='b', marker='o', label='y=1')
    ax1.scatter(x[neg, 0], x[neg, 1], c='r', marker='x', label='y=0')

    # decision boundary
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    xs = np.linspace(x_min, x_max, 300)
    if abs(w[1]) > 1e-12:
        ys = -(w[0] * xs + b) / w[1]
        ax1.plot(xs, ys, 'g-', label='decision boundary')
    else:
        if abs(w[0]) > 1e-12:
            xv = -b / w[0]
            ax1.axvline(x=xv, color='g', label='decision boundary')

    ax1.set_xlabel('Feature 1 (normalized)')
    ax1.set_ylabel('Feature 2 (normalized)')
    ax1.set_title('Decision boundary with sigmoid probabilities (right axis)')
    ax1.grid(True)

    # right y-axis: sigmoid probabilities vs x1 (fix x2 at mean of dataset)
    ax2 = ax1.twinx()
    x2_fixed = np.mean(x[:, 1])
    # compute z = w0 * x1 + w1 * x2_fixed + b
    z_vals = w[0] * xs + w[1] * x2_fixed + b
    prob = 1 / (1 + np.exp(-z_vals))
    ax2.plot(xs, prob, color='m', linestyle='--', label='sigmoid (x2 fixed at mean)')
    ax2.set_ylabel('Sigmoid probability', color='m')
    ax2.tick_params(axis='y', labelcolor='m')

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()
main()

        
    

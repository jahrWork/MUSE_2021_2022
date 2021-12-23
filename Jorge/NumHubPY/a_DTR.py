import numpy as np
import matplotlib.pyplot as plt

def DTR_NEU_A(f, D, T, R, gamma, beta, n):
    A = np.zeros([n+1,n+1])
    h = 1/n
    ZDC = [1/(2*h),-2/h,3/(2*h)]
    ZI = [0,-1/h,1/h]

    x = np.linspace(0, 1, n+1)
    AD = (2*np.eye(n+1) - np.diag(np.ones(n), 1) - np.diag(np.ones(n), -1))*D/h**2
    AT = (np.diag(np.ones(n), 1) - np.diag(np.ones(n), -1))*T/(2*h)
    AR = np.eye(n+1)*R
    A = AD + AT + AR
    A[n,:] = np.concatenate([np.zeros(n), [1]])
    A[0,:] = np.concatenate([-np.flip(ZDC), np.zeros(n-2)])
    F = np.concatenate([[gamma], f(x[1:n]), [beta]])
    u = np.linalg.lstsq(A, F)[0]

    plt.plot(x, u)
    plt.show()
    return x, u

def DTR_NEU_B(f, D, T, R, alfa, gamma, n):
    h = 1/n
    ZDC = [1/(2*h),-2/h,3/(2*h)]
    ZI = [0,-1/h,1/h]

    x = np.linspace(0, 1, n+1)
    AD = (2*np.eye(n+1) - np.diag(np.ones(n), 1) - np.diag(np.ones(n), -1))*D/h**2
    AT = (np.diag(np.ones(n), 1) - np.diag(np.ones(n), -1))*T/(2*h)
    AR = np.eye(n+1)*R
    A = AD + AT + AR
    A[0,:] = np.concatenate([[1],np.zeros(n)])
    A[n,:] = np.concatenate([np.zeros(n-2),ZDC])
    F = np.concatenate([[alfa], f(x[1:n]), [gamma]])

    u = np.linalg.lstsq(A, F)[0]
    plt.plot(x, u)
    plt.show()
    return x, u

def DTR_NEU_BPH(f, D, T, R, alfa, gamma, n):
    h = 1/n
    x = np.linspace(0,1,n+1)
    AD = (2*np.eye(n+1)-np.diag(np.ones(n),1)-np.diag(np.ones(n),-1))*D/h**2
    AT = (np.diag(np.ones(n),1)-np.diag(np.ones(n),-1))*T/(2*h)
    AR = np.eye(n+1)*R
    A = AD + AT + AR
    A[0,:] = np.concatenate([[1],np.zeros(n)])
    A[n,n-1] = -1/h**2
    A[n,n] = 1/h**2
    F = np.concatenate([[1], f(x[1:n]), [0.5*f(x[n])+1/h*(-np.exp(3))]])    

    u = np.linalg.lstsq(A, F)[0]
    plt.plot(x, u)
    plt.show()
    return x, u

def DT_UPW_A(f, D, T, alfa, beta, n):
    def R(x):
        return(0*x)
    h = 1/n

    x = np.linspace(0+h, 1-h, n-1)
    AD = (2*np.eye(n-1) - np.diag(np.ones(n-2), 1) - np.diag(np.ones(n-2), -1))*D/h**2
    AT = (-np.diag(np.ones(n-1), 0) + np.diag(np.ones(n-2), 1))*T/h
    AR = np.diag(R(x), 0)
    A = AD + AT + AR
    F = f(x)
    F[0] = F[0] + (D/h**2)*alfa
    F[n-2] = F[n-2] + (D/h**2 - T/h)*beta

    u=np.linalg.lstsq(A, F)[0]
    plt.plot(np.concatenate([[0], x, [1]]), np.concatenate([[alfa],u,[beta]]))
    plt.show()
    return (np.concatenate([[0], x, [1]]), np.concatenate([[alfa],u,[beta]]))

def DT_UPW_I(f, D, T, alfa, beta, n):
    def R(x):
        return 0*x
    h = 1/n

    x = np.linspace(0+h, 1-h, n-1)
    AD = (2*np.eye(n-1) - np.diag(np.ones(n-2), 1) - np.diag(np.ones(n-2), -1))*D/h**2
    AT = (np.diag(np.ones(n-1), 0) - np.diag(np.ones(n-2), -1))*T/h
    AR = np.diag(R(x), 0)
    A = AD + AT + AR
    F = f(x)
    F[0] = F[0] + (D/h**2 + T/h)*alfa
    F[n-2] = F[n-2] + (D/h**2)*beta

    u = np.linalg.lstsq(A, F)[0]
    plt.plot(np.concatenate([[0], x, [1]]),np.concatenate([[alfa],u,[beta]]))
    plt.show()
    return np.concatenate([[0], x, [1]]),np.concatenate([[alfa],u,[beta]])

""""
%tests

D=1
T=1
R=1
alfa=1
gamma=1
beta=1
n=20

def f(x):
    return(D*np.exp(3*x)*(9*x**2+3*x-4)+T*np.exp(3*x)*(x-3*x**2+1)+R*(np.exp(3*x)*(x-x**2)+1))

def g(x):
    return(np.exp(3*x)*(-4 +3*x+9*x**2))

def w(x):
    return(np.exp(3*x)*(T-4*D+(T+3*D)*x+3*(3*D-T)*x**2))

DTR_NEU_A(f,D,T,R,gamma,beta,n)

alfa=1
gamma=-np.exp(3)
DTR_NEU_B(f,D,T,R,alfa,gamma,n)

DTR_NEU_BPH(g,D,T,R,alfa,gamma,n)

D=1e-4
T=1
DT_UPW_A(w,D,T,alfa,beta,n)

DT_UPW_I(w,D,T,alfa,beta,n)
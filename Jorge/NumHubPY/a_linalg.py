import numpy as np

#A = np.array([[1, 2, 3, 6], [6, 3, 4, 5], [1, 7, 40., 6], [7, 8, 9, 10]])
#b = np.array([1, 2, 3, 4])
#A = np.array([[4, 1], [1, 4.]])
#b = [2,1]

def FW(A, b):
    n = len(A)
    L = np.tril(A)
    x = np.zeros(n)

    x[0] = b[0]/L[0,0]
    for i in np.arange(1, n):
      x[i] = (b[i] - sum(L[i,0:i]*x[0:i]))/L[i,i]
    return x

def BK(A, b):
    n = len(A)
    U = np.triu(A)
    x = np.zeros(n)

    x[n-1] = b[n-1]/U[n-1,n-1]
    for i in np.arange((n-2), -1, -1):
      x[i] = (b[i] - sum(U[i,(i+1):n]*x[(i+1):n]))/U[i,i]
    return x

def LU(A):
    n = len(A)
    x = np.zeros(n)
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    As = A
    for k in np.arange(0, n-1):
        for i in np.arange(k+1, n, 1):
            L[i,k] = As[i,k]/As[k,k]
            for j in np.arange(k, n, 1):
                As[i,j] = As[i,j] - L[i,k]*As[k,j]
    L = L+np.eye(n)
    U = np.triu(As)
    return L, U

def TH(A):
    x = np.zeros(n)
    a = np.diag(A)
    e = np.diag(A, -1)
    c = np.diag(A, 1)
    alpha = np.zeros(n)
    beta = np.zeros(n-1)
    L = np.zeros([n,n])
    U = np.zeros([n,n])
    alpha[0] = a[0]
    for i in np.arange(1, n):
        beta[i-1] = e[i-1]/alpha[i-1]
        alpha[i] = a[i] - beta[i-1]*c[i-1]
    L=np.diag(beta, -1) + np.eye(n)
    U=np.diag(c, 1) + np.diag(alpha)
    return L, U

def CH(A):
    n = len(A)
    C = np.tril(A)
    A = C + np.transpose(C) + 5*np.diag(np.diag(C))
    R = np.zeros([n,n])
    R[0,0] = np.sqrt(A[0,0])
    for i in np.arange(1, n, 1):
        for j in np.arange(0, i, 1):
            R[j,i] = (A[i,j] - (np.transpose(R[0:j,i])@R[0:j,j]))/R[j,j]
        R[i,i] = np.sqrt(A[i,i] - (np.transpose(R[0:i,i])@R[0:i,i]))
    return R

def CH2(A):
    n = len(A)
    C = np.tril(A)
    A = C + np.transpose(C) + 5*np.diag(np.diag(C))
    R = np.zeros([n,n])
    print(A)
    R[0,0] = np.sqrt(A[0,0])
    for i in np.arange(1, n, 1):
        ooo = 0
        for j in np.arange(0, i, 1):
            oo = 0
            for k in np.arange(0, j, 1):
                oo = oo + R[k,i]*R[k,j]
            R[j,i] = (A[i,j] - (oo))/R[j,j]
        for k in np.arange(0, i):
            ooo = ooo + R[k,i]**2
        R[i,i] = np.sqrt(A[i,i] - (ooo))
    return R

def GR(A, b):
    n = len(A)
    x = np.ones(n)
    r = b - A@x
    n = 0
    P = np.diag(np.diag(A))
    a = 0.2

    while np.linalg.norm(r)/np.linalg.norm(b) > 1e-10 and n < 1e5:
        n = n+1
        z = np.linalg.lstsq(P,r)[0]
        a = (z@r)/(z@(A@z))
        x = x + a*z
        r = r - a*A@z
    print('Number of iterations:', n)
    return x

def GRC(A, b):
    n = len(A)
    x = np.ones(n)
    r = b - A@x
    p = r
    n = 0
    while np.linalg.norm(r)/np.linalg.norm(b) > 1e-10 and n < 1e5:
        n = n + 1
        a = (p@r)/(p@(A@p))
        x = x + a*p
        r = r - a*(A@p)
        bt = (p@(A@r))/(p@(A@p))
        p = r - bt*p
    print('Number of iterations:',n)
    return x

def P_inv_shift(A,shift,inv_1):
    m = len(A)
    x = np.ones(m)
    y = x/np.linalg.norm(x)
    n = 0
    M = A - shift*np.eye(m)

    while n < 1000:
        n = n + 1
        if inv_1 == 1:
            x = np.linalg.lstsq(M, y)[0]
        else:
            x = A@y
        y = x/np.linalg.norm(x)
        bamba = np.conj(y)@(A@y)
    print('Number of iterations:', n)
    return bamba

def QR(A):
    n = 0
    while n < 1e4 and (np.max(np.abs(np.tril(A, -1)))) > 1e-12:
        Q, R = np.linalg.qr(A)
        A = R@Q
        n = n + 1
    print('Number of QR iterations:', n)
    return np.diag(A)
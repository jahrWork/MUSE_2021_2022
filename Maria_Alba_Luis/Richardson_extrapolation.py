import numpy as np
from numerical_schemes import Euler_forward, Euler_backward, RungeKutta4, CrankNicholson
from Cauchy_problem import Cauchy_problem, Cauchy_analitic_solution
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Primero velocidad de convergencia

# PASOS A SEGUIR PARA EJECUTAR EL CODIGO ######

#### 1) Select initial conditions.
#### 2) WAIT!
# 1)

y10 = 0
y20 = 1
t0 = 0                      # tiempo inicial
tf = 1                      # tiempo final

#NU = np.arange(10,10000,100) #10000 es cuando conseguimos petar el Runge Kutta , para calcular los q, solo correr de 5 a 205 de 5 en 5
NU = np.arange(5,205,5) #usar este para calcular las q
NV = 2*NU

esquemas = ["Euler_forward","RungeKutta4","CrankNicholson","Euler_backward"]

comparativa = np.zeros((4,NU.size))

for m in esquemas:
    f = esquemas.index(m)
    U_Error_vector = np.zeros(NU.size)

    for i in range(NU.size):
        TimeU = np.linspace(t0,tf,NU[i]+1)
        TimeV = np.linspace(t0,tf,NV[i]+1)

        yU = Cauchy_problem(y10,y20,t0,tf,NU[i],m)
        yU1 = yU[:,0]
        yU2= yU[:,1]

        yV = Cauchy_problem(y10,y20,t0,tf,NV[i],m)
        yV1 = yV[:,0]
        yV2= yV[:,1]

        U_Error = np.zeros((NU[i]+1,2))
        U_Error_norma=np.zeros(NU[i]+1)
        for j in range(NU[i]+1):
            #Error[j] =  yU[j] - yV[2*j]
            U_Error[j,0] = yU1[j] - yV1[2*j]
            U_Error[j,1] =  yU2[j] - yV2[2*j]
            U_Error_norma[j]=np.linalg.norm(U_Error[j,:])
        #Error = np.linalg.norm(Error)
        #Error_vector[i] = np.log10(Error)
        U_Error_vector[i] = np.log10(U_Error_norma[int(NU[i]/5)])
    #print(yU[1] - yV[2])
    #print(Error_vector)

    comparativa[f,:] = U_Error_vector #guarda los log de los errores de U

print(comparativa)
 


#Linear Regression

x=np.log10(NU)
m= np.zeros((4,1))
n= np.zeros((4,1))
comparativa_regresion=np.zeros((4,NU.size))
for a in esquemas:
    f = esquemas.index(a)
    regresion_lineal = LinearRegression()
    regresion_lineal.fit(x.reshape(-1,1),comparativa[f,:] )

    m[f,0]=regresion_lineal.coef_
    n[f,0]=regresion_lineal.intercept_

    for i in range(len(NU)):
        comparativa_regresion[f,i]=m[f,0]*x[i]+n[f,0]
 
print(m)
  

# Segundo ya tenemos la q de la velocidad de convergencia con ella sacamos el error para Richardson
log_Error_Richardson=np.zeros((4,NU.size))
for p in esquemas:
    f = esquemas.index(p)
    for i in range(NU.size):
        log_Error_Richardson[f,i]=comparativa[f,i]-np.log10((1-(1/2)**abs(m[f,0])))
           

# para plotear la regresion lineal frente a los valores dispersos
plt.plot(x,comparativa_regresion[3,:])
plt.plot(x,comparativa[3,:],'ro')
plt.legend(['Linear regression', 'Scattered values'] )
plt.xlabel('log(N)')
plt.ylabel('log(U2-U1)')
plt.title('Euler Backward')
plt.show()


plt.plot(x,comparativa_regresion[0,:])
plt.plot(x,comparativa[0,:],'ro')
plt.legend(['Linear regression','Scattered values'])
plt.xlabel('log(N)')
plt.ylabel('log(U2-U1)')
plt.title('Euler Forward')
plt.show()

plt.plot(x,comparativa_regresion[1,:])
plt.plot(x,comparativa[1,:],'ro')
plt.legend(['Linear regression','Scattered values'])
plt.xlabel('log(N)')
plt.ylabel('log(U2-U1)')
plt.title('Runge Kutta 4')
plt.show()

plt.plot(x,comparativa_regresion[2,:])
plt.plot(x,comparativa[2,:],'ro')
plt.legend(['Linear regression','Scattered values'])
plt.xlabel('log(N)')
plt.ylabel('log(U2-U1)')
plt.title('Crank Nicolson')
plt.show()

#para plotear  error richardson
plt.plot(x,log_Error_Richardson[3,:],'r-')
plt.plot(x,log_Error_Richardson[0,:],'b-')
plt.plot(x,log_Error_Richardson[2,:],'g-')
plt.plot(x,log_Error_Richardson[1,:],'k-')
plt.legend([ 'Euler Backward', 'Euler Forward','Crank Nicolson', 'Runge Kutta 4'])
plt.xlabel('log(N)')
plt.ylabel('log(E)')
plt.title('Error by means of Richardson extrapolation')
plt.show()

## cool plot
#Esto es error_u frente a NU en todos los esquemas, sin la regrecsión, son las graficas de antes
fig = go.Figure()
fig.add_trace(go.Scatter(x = np.log10(NU), y = comparativa[0,:], name = "Euler_forward"))
fig.add_trace(go.Scatter(x = np.log10(NU), y = comparativa[1,:], name = "RungeKutta4"))
fig.add_trace(go.Scatter(x = np.log10(NU), y = comparativa[2,:], name = "CrankNicholson"))
fig.add_trace(go.Scatter(x = np.log10(NU), y = comparativa[3,:], name = "Euler_backward"))
fig.update_layout(
    title = "Mapa de convergencia",
    xaxis = {
        'domain': [0.25, 0.75],
        #'range': [0,10]
        },
    yaxis = {
        #'range': [-1,1]
        }
    )
fig.show()
#fig.write_image("fig1.png")


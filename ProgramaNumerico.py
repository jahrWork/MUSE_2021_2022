import numpy as np
from Kepler_problem import Kepler_orbit
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from Cauchy_problem import Cauchy_problem, Cauchy_analitic_solution
from OsciladorArmonico import Oscillator
from numerical_schemes import Stability_region_Euler_forward,Stability_region_Euler_backward, Stability_region_RungeKutta4, Stability_region_LeapFrog, Stability_region_CrankNicholson

# PASOS A SEGUIR PARA EJECUTAR EL CODIGO ######

#### 1) elegir Kepler problem o Cauchy problem segun quieras ejecutar uno u otro problema, si no tarda mucho y llena demasiada memoria
#### 2) elegir las condiciones iniciales del problema, asi como tf y el numero de puntos de integracion
#### 3) elegir el esquema numerico que se desea utilizar: Euler_forward, Euler_backward, CrankNicholson, RungeKutta4
####        (no he puesto todos a la vez porque no tiene sentido compararlos y es muy lento)

########### aviso para Alba y Luis: He puesto la extrapolacion de Richardson en otro sitio porque no estoy segura de si esta bien...
########### ... si quereis probarla ejecutad el archivo que se llama Richardson_extrapolation directamemte (cuando sepa si esta bien lo cambio)



# 1)

#problem_election = "Kepler_problem"
#problem_election = "Cauchy_problem"
problem_election = "Linear_Oscillator"

numerical_method= "CrankNicholson"   # "Euler_forward"  ;  ; "RungeKutta4","CrankNicholson" "Euler_backward" "LeapFrog"


## KEPLER PROBLEM #################################################################################################################



if problem_election == "Kepler_problem":

    # kepler problem (pinta en una grafica x-y)

    # 2)

    x0 = 1
    y0 = 0
    xdot0 = 0
    ydot0 = 1
    N = 100                # numero de puntos de integracion
    t0 = 0                      # tiempo inicial
    tf = 12*3              # tiempo final
    #delta_t=0.0036
    # 3)

    scheme = numerical_method

    orbita, tiempo = Kepler_orbit(x0,y0,xdot0,ydot0,t0,tf,N,scheme)


    print(orbita[:,0],orbita[:,1])

    #normal plot

    a='Crank Nicholson, N=100 '
    x= orbita[:,0]
    y = orbita[:,1]
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(a)
    #plt.xlim(-1.2,1.2)
    #plt.ylim(-1.2,1.2)
    plt.show()

    b='Crank Nicholson, N=100 : x (t), y(t)'
    plt.plot(tiempo,x)
    plt.plot(tiempo,y)
    plt.xlabel('t')
    plt.ylabel('x,y')
    plt.title(b)
    plt.legend(['x','y'])
    #plt.xlim(-1.2,1.2)
    #plt.ylim(-1.2,1.2)
    plt.show()



    # cool plot

    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x = orbita[:,0], y = orbita[:,1]))
    #fig.update_layout(
    #    title = scheme,
    #    xaxis = {
    #        'domain': [0.25, 0.75],
    #       #'range': [-1,1]
    #       },
    #    yaxis = {
    #       # 'range': [-1,1]
    #        }
    #    )
    #fig.show()
    #fig.write_image("fig1.png")




## CAUCHY PROBLEM  ############################################################################################################



elif problem_election == "Cauchy_problem":

    ## CAUCHY PROBLEM (pinta la solucion analitica vs la numerica en y1 y y2)

    # 2)

    y10 = 0
    y20 = 1
    N = 10                   # numero de puntos de integracion
    t0 = 0                      # tiempo inicial
    tf = 1                      # tiempo final

    # 3)

    scheme = numerical_method

    y = Cauchy_problem(y10,y20,t0,tf,N,scheme)

    Time = np.linspace(t0,tf,N+1)

    print(y[:,0],y[:,1],Time)


    ############ analitical vs numerical solution plot #########################################################################

    y0 = np.array([y10, y20])
    y_analitic = y0
    for i in Time:
        print(i)
        y_analitic = np.vstack((y_analitic,Cauchy_analitic_solution(i)))  #Cuidado! el primer punto esta repetido, hay qeu tener cuidado al plotear
    
    print(y_analitic)
    # cool plot

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = Time, y = y_analitic[1:,0], name = "y1 analitica"))
    fig.add_trace(go.Scatter(x = Time, y = y[:,0], name = "y1 numerica"))
    fig.update_layout(
        title = scheme,
        xaxis = {
            'domain': [0.25, 0.75],
            #'range': [-1,1]
            },
        yaxis = {
            #'range': [-1,1]
            }
        )
    fig.show()
    #fig.write_image("fig1.png")

    # cool plot

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = Time, y = y_analitic[1:,1], name = "y2 analitica"))
    fig.add_trace(go.Scatter(x = Time, y = y[:,1], name = "y2 numerica"))
    fig.update_layout(
        title = scheme,
        xaxis = {
            'domain': [0.25, 0.75],
            #'range': [-1,1]
            },
        yaxis = {
            #'range': [-1,1]
            }
        )
    fig.show()
    #fig.write_image("fig1.png")





elif problem_election == "Linear_Oscillator":

    ## Linear_Oscillator (pinta la solucion analitica vs la numerica en y1 y y2)

    # 2)
    x0=0
    x0_dot=1
    x0_vec = ([x0,x0_dot])
    N = 250               # numero de puntos de integracion
    t0 = 0                      # tiempo inicial
    tf = 12*np.pi    # tiempo final

    delta_t=(tf-t0)/N
    print(delta_t)

    # 3)

    #Stability_region_Euler_forward()
    #Stability_region_Euler_backward()
    #Stability_region_RungeKutta4()
    #Stability_region_LeapFrog()
    #Stability_region_CrankNicholson()

    #Time = np.linspace(t0,tf,N+1)

    scheme = "Euler_forward"
    x,Time = Oscillator(x0_vec,t0,tf,N,scheme)
    print(x[:,0],x[:,1],Time)


    plt.plot(Time,x[:,0] )
    plt.title('Euler Forward, delta_t=0.15')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.grid()
    plt.show()


    scheme = "Euler_backward"
    x,Time = Oscillator(x0_vec,t0,tf,N,scheme)
    print(x[:,0],x[:,1],Time)


    plt.plot(Time,x[:,0] )
    plt.title('Euler Backward, delta_t=0.15')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.grid()
    plt.show()


    scheme = "CrankNicholson"
    x,Time = Oscillator(x0_vec,t0,tf,N,scheme)
    print(x[:,0],x[:,1],Time)

    plt.plot(Time,x[:,0] )
    plt.title('Crank Nicolson, delta_t=0.15')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.grid()
    plt.show()

    scheme = 'RungeKutta4'
    x,Time = Oscillator(x0_vec,t0,tf,N,scheme)
    print(x[:,0],x[:,1],Time)


    plt.plot(Time,x[:,0] )
    plt.title('Runge Kutta 4 , delta_t=0.15')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.grid()
    plt.show()



    scheme = 'LeapFrog'
    x,Time = Oscillator(x0_vec,t0,tf,N,scheme)
    print(x[:,0],x[:,1],Time)

    plt.plot(Time,x[:,0] )
    plt.title('Leap-Frog , delta_t=0.15')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.grid()
    plt.show()

     
else:
    raise Exception("Mal escrito el problema")

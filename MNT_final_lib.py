import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.njit
def Att_velh_ast(velh, velv, vel_U, velh_ast, Re, Nx, Ny, dx, dy, dt):
    # atualizando velocidade horizontal
    for i in range(1, Nx):
        for j in range(0, Ny):
            C1 = 0.25*(velv[i, j+1]+velv[i-1, j+1]+velv[i, j]+velv[i-1, j])
            R = -dt*(velh[i, j]*(velh[i+1, j]-velh[i-1, j])/(2*dx))
            R += -dt*(C1*(velh[i, j+1]-velh[i, j-1])/(2*dy))
            R += (dt/Re)*(velh[i+1, j]-2*velh[i, j]+velh[i-1, j])/(dx*dx)
            R += (dt/Re)*(velh[i, j+1]-2*velh[i, j]+velh[i, j-1])/(dy*dy)

            velh_ast[i, j] = velh[i, j] + R

    for j in range(Ny):
        velh_ast[0, j] = 0
        velh_ast[Nx, j] = 0

    for i in range(Nx+1):
        velh_ast[i, -1] = -velh_ast[i, 0]
        velh_ast[i, Ny] = 2*vel_U[i] - velh_ast[i, Ny-1]

    return velh_ast

@numba.njit
def Att_velv_ast(velv, velh, velv_ast, Re, Nx, Ny, dx, dy, dt):
    # atualizando velocidade vertical
    for i in range(0, Nx):
        for j in range(1, Ny):
            C2 = 0.25*((velh[i+1, j]+velh[i, j]+velh[i+1, j-1]+velh[i, j-1]))
            R = -dt*(velv[i, j]*(velv[i, j+1]-velv[i, j-1])/(2*dy))
            R += -dt*(C2*(velv[i+1, j]-velv[i-1, j])/(2*dx))
            R += (dt/Re)*(velv[i, j+1]-2*velv[i, j]+velv[i, j-1])/(dy*dy)
            R += (dt/Re)*(velv[i+1, j]-2*velv[i, j]+velv[i-1, j])/(dx*dx)

            velv_ast[i, j] = velv[i, j] + R

    for j in range(Ny+1):
        velv_ast[-1, j] = -velv_ast[0, j]
        velv_ast[Nx, j] = -velv_ast[Nx-1, j]

    for i in range(Nx):
        velv_ast[i, 0] = 0
        velv_ast[i, Ny] = 0
    return velv_ast

@numba.njit
def Lambda(dx, dy, x=-1, y=-1):
    return x/(dx*dx)  + y/(dy*dy)
@numba.njit
def Influencia_inercia(matz_velh, matz_velv, dx, dy, dt, i, j):
    return (matz_velh[i+1, j] - matz_velh[i, j])/(dt*dx) + (matz_velv[i, j+1] - matz_velv[i, j])/(dt*dy)
@numba.njit
def Res_press_e(matz_press, dx, i, j):
    return (matz_press[i-1, j] - matz_press[i, j])/(dx*dx)
@numba.njit
def Res_press_d(matz_press, dx, i, j):
    return (-matz_press[i, j] + matz_press[i+1, j])/(dx*dx)
@numba.njit
def Res_press_c(matz_press, dy, i, j):
    return (-matz_press[i, j] + matz_press[i, j+1])/(dy*dy)
@numba.njit
def Res_press_b(matz_press, dy, i, j):
    return (matz_press[i, j-1] - matz_press[i, j])/(dy*dy)

@numba.jit(nopython=True)
def Att_press(press, velh_ast, velv_ast, Nx, Ny, dx, dy, dt, tol=1.e-3):
    # atualizando a pressao
    error = 100
    lambd = 0
    ite = 0
    while error > tol:
        ite += 1
        Rm = 0 # erro maximo
        for i in range(Nx):
            for j in range(Ny):
                R = Influencia_inercia(velh_ast, velv_ast, dx, dy, dt, i, j)
                if i == 0 and j == 0:
                    lambd = Lambda(dx, dy)
                    R -= (Res_press_d(press, dx, i, j) + Res_press_c(press, dy, i, j))

                elif i == 0 and j == Ny-1:
                    lambd = Lambda(dx, dy)
                    R -= (Res_press_d(press, dx, i, j) + Res_press_b(press, dy, i, j))

                elif i == Nx-1 and j == 0:
                    lambd = Lambda(dx, dy)
                    R -= (Res_press_e(press, dx, i, j) + Res_press_c(press, dy, i, j))

                elif i == Nx-1 and j == Ny-1:
                    lambd = Lambda(dx, dy)
                    R -= (Res_press_e(press, dx, i, j) + Res_press_b(press, dy, i, j))

                elif i == 0 and (j != 0 and j != Ny-1):
                    lambd = Lambda(dx, dy, y=-2)
                    R -= (Res_press_d(press, dx, i, j))
                    R -= (Res_press_c(press, dy, i, j) + Res_press_b(press, dy, i, j))

                elif i == Nx-1 and (j != 0 and j != Ny-1):
                    lambd = Lambda(dx, dy, y=-2)
                    R -= (Res_press_e(press, dx, i, j))
                    R -= (Res_press_c(press, dy, i, j) + Res_press_b(press, dy, i, j))

                elif j == 0:
                    lambd = Lambda(dx, dy, x=-2)
                    R -= (Res_press_e(press, dx, i, j) + Res_press_d(press, dx, i, j))
                    R -= (Res_press_c(press, dy, i, j))

                elif j == Ny-1:
                    lambd = Lambda(dx, dy, x=-2)
                    R -= (Res_press_e(press, dx, i, j) + Res_press_d(press, dx, i, j))
                    R -= (Res_press_b(press, dy, i, j))

                else:
                    lambd = Lambda(dx, dy, x=-2, y=-2)
                    R -= (Res_press_e(press, dx, i, j) + Res_press_d(press, dx, i, j))
                    R -= (Res_press_c(press, dy, i, j) + Res_press_b(press, dy, i, j))

                R = R/lambd
                press[i, j] = press[i, j] + R
                if abs(R) > Rm:
                    Rm = abs(R)
        error = Rm


    for i in range(Nx):
        press[i, -1] = press[i, 0]
        press[i, Ny] = press[i, Ny-1]

    for j in range(Ny):
        press[-1, j] = press[0, j]
        press[Nx, j] = press[Nx-1, j]

    press[-1, -1] = press[0, 0]
    press[-1, Ny] = press[0, Ny-1]
    press[Nx, -1] = press[Nx-1, 0]
    press[Nx, Ny] = press[Nx-1, Ny-1]

    return press, ite

@numba.njit
def Att_velh_new(press, velh, velh_ast, Nx, Ny, dx, dt):
    # Nova matriz velocidade 
    for i in range(1, Nx):
        for j in range(-1, Ny+1):
            velh[i, j] = velh_ast[i, j] - dt*(press[i, j] - press[i-1, j])/dx
    return velh

@numba.njit
def Att_velv_new(press, velv, velv_ast, Nx, Ny, dy, dt):
    for i in range(-1, Nx+1):
        for j in range(1, Ny):
            velv[i, j] = velv_ast[i, j] - dt*(press[i, j] - press[i, j-1])/dy
    return velv

@numba.jit(nopython=True)
def Matriz_plot(matriz, velh, velv, Nx, Ny, dx, dy, dt, tol=1.e-3):
# Matriz para plot
    lambd = Lambda(dx, dy, x=-2, y=-2)
    error = 100
    while error > tol:
        Rm = 0
        for i in range(1, Nx):
            for j in range(1, Ny):
                R = -(-velv[i-1, j] + velv[i, j])/dx
                R += (-velh[i, j-1] + velh[i, j])/dy
                R -= Res_press_e(matriz, dx, i, j) + Res_press_d(matriz, dx, i, j)
                R -= Res_press_c(matriz, dy, i, j) + Res_press_b(matriz, dy, i, j) 

                R = R/lambd
                matriz[i, j] += R
                if abs(R) > Rm:
                    Rm = abs(R)
        error = Rm
    return matriz

# @numba.jit
def Evolucao(velocidade_tampa, Lx, Ly, Re, Nx, Ny, tempo, tol=1.e-3):

    # parametros
    dx = Lx/Nx
    dy = Ly/Ny
    dt = np.min([Re*0.1*(dx*dx), 0.5*dx])

    # matrizes
    press = np.zeros((Nx+2, Ny+2), dtype=float) # pressao
    velh = np.zeros((Nx+1, Ny+2), dtype=float) # velocidade horizontal
    velv = np.zeros((Nx+2, Ny+1), dtype=float) # velocidade vertical

    velh[0:Nx+1, Ny] = 2*velocidade_tampa[0: Nx+1] # ghost points de contorno
    velh_ast = velh.copy()
    velv_ast = velv.copy()
    t = 0
    while t < tempo+1.e-8:
        t += dt
        print(t)
        velh_ast = Att_velh_ast(velh, velv, velocidade_tampa, velh_ast, Re, Nx, Ny, dx, dy, dt)
        velv_ast = Att_velv_ast(velv, velh, velv_ast, Re, Nx, Ny, dx, dy, dt)
        press, ite = Att_press(press, velh_ast, velv_ast, Nx, Ny, dx, dy, dt, tol=tol)
        velh = Att_velh_new(press, velh, velh_ast, Nx, Ny, dx, dt)
        velv = Att_velv_new(press, velv, velv_ast, Nx, Ny, dy, dt)
    
    matriz = np.zeros((Nx+1, Ny+1), dtype=float)
    matriz = Matriz_plot(matriz, velh, velv, Nx, Ny, dx, dy, dt, tol=tol)

    return press, velh, velv, matriz, ite

def Vel_plot(Nx, Ny, velh, velv):
    uplot = np.zeros((Nx+1, Ny+1), dtype=float)
    vplot = np.zeros((Nx+1, Ny+1), dtype=float)
    for i in range(Nx+1):
        for j in range(Ny+1):
            uplot[i, j] = 0.5*(velh[i, j]+velh[i, j-1])
            vplot[i, j] = 0.5*(velv[i, j]+velv[i-1, j])
    return uplot, vplot

def Press_plot(press, Nx, Ny):

    press_plot = np.zeros((Nx+1, Ny+1), dtype=float)
    for i in range(Nx):
        for j in range(Ny):
            if i == 0 and j == 0:
                press_plot[i, j] = 0.333*(press[i, j-1]+press[i-1, j]+press[i, j])
            elif i == 0 and j == Ny:
                press_plot[i, j] = 0.333*(press[i-1, j-1]+press[i-1, j]+press[i, j])
            elif i == Nx and j == 0:
                press_plot[i, j] = 0.333*(press[i-1, j-1]+press[i, j-1]+press[i-1, j]+press[i, j])
            else:
                press_plot[i, j] = 0.25*(press[i-1, j-1]+press[i, j-1]+press[i-1, j]+press[i, j])
    return press_plot

# ---------------------------------------------------------
def Plot(x, y, uplot, vplot, tempo, Re, matriz_fundo, color='yellow', strem_function=False):
    # plot
    if strem_function:
        plt.contourf(x, y, matriz_fundo.transpose(), cmap='Spectral')
    else:
        plt.contourf(x, y, matriz_fundo.transpose(), cmap='plasma')
    # plt.quiver(x, y, uplot.transpose(), vplot.transpose())
    plt.streamplot(x, y, uplot.transpose(), vplot.transpose(), color=color)
    plt.title(f'tempo:{tempo:.2f}s; Re:{Re}')
    plt.axis('scaled')
    plt.show()
# ---------------------------------------------------------

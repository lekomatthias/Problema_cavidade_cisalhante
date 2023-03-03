from MNT_final_lib import *

# entradas
Re = 10
Nx = 32
Ny = 32
Lx = 1
Ly = 1
vel_t = 1
tempo = 20
tol = 1.e-5

# velocidade na tampa
U = np.ones(Nx+1, dtype=float) # velocidade da tampa
U = U*vel_t
# matrizes
# pressão, u, v, Stream function
press, velh, velv, matriz , ite = Evolucao(U, Lx, Ly, Re, Nx, Ny, tempo, tol)

# ---------------------------------------------------------
# malha de plot

x, y = np.meshgrid(np.linspace(0, Lx, Nx+1), np.linspace(0, Ly, Ny+1))
# vetor velocidade
uplot, vplot = Vel_plot(Nx, Ny, velh, velv)

Plot(x, y, uplot, vplot, tempo, Re, Press_plot(press, Nx, Ny), color='yellow')
# Plot(x, y, uplot, vplot, tempo, Re, matriz, strem_function=True, color='black')

#---------------------------------------------------------
# verificacao com a literatura

# u_gabarito = np.array([0, -0.03854258, -0.0696238561, -0.096983962, -0.122721979, -0.147636199, 
#         -0.171260757, -0.191677043, -0.205164738, -0.205770198, -0.184928116, -0.1313892353, 
#         -0.031879308, 0.126912095, 0.354430364, 0.650529292])
# y_gabarito = np.linspace(0, 1, 16)
# u_atual = velh[16, 0:32:2]

# plt.plot(u_atual, y_gabarito)
# plt.plot(u_gabarito, y_gabarito, 'ro')
# plt.xlabel('u(m/s)')
# plt.ylabel('y(m)')

# plt.plot(y_gabarito, np.absolute(np.absolute(u_gabarito)-np.absolute(u_atual)))
# ---------------------------------------------------------


plt.show()

# erro_m_abs = np.max(np.absolute(np.absolute(u_gabarito)-np.absolute(u_atual)))
# erro_perc = np.absolute(u_gabarito[1:]-u_atual[1:])/u_gabarito[1:]*100

# print(np.max(erro_perc))

# print(ite)

import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constant Parameters
RC = 80 #kW
Tevap = 5 #Degree Celsius
Tcond = 45 #Degree Celsius
Twi = 30 #Water inlet temp; Degree Celsius
Twe = 35 #Water outlet temp; Degree Celsius
Do = 0.016 #Pipe outer diameter; m
Di = 0.014 #Pipe inner diameter; m
HRR = 1.27 #Heat Rejection Ratio; Qcond/Qevap
k_pipe = 390 #Thermal conductivity of copper; J/mK
N = 42 #number of pipes
Pass = 2 #number of pass
refrigerant = 'R134a' #refrigerant
V_flow = 13 #Number of vertical flow

# Iteration Variable
delta_T = 5 #Assumed delta T

T_array = np.arange(35.01,45.01,0.01) #Array of Condenser Temperature
T = np.flip(T_array) #Flipped Condenser Array

def Designer(Tevap, Tcond, Twi, Twe, Do, Di, HRR, k_pipe, N, Pass, V_flow, delta_T, refrigerant):
	'''
	This function computes the actual temperature difference and
	the condenser tube length given condenser temperature as input

	Parameters:
	Tevap - Evaporator Temperature
	Tcond - Condenser Temperature
	Twi - Water inlet Temperature
	Twe - Water outlet Temperature
	Do - Pipe outer diameter
	Di - Pipe inner diameter
	HRR - Heat Rejection Ratio
	k_pipe - Pipe thermal conductivity
	N - Number of pipes
	Pass - Number of Bends + 1
	V_flow - Number of fluid flow perpendicular to pipe
	delta_T - Temperature Difference
	refrigerant - Name of the Refrigerant

	Return:
	delta_T_prime - Converged Temperature Difference
	L_tube - Condenser Tube Length
	Pcond - Condenser Pressure
	'''

	# Refrigerant Side
	Tcond_K = Tcond +273.15 #Kelvin
	Pcond = CP.PropsSI('P','T',Tcond_K,'Q',0,refrigerant)/1000 #@Tcond and Q=0; kPa

	N_effective = (N/V_flow)
	k_ref = CP.PropsSI('CONDUCTIVITY','T',Tcond_K,'Q',0,refrigerant) #@Tcond and Q=0
	rho_ref = CP.PropsSI('D','T',Tcond_K,'Q',0,refrigerant) #@Tcond and Q=0
	hfg = CP.PropsSI('H','T',Tcond_K,'Q',1,refrigerant) - CP.PropsSI('H','T',Tcond_K,'Q',0,refrigerant) #J/kg
	mu_ref = CP.PropsSI('V','T',Tcond_K,'Q',0,refrigerant) #@Tcond and Q=0

	Numerator = (k_ref**3)*(rho_ref**2)*9.81*hfg
	Denominator = (N_effective*mu_ref*delta_T*Do)

	h_refrigerant = 0.725*((Numerator/Denominator)**0.25)

	# Conduction in pipe
	delta_x = (Do-Di)/2

	h_pipe = delta_x/k_pipe

	# Fouling Factor
	h_fouling = 0.000176

	# Convection in Water
	T_bulkmean = (Twi+Twe)/2 #Degree Celsius
	T_bulkmean_K = T_bulkmean + 273.15 #Kelvin
	P_water = 101325 #Pascal

	Cp_water = CP.PropsSI('C','T',T_bulkmean_K,'P',P_water,'water') #@Tbulkmean and P water
	k_water = CP.PropsSI('CONDUCTIVITY','T',T_bulkmean_K,'P',P_water,'water') #@Tbulkmean and P water
	mu_water = CP.PropsSI('V','T',T_bulkmean_K,'P',P_water,'water') #@Tbulkmean and P water
	rho_water = CP.PropsSI('D','T',T_bulkmean_K,'P',P_water,'water') #@Tbulkmean and P water


	Q_cond = HRR*RC*1000 #Heat Rejection
	m_water = Q_cond/(Cp_water*(Twe-Twi)) #Total mass flow rate

	m_pipe = m_water/(N/Pass) #mass flow per pipe

	U = m_pipe/(rho_water*(np.pi*(Di**2)*0.25)) #Flow rate/ Flow velocity
	h_water = Dittus_Boelter(Di,k_water,U,rho_water,mu_water,Cp_water) #Enthalpy Water

	# Heat calculation
	R_refrigerant = 1/(h_refrigerant*np.pi*Do) #Thermal Resistance for Refrigerant; 1/(hA)
	R_pipe = (h_pipe)/(np.pi*(0.5*(Do+Di))) #Thermal Resistance for Pipe; delta x/(kA)
	R_fouling = h_fouling/(np.pi*Di) #Thermal Resistance for Fouling Factor; 1/(hA)
	R_water = 1/(h_water*np.pi*Di) #Thermal Resistance for Water; 1/(hA)

	R_total = R_refrigerant+R_pipe+R_fouling+R_water #Totla Thermal Resistance

	LMTD = ((Tcond-Twe)-(Tcond-Twi))/np.log((Tcond-Twe)/(Tcond-Twi)) #Log Mean Temperature Difference

	L_total = (Q_cond*R_total)/LMTD #Total condenser length
	L_tube = L_total/N #Length per tube

	delta_T_prime = Q_cond/(h_refrigerant*np.pi*Do*L_tube*N) #New delta T

	# print to debug
	'''
	print(f'h refrigerant: {h_refrigerant}')
	print(f'h pipe: {h_pipe}')
	print(f'h fouling: {h_fouling}')
	print(f'h water: {h_water}')
	print(f'mass flow: {m_water}')
	print(f'Delta T: {delta_T_prime}')
	print(f'Total Length: {L_total}')
	print(f'Length per tube: {L_tube}\n')
	'''
	return delta_T_prime, L_tube, Pcond


def Dittus_Boelter(Di,k_water,U,rho,mu_water,Cp_water):
	'''
	This is the Dittus-Boelter Equation used to relate water enthalpy
	Nu = K*(Re**m)*(Pr**n)

	Parameters:
	Di - Pipe inner diameter
	k_water - Fluid thermal conductivity
	U - Fluid flow rate
	rho - Fluid Density
	mu_water - Fluid viscosity
	Cp_water - Constant pressure specific heat

	Return:
	h_water - Fluid enthalpy'''
	Re = (Di*U*rho/mu_water) #Reynold's Number
	Pr = ((Cp_water*mu_water)/k_water) #Prandlt Number
	Nu = 0.023*((Re**0.8)*(Pr*0.4)) #Nusselt Number
	return (k_water*Nu)/Di

def iterator(Tevap, Tcond, Twi, Twe, Do, Di, HRR, k_pipe, N, Pass, V_flow, delta_T, refrigerant):
	'''
	This function iterates until the temperature difference
	converge with 1% error
	'''
	Dt, L_tube, Pcond = Designer(Tevap, Tcond, Twi, Twe, Do, Di, HRR, k_pipe, N, Pass, V_flow, delta_T, refrigerant)
	tolerance = 0.01
	max_iter = 1000
	counter = 0
	while counter < max_iter:
		if abs(Dt-delta_T) > tolerance:
			counter += 1
			delta_T = Dt
			Dt, L_tube, Pcond = Designer(Tevap, Tcond, Twi, Twe, Do, Di, HRR, k_pipe, N, Pass, V_flow, delta_T, refrigerant)
		else:
			break
	return Pcond, L_tube, Dt

P = [] #List for Pressure
L = [] #List for Tube Lengths
DT =[] #list for temperature difference

for t in T:
	p, l, dt= iterator(Tevap, t, Twi, Twe, Do, Di, HRR, k_pipe, N, Pass, V_flow, delta_T, refrigerant)
	P.append(p)
	L.append(l)
	DT.append(dt)
title = refrigerant+' Condenser Pressure vs Condenser Tube Length' #Plot title

data = {'Condenser Temperature (C)':T, 'Condenser Pressure (kPa)':P, 'Condenser Tube Length (m)':L, 'Temperature Difference':DT}

df = pd.DataFrame(data)

# Exporting data to csv file
filename = refrigerant + ' Tube Length Data.csv' #CSV file name
df.to_csv(filename, index=False)


# Plotting the data
plt.plot(L,P)
plt.xlabel('Condenser Tube Length (m)')
plt.ylabel('Condenser Pressure (kPa)')
plt.title(title)
plt.show()

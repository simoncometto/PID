from PID import pid
import numpy as np
import matplotlib.pyplot as plt

dt = 0.05
f_sample = 1/dt
controlador_pid = pid(Kp=1, Ki=0.2, Kd=0.03)

f = np.logspace(-1,3, num=500)
modulo = np.array([])

'''
freq = 0.08
t = np.arange(0, 2*np.pi/freq, dt)
x = np.sin(freq*t)
y = np.array([])
for sample in x:
    c = controlador_pid(sample, dt=dt)
    y = np.append(y, c)

plt.step(t,y)
plt.step(t,x)
plt.legend(['PID','Setpoint'])
plt.show()'''

for freq in f:
    print(freq)
    t = np.arange(0, 2*np.pi/freq, dt)
    x = np.sin(freq*t)
    y = np.array([])
    for sample in x:
        c = controlador_pid(sample, dt=dt)
        y = np.append(y, c)

    f_i = np.sqrt(np.mean(y**2))
    modulo = np.append(modulo, f_i)

modulo = (20*np.log10(modulo/np.sqrt(2)))
plt.plot(f, modulo)
plt.xscale('log')
plt.ylim(-25,6)
plt.ylabel('Ganancia [dB]')
plt.xlabel('Frecuncia [Hz]')
plt.title('Respuesta en frecuencia PID (Kp=1,Ki=0.2,Kd=0.03) f_sample = 20Hz')
plt.axvline(x=20, color='k', ls='--')
plt.savefig('Freq resp P=1,I=0.2,D=0.03 20Hz.png', dpi=300)
plt.show()
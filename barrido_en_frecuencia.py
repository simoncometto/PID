from PID import pid
import numpy as np
import matplotlib.pyplot as plt

dt = 0.05
f_sample = 1/dt
controlador_pid = pid(Kp=1, Ki=1, Kd=0.1)

f = np.logspace(-2,1.6, num=300)
moduloy = np.array([])
modulox = np.array([])
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

    #f_i = np.sqrt(np.mean(y**2))
    f_i = np.mean(np.abs(y)) * (np.pi / 2)
    moduloy = np.append(moduloy, f_i)
    f_i = np.mean(np.abs(x)) * (np.pi / 2)
    modulox = np.append(modulox, f_i)

bode = (20*np.log10(moduloy/modulox))
plt.plot(f, bode)
plt.xscale('log')
#plt.ylim(0,30)
plt.ylabel('Ganancia [dB]')
plt.xlabel('Frecuncia [Hz]')
plt.title('Respuesta en frecuencia PID (Kp=1,Ki=1,Kd=0.1) f_sample = 20Hz')
plt.axvline(x=20, color='k', ls='--')
plt.savefig('Bode P=1,I=1,D=0.1 20Hz (2).png', dpi=300)
plt.show()
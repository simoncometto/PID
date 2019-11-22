import numpy as np
from time import time
import matplotlib.pyplot as plt

class pid:
    '''
        "Dsicret PID controller"
    '''

    def __init__(self,
                 Kp=1.0, Ki=0, Kd=0,
                 setpoint=0,
                 output_limits=(None, None),
                 integrate_methode = 'trapezoidal'):

        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self._min_output, self._max_output = output_limits
        self.integrate_methode = integrate_methode

        self.__last_t = time()
        self.__last_error = np.zeros(3, dtype=float)
        self.__last_I = 0
        return

    def __call__(self, measurment, dt=None):
        error = self.setpoint - measurment
        self.__last_error = np.roll(self.__last_error, -1)
        self.__last_error[-1] = error

        if dt == None:
            now = time()
            dt = now - self.__last_t
            self.__last_t = now

        #Termino Propocional -------------------------------------------
        P = error * self.Kp

        #Termino Integral ----------------------------------------------
        I = self.__last_I + self.Ki * self.__integrate(dt)
        self.__last_I = I
        #Termino Diferencial -------------------------------------------
        D = self.Kd * self.__differentiate(dt)

        output = P + I + D
        return output

    def set_parameters(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        return

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint
        return

    def __integrate(self, dt):
        if self.integrate_methode == 'rectangular':
            I = self.__last_error[-1] * dt
        elif self.integrate_methode == 'trapezoidal':
            I = dt * 0.5 * (self.__last_error[-2] + self.__last_error[-1])
        elif self.integrate_methode == 'simpson':
            #No implementado
            I = 0
        else:
            disp = 'No se reconoce a :' + self.integrate_methode + 'como un método de integración'
            raise ValueError(disp)
        return I

    def __differentiate(self, dt):
        D = (self.__last_error[-1] - self.__last_error[-2]) / dt
        return D


if __name__ == '__main__':
    dt = 0.5
    t_step = 4 + dt
    t_end = 11

    t_init = np.arange(0, t_step, dt)
    t = np.arange(t_step, t_end,dt)

    x_init = np.zeros(len(t_init))
    x = np.ones(len(t))

    t = np.concatenate((t_init, t))
    x = np.concatenate((x_init, x))
    #x = np.zeros(len(t))
    #x[5] = 1
    y = np.empty(len(t))

    PID = pid(Kp=1, Ki=0.1, Kd=0.3)

    #print(x)
    for i in range(len(t)):
        PID.set_setpoint(x[i])
        y[i] = PID(0, dt=dt)

    plt.step(t,y) #t,x)
    plt.step(t,x)
    plt.legend(['PID','Setpoint'])
    #plt.savefig('PID 1,0.2,0.03 impulse.png', dpi=300)
    plt.show()

    '''
    y = np.empty(len(t))

    measurment = 0   #Lazo abierto

    PID = pid(Kp=1.0, Ki=0.2, Kd=0.005)

    f = np.arange(0,(100)*100,100)
    P = np.empty(100, dtype=float)

    for j in range(100):
        freq = j * 100
        x = np.sin(2 * np.pi * freq * t)

        for i in range(len(t)):
            PID.set_setpoint(x[i])
            y[i] = PID(measurment, dt=dt)

        P[j] = np.sqrt(np.mean(y**2))


#plt.plot(t, y)
print(len(f))
print(len(P))
plt.plot(f, P)
plt.show()'''
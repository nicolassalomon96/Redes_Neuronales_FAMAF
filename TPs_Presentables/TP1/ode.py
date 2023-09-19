import numpy as np

def euler(f, x, t, h, p):
    """
    Realiza un paso de integración utilizando el método de Euler.

    Args:
        f: Función que caracteriza el lado derecho de la ODE. Debe tomar los argumentos (x, t, p) y devolver un vector.
        x: Valor inicial.
        t: Tiempo actual.
        h: Paso de integración.
        p: Vector de parámetros.

    Returns:
        Nuevo valor de x después de un paso de integración.
    """
    # Calcula el incremento utilizando la función f en el punto actual
    increment = f(x, t, p) * h
    
    # Calcula el nuevo valor de x después del paso de integración
    new_x = x + increment
          
    return new_x

def integrador_ode(m, f, x0, a, b, k, p, c=None):
    """
    Integra una ODE utilizando el método dado para computar sucesivos pasos de integración.

    Args:
        m: Función que aproxima el paso de integración.
        f: Función que caracteriza el lado derecho de la ODE. Debe tomar los argumentos (x, t, p) y devolver un vector.
        x0: Condición inicial.
        a: Tiempo inicial.
        b: Tiempo final.
        k: Número de pasos de integración.
        p: Vector de parámetros.
        c: Función condicionante opcional que permite intervenir en el cómputo de la trayectoria.

    Returns:
        Lista de valores de x en cada paso de integración.
    """
    x_values = [x0]
    t = a
    h = (b - a) / k
    time_values = np.arange(a, b + h, h)

    for _ in range(k):
        if c is None:
            x_new = m(f, x_values[-1], t, h, p)
            x_values.append(x_new)
        else:
            x_values.append(c(m(f, x_values[-1], t, h, p), t, p))
        t += h

    return time_values, np.array(x_values)

def rk4(f, x, t, step_size, parameters):
    def runge_kutta_4(f, x, t, h, *args):
        k1 = h * f(x, t, *args)
        k2 = h * f(x + k1 / 2, t + h / 2, *args)
        k3 = h * f(x + k2 / 2, t + h / 2, *args)
        k4 = h * f(x + k3, t + h, *args)
        return x + (k1 + 2*k2 + 2*k3 + k4) / 6

    # Solución numérica utilizando el método de RK4
    new_x = runge_kutta_4(f, x, t, step_size, parameters)

    return new_x
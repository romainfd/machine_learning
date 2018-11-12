from numpy import *
from math import e
from math import pow

def sigmoid(z):
    # Computes the sigmoid of z.
    g = zeros_like(z)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the sigmoid function as given in the
    # assignment (and use it to *replace* the line above).
    g = 1 / (1 + exp(-z))

    # we prevent overflow
    g = where(g < 1e-4, 1e-4, g)
    g = where(g > 1 - 1e-4, 1 - 1e-4, g)

    # =============================================================
    return g


from numpy import *

def get_circle(center=(0.0, 0.0), r=1.0, numpoints=150):  
    # use polar coordinates to get uniformly distributed points  
    step = pi * 2.0 / numpoints  
    t = arange(0, pi * 2.0, step)  
    x = center[0] + r * cos(t)  
    y = center[1] + r * sin(t)  
    return column_stack((x, y))
    
def get_noise(stddev=0.2, numpoints=150):  
    # 2d gaussian random noise  
    x = random.normal(0, stddev, numpoints)  
    y = random.normal(0, stddev, numpoints)  
    return column_stack((x, y))    

def generateData():
    circles = []  
    for radius in (1.0, 3.2, 5.0):  
        circles.append(get_circle(r=radius) + get_noise())  

    points = vstack(circles)
    return points

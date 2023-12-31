# draw graphs to show the number of order combinations
import matplotlib.pyplot as plt
import numpy as np

# num of variables = N
# num of objects = O
# num of constraints = C

# functions for order combinations
def factorial(n):
    if n == 0:
        return 1
    return n*factorial(n-1)

# when variables are distributed evenly to o objects
def orderWithObjects(n, o):
    if n < o:
        return orderWithObjects(n, n)
    varNumInObjects = n//o
    upperOrder = factorial(o)
    lowerOrder = factorial(varNumInObjects)**o
    return upperOrder * lowerOrder

# when variables are constrained by a < b, (c constraints are introduced)
def orderWithConstraints(n, c):
    return factorial(n)/(2**c)


# functions for possible structure combinations
def numStructure(n):
    orderNum = factorial(n)
    for i in range(1, n+1):
        orderNum *= (2**i)
    return orderNum

def numStructureWithObjects(n, o):
    varNumInObjects = n//o
    wideNum = numStructure(o)
    inObjNum = numStructure(varNumInObjects)**o
    return wideNum * inObjNum

def numStructureWithConstraints(n, c):
    orderNum = orderWithConstraints(n, c)
    for i in range(1, n+1):
        orderNum *= (2**i)
    return orderNum / (2**c)

# draw graphs
def drawOrder(n, o, c):
    # order combinations
    x = np.arange(1, n+1, 1)
    objNums = np.arange(2, o+1, 1)
    constNums = np.arange(1, c+1, 1)

    Vbase = np.vectorize(factorial)
    Vobj = np.vectorize(orderWithObjects)
    Vconst = np.vectorize(orderWithConstraints)

    base = Vbase(x)
    objs = [Vobj(x, oi) for oi in objNums]
    consts = [Vconst(x, ci) for ci in constNums]
    print(base)
    print(objs)
    print(consts)

    for i in range(len(constNums)):
        plt.plot(x, consts[i], label=f"constraints={constNums[i]}")
    plt.plot(x, base, label="base", color="black", linewidth=3)
    plt.legend()
    plt.yscale("log")
    plt.show()

if __name__ == "__main__":
    drawOrder(10, 5, 9)

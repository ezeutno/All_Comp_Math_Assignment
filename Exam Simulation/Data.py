from sympy import *
import numpy as np
import math as m
import matplotlib.pyplot as plt

#Latihan Exam BiMay

#Number 1

def fact(num):
    res = 1
    for i in range(1,num+1):
        res *= i
    return res

def Maclaurin(x,n):
    res = 0
    if type(n) is np.ndarray:
        data = []
        for a in np.nditer(n):
            for k in range(a):
                res += ((-1)**k/fact(2*k+1))*(x**(2*k+1))
            data.append(res)
            res = 0
        return np.array(data)
    else:
        for k in range(n):
            res += ((-1)**k/fact(2*k+1))*(x**(2*k+1))
        return res
        
            
print("Number 1")
maclaurinRes = Maclaurin(np.pi/3,100)
exactValueSIN = 0.5*m.sqrt(3)
print("SIN 60 : ",maclaurinRes,"|",exactValueSIN)
print("Margin of Error : ", (abs(exactValueSIN - maclaurinRes)/exactValueSIN)*100,"%")

#Number 2
print("Number 2")
n = np.array([5,10,20,50,100])
multiMacRes = abs(np.ones(5)*exactValueSIN - Maclaurin(np.pi/3,n))
plt.plot(n,multiMacRes,label = "ERROR")
plt.legend()
plt.show()

#Number 3
print("Number 3")
x = symbols('x')
expression = sin(x)/(1+x**2)
resInter = integrate(expression,(x,0,pi/2)).evalf()
print("Result : ",resInter)

#Number 4
print("Number 4")
f = lambda x: np.sin(x)/(1+x**2)
def simpIntegrat(n, fx=f, a=0, b=np.pi/2):
    if type(n) is np.ndarray:
        data = []
        for h in np.nditer(n):
            x = np.linspace(a,b,h+1)
            w = np.ones(h+1)
            w[1:-1:2] = 4
            w[2:-2:2] = 2
            data.append(sum(w*fx(x))*abs(x[0]-x[1])/3)
        return np.array(data)
    else:
        x = np.linspace(a,b,n+1)
        w = np.ones(n+1)
        w[1:-1:2] = 4
        w[2:-2:2] = 2
        return sum(w*fx(x))*abs(x[0]-x[1])/3
n = np.array([2,5,10,50,100,200,1000,10000,20000,30000,35000])
res = simpIntegrat(n)
print("n".ljust(5),"estimassion".ljust(23),"error%")
print("="*50)
for i in range(len(n)):
    print(repr(n[i]).ljust(5),repr(res[i]).ljust(23),(abs(res[i]-resInter)/resInter)*100)
print("="*50)

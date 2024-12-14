import math
x=3
f=x*(math.e)**(-x)+math.cos(x)
f1=x * (math.e ** (-x)) - math.e ** (-x) + math.sin(x)
f2=x * (math.e ** (-x)) + 2 * math.sin(x) - math.cos(x)
dx=-f1/f2
i=0
while abs(f1)>1e-10:
    i=i+1
    x=x+0.1*dx
    f=x*(math.e)**(-x)+math.cos(x)
    f1=x * (math.e ** (-x)) - math.e ** (-x) + math.sin(x)
    f2=x * (math.e ** (-x)) + 2 * math.sin(x) - math.cos(x)
    dx=-f1/f2   
    print(i,'x:',x,'f:',f,'f1:',f1,'f2:',f2)



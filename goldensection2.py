import math
import numpy as np

def f(x,p,s):
    x=x+s*p
    f = 3 + (x[0]-1.5*x[1])**2 + (x[1]-2)**2
    return f
    

salt=-10
sust=40
ds = 0.0001 
alpha = (1+math.sqrt(5))/2
tau=1-1/alpha
epsilon= ds/(sust-salt)
N= round(-2.078*math.log(epsilon))

k=0
s1=salt+tau*(sust-salt); f1=f(s1)
s2=sust-tau*(sust-salt); f2=f(s2)

print(k,s1,s2,f1,f2)

while abs(s1-s2)>ds:
    k+=1
    if f1>f2:
        salt=1*s1; s1=1*s2;  f1=1*f2;
        s2=sust-tau*(sust-salt); f2=f(s2);
    else:
        sust=1*s2; s2=1*s1; f2=1*f1;
        s1=salt+tau*(sust-salt); f1=f(s1)
    print(k,s1,s2,f1,f2)

s=np.mean([s1,s2])
print(s)

## üst alt değerler belirledikten sonra tauyu 1-altınOran şeklinde tanımladık.
# burada dx değerimiz çözümün hata payını belirtiyor. 
# bu dx i kullanarak daha çözüme başlamadan N ile toplam iterasyon sayımızı bulduk
# Hata payına kadar devam eden bir döngüyü while içine yazarak güncellemelere başladık ve bitirdik.
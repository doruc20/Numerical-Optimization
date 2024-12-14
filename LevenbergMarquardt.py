import numpy as np
import math
from ornekFonksiyon2 import f,error,jacobian

def GSf(sk,xk,pk):
    sonuc = f(xk+sk*pk)
    return sonuc

def GS(xk,pk):
    salt=0
    sust=1
    ds=0.0001
    alpha = (1+math.sqrt(5))/2               #Burada Golden Section fonksiyonuna xk -> güncel bulunduğum noktayı ve
    tau=1-1/alpha                            # mevcut noktanın -gradient değerini gönderiyorum.
    epsilon=ds/(sust-salt)                   # Golden section xk+sk*pk değerini f fonksiyonuna gönderip
    N=round(-2.078*math.log(epsilon))        #sonuçları karşılaştırıyor. En adım aralığını döndürüyor.
    k=0
    s1=salt+tau*(sust-salt); f1=GSf(s1, xk, pk)
    s2=sust-tau*(sust-salt); f2=GSf(s2, xk, pk);
    while abs(s1-s2)>ds:
        k+=1
        if f1>f2:
            salt = 1*s1;    s1= 1*s2;    f1= 1*f2;
            s2 = sust - tau*(sust-salt); f2 = GSf(s2,xk,pk);
        else:
            sust= 1*s2;  s2= 1*s1;    f2= 1*f1;
            s1 = salt+tau*(sust-salt); f1 = GSf(s1,xk,pk);
    s = np.mean([s1,s2])
    return s

#--------------------------------------------------------------------------
MaxIter=500
epsilon1 =1e-9                  #SONLANDIRMA KRİTERLERİ
epsilon2 =1e-9
epsilon3 =1e-9
mumax=1e99
#--------------------------------------------------------------------------
x1=[np.random.random()]
x2=[np.random.random()]                          # xk vektörü = [x1,x2]=[0,0] ->>> Başlangıç Noktam
xk= np.array([x1[0],x2[0]])


ek=error(xk)
fk=sum(ek**2)
mu=1; muscal=10; I= np.identity(2);

#---------------------------------------------------------------------------

k=0; C1=True; C2=True;  C3=True; C4=True;

while C1 & C2 & C3 & C4:            #Ana merkez
    ek=error(xk)
    Jk=jacobian(xk)
    gk=np.array((2*Jk.transpose().dot(ek)).tolist()[0])
    Hk=2*Jk.transpose().dot(Jk)
    #sk=GS(xk,pk)   #golden section gönderilen xk,pk değerlerine göre en uygun adım aralığını sk ya atıyor.(Bu algoritma için çok da gerekli değilmiş.)
    sk=1
    loop=True
    while loop:
        zk= -np.linalg.inv(Hk+mu*I).dot(gk)
        zk= np.array(zk.tolist())[0]
        ez = error(xk+sk*zk)
        fz= sum(ez**2)
        if fz<fk:
            pk=1*zk
            mu=mu/muscal
            k+=1
            xk= xk+sk*pk
            x1.append(xk[0]) 
            x2.append(xk[1])
            loop=False
            print('k:',k,'x1',format(xk[0],'f'),'x2:',format(xk[1],'f'))
        else:
            mu=mu*muscal
            if mu>mumax:
                loop=False
                C2=False
    #---------------------------------------
    C1=k<MaxIter
    C2=epsilon1<abs(f(xk)-f(xk+sk*pk)) 
    C3=epsilon2<np.linalg.norm(sk*pk)               #Bu koşullar sağlandığı sürece döngü devam edecek..
    C4=epsilon3<np.linalg.norm(gk)
    #------------------------------------
print(C1,C2,C3,C4)  #hangi koşuldan patladığını inceliyoruz.(opsiyonel)
#------------------------------------

import matplotlib.pyplot as plt
plt.plot(x1,x2,color='darkred', linestyle='solid',linewidth = 1, marker= 'o')
plt.xlabel('x1')                #ÇİZİM    
plt.ylabel('x2')
plt.title('Steepest Descent Algoritması',fontstyle='italic')
plt.grid(color='green',linestyle='--',linewidth=0.1)
plt.show()
#-----------------------------------------------------------------------------
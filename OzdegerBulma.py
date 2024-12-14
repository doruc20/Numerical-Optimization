import numpy as np
from numpy.linalg import eig
#python sayisal_opt_sinav.py şeklinde çalıştır

A= np.array([[2,-3],[-3,0]]) #hesian matrisini tanımladık.

ozdeger,ozvektor = eig(A) #ozdeger bulundu.


i=0

print("Ozdegerler:",ozdeger)
if ozdeger[i] >0 and ozdeger[i+1] > 0:
    print("Ozdegerler pozitif. Yerel minimum noktasıdır.")
        
elif ozdeger[i] <0 and ozdeger[i+1] <0:
    print("Ozdegerler pozitif değil. Yerel maximum noktasıdır.")
        
else:
    print("Ozdegerler farklı işarete sahip . Semer noktasıdır.")
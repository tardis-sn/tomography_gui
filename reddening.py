#!/usr/bin/env python
from __future__ import print_function
import numpy as np

def calculate_reddening_correction(Wavelengths,L,E_BV):
    LRed=[]
    mask = []
    Rv=3.1
    A_V=Rv*E_BV
    N=len(Wavelengths)
    for i in xrange(N):
        X=1/(Wavelengths[i]*1e-4)
        mask.append(1)
        if X>=0.3 and X<=1.1:
            a=0.574*(X**1.61)
            b=-0.527*(X**1.61)
        elif X>1.1 and X<3.3:
            y=X-1.82
            a=1+0.17699*y-0.50447*(y**2)-0.02427*(y**3)+0.72085*(y**4)+0.01979*(y**5)-0.7753*(y**6)+0.32999*(y**7)
            b=1.41338*y+2.28305*(y**2)+1.07233*(y**3)-5.38434*(y**4)-0.62251*(y**5)+5.3026*(y**6)-2.09002*(y**7)
        elif X>=3.3 and X<=8:
            if X<5.9:
                Fa=Fb=0
            else:
                Fa=-0.04473*((X-5.9)**2)-0.009779*((X-5.9)**3)
                Fb=0.213*((X-5.9)**2)+0.1207*((X-5.9)**3)
            a=1.752-0.316*X-0.104/((X-4.67)**2+0.341)+Fa
            b=-3.090+1.825*X+1.206/((X-4.62)**2+0.263)+Fb
        else:
            #raise Exception("Error: reddening only works for 0.3 <= X <= 8")
            mask[-1] = 0
            continue
        A_W=(a+b/Rv)*A_V
        Red=10**(-0.4*A_W)
        LRed.append(Red)

    return np.array(LRed), np.array(mask, dtype=np.bool)

#if __name__== "__main__":
#    calculate_reddening_correction(self.old_data[:,0], self.old_data[:,1],self.reddening)




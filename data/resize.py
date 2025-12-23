import numpy as np

Ntomo = 10
Ncl = 20

# size = int((Ntomo * (Ntomo + 1)/2 + Ntomo + Ntomo * Ntomo ) * Ncl)
size = 2200
print(size)
datav = np.zeros(size)
mask = np.zeros(size)

index = np.arange(size)
dv_KL = np.loadtxt('Roman_Ntomo10_KL.datavector', usecols=(1))
size_KL = len(dv_KL)
datav[:size_KL] = dv_KL
mask[:size_KL] = 1

np.savetxt('roman_kl.datavector', np.column_stack((index, datav)), fmt='%d %1.6e')
np.savetxt('roman_kl.mask', np.column_stack((index, mask)), fmt='%d %1.1f')

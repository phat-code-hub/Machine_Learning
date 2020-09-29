import numpy as np
A=np.array([15,16,18,19,22,24,29,30,34])
A=np.ones(6)
A=np.array([10,10,15,10,10,16,10])
A=A+4
mean=np.mean(A).round(2)
print('Mean:',mean)
print('Median: ',np.median(A))
print("25% :",np.percentile(A,25))
print("50% :",np.percentile(A,50))
print("75% :",np.percentile(A,75))
B=abs(np.subtract(A,np.mean(A)))
print(B)
C=np.sum(B**2)/np.size(B)
print("Variance: " ,np.var(A).round(2))
D=np.sqrt(C).round(2)
print('Standard deviation: ',np.std(A).round(2))

print(np.mean([0.7,0.6,0.6,0.8,0.8]))
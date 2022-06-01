import pandas as pd
import math 
import numpy as np
path='D:\\Magnetic simulation\\python\\core loss\\data\\N87\\N87-Triangular(D=0.1).csv'
temp1=pd.read_csv(path,header=0)
data1=np.array(temp1)
path='D:\\Magnetic simulation\\python\\core loss\\data\\N87\\N87-Triangular(D=0.5).csv'
temp2=pd.read_csv(path,header=0)
data2=np.array(temp2)
path='D:\\Magnetic simulation\\python\\core loss\\data\\N87\\N87-Triangular(D=0.9).csv'
temp3=pd.read_csv(path,header=0)
data3=np.array(temp3)
data=np.vstack([data1,data2])
data=np.vstack([data,data3])
fs=data[:,0]
Bmax=data[:,1]
loss=data[:,2]/1000
D=data[:,3]
'''
Ki=0.1907
alpha=0.87447
beta=1.87447
'''
Ki=0.0123
alpha=1.6159
beta=2.4982
################################
cal_loss_SE=np.zeros(np.size(fs))
for i in range(np.size(fs)):
    cal_loss_SE[i]=Ki*(fs[i]**alpha)*(Bmax[i]**beta)/1000

cal_loss_iGSE=np.zeros(np.size(fs))
for i in range(np.size(fs)):
    cal_loss_iGSE[i]=Ki*((2*Bmax[i])**(beta-alpha))*(D[i]*((2*Bmax[i]/D[i]*fs[i])**alpha)+(1-D[i])*((2*Bmax[i]/(1-D[i])*fs[i])**alpha))/1000
################################

data=pd.DataFrame({'frequency':fs,'peak flux density':Bmax,'duty cycle':D,'loss calculted by SE':cal_loss_SE,\
    'loss calculted by iGSE':cal_loss_iGSE,'Measured loss':loss,'error for SE':(loss-cal_loss_SE)/loss,\
        'error for iGSE':(loss-cal_loss_iGSE)/loss,'loss error percent by iGSE':loss/cal_loss_iGSE})

data.to_csv('D:\\Magnetic simulation\\python\\core loss\\programme\\parallel(SE)\\data(N87).csv')



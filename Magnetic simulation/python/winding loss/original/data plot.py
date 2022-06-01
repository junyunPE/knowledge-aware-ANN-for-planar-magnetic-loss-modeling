import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


path='D:\\file\\ECCE\\Magnetic simulation\\python\\winding loss\\original\\data_eval.csv'
temp=pd.read_csv(path,header=0)
temp=np.array(temp)
data=temp[:,8]
frequency1=[]
frequency2=[]
frequency3=[]
frequency4=[]
i=0
while i<len(data):
    if  0<=data[i]<25:
        frequency1.append(i)
        i=i+1
    elif 25<=data[i]<100:
        frequency2.append(i)
        i=i+1
    elif 100<=data[i]<500:
        frequency3.append(i)
        i=i+1
    else:
        frequency4.append(i)
        i=i+1

index=['0~25','25~100','100~500','500~2700']
Num=len(frequency1)+len(frequency2)+len(frequency3)+len(frequency4)
values=[len(frequency1)/Num*100,len(frequency2)/Num*100,len(frequency3)/Num*100,len(frequency4)/Num*100]
plt.bar(index,values)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Error distribution[%]',fontsize=25)
plt.ylabel('Distribution rate[%]',fontsize=25)
plt.ylim([0,90])
plt.tight_layout()

for a,b in zip(index,values):
 plt.text(a, b-0.3,'%.2f'%b+'%', ha = 'center',va = 'bottom',fontsize=20)
plt.show()









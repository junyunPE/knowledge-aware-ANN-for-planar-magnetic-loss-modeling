import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


path='D:\\file\\ECCE\\Magnetic simulation\\python\\winding loss\\data_eval.csv'
temp=pd.read_csv(path,header=0)
temp=np.array(temp)
data=temp[:,9]
frequency1=[]
frequency2=[]
frequency3=[]
frequency4=[]
frequency5=[]
i=0
while i<len(data):
    if  0<=data[i]<3:
        frequency1.append(i)
        i=i+1
    elif 3<=data[i]<6:
        frequency2.append(i)
        i=i+1
    elif 6<=data[i]<9:
        frequency3.append(i)
        i=i+1
    else:
        frequency4.append(i)
        i=i+1

index=['0~3','3~6','6~9','9~12']
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

data=temp[:,10]
frequency1=[]
frequency2=[]
frequency3=[]
frequency4=[]
frequency5=[]
i=0
while i<len(data):
    if  0<=data[i]<7:
        frequency1.append(i)
        i=i+1
    elif 7<=data[i]<14:
        frequency2.append(i)
        i=i+1
    elif 14<=data[i]<21:
        frequency3.append(i)
        i=i+1
    else:
        frequency4.append(i)
        i=i+1

index=['0~7','7~14','14~21','21~28']
Num=len(frequency1)+len(frequency2)+len(frequency3)+len(frequency4)
values=[len(frequency1)/Num*100,len(frequency2)/Num*100,len(frequency3)/Num*100,len(frequency4)/Num*100]
plt.bar(index,values)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Error distribution[%]',fontsize=25)
plt.ylabel('Distribution rate[%]',fontsize=25)
plt.ylim([0,50])
plt.tight_layout()

for a,b in zip(index,values):
 plt.text(a, b-0.3,'%.2f'%b+'%', ha = 'center',va = 'bottom',fontsize=20)
plt.show()








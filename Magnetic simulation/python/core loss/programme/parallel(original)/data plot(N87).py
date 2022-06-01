import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


path='D:\\file\\ECCE\\Magnetic simulation\\python\\core loss\\programme\\parallel(original)\\data_val(N87200).csv'
temp=pd.read_csv(path,header=0)
temp=np.array(temp)
data=temp[:,6]
frequency1=[]
frequency2=[]
frequency3=[]
frequency4=[]
frequency5=[]
i=0
while i<len(data):
    if  0<=data[i]<50:
        frequency1.append(i)
        i=i+1
    elif 50<=data[i]<100:
        frequency2.append(i)
        i=i+1
    elif 100<=data[i]<150:
        frequency3.append(i)
        i=i+1
    else:
        frequency4.append(i)
        i=i+1

index=['0~50','50~100','100~150','150~1500']
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



temp=pd.read_csv(path,header=0)
temp=np.array(temp)
loss=temp[:,5]
loss=(loss-np.min(loss))/(np.max(loss)-np.min(loss))
error=temp[:,6]


plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Normalized core loss',fontsize=25)
plt.ylabel('Error[%]',fontsize=25)
plt.scatter(loss, error)
plt.tight_layout()
plt.show()





'''
matplotlib.rcParams['font.family']='SimHei'

names=np.array(['0%~5%','5%~10%','15%~20%','20%~25%'])
frequency=np.array([len(frequency1),len(frequency2),len(frequency3),len(frequency4)])
rate=frequency/np.sum(frequency)
explode=np.zeros((len(frequency)))
explode[0]=0.1

colors = ['red','blue','yellow','green']

plt.figure(figsize=(20, 6.5))
patches,l_text,p_text=plt.pie(rate,explode=explode,labels=rate,autopct='%.2f%%',colors=colors)


plt.legend(['0%~5%','5%~10%','15%~20%','20%~25%'])
plt.legend(loc='center right')
plt.title('Error distribution')
plt.axis('equal')


for t in p_text:
    t.set_size(30)

for t in l_text:
    t.set_size(20)
plt.show()

'''


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

path='D:\\file\\ECCE\\Magnetic simulation\\python\\core loss\\programme\\parallel(SE)\\data(3C94).csv'
temp=pd.read_csv(path,header=0)
temp=np.array(temp)
data=abs(temp[:,7]*100)
frequency1=[]
frequency2=[]
frequency3=[]
frequency4=[]
frequency5=[]
i=0
while i<len(data):
    if  90<=data[i]<92:
        frequency1.append(i)
        i=i+1
    elif 92<=data[i]<94:
        frequency2.append(i)
        i=i+1
    elif 94<=data[i]<96:
        frequency3.append(i)
        i=i+1
    else:
        frequency4.append(i)
        i=i+1
Num=len(frequency1)+len(frequency2)+len(frequency3)+len(frequency4)
values=[len(frequency1)/Num*100,len(frequency2)/Num*100,len(frequency3)/Num*100,len(frequency4)/Num*100]

path='D:\\file\\ECCE\\Magnetic simulation\\python\\core loss\\programme\\parallel(SE)\\data(N87).csv'
temp=pd.read_csv(path,header=0)
temp=np.array(temp)
data=abs(temp[:,7]*100)
freq1=[]
freq2=[]
freq3=[]
freq4=[]
i=0
while i<len(data):
    if  90<=data[i]<92:
        freq1.append(i)
        i=i+1
    elif 92<=data[i]<94:
        freq2.append(i)
        i=i+1
    elif 94<=data[i]<96:
        freq3.append(i)
        i=i+1
    else:
        freq4.append(i)
        i=i+1
Num=len(freq1)+len(freq2)+len(freq3)+len(freq4)
values2=[len(freq1)/Num*100,len(freq2)/Num*100,len(freq3)/Num*100,len(freq4)/Num*100]

index=['90~92','92~94','94~96','96~100']


x = np.arange(len(index))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, values, width, label='3C94')
rects2 = ax.bar(x + width/2, values2, width, label='N87')
plt.rcParams.update({'font.size': 20})
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Distribution rate[%]',fontsize=25)
ax.set_xlabel('Error distribution[%]',fontsize=25)
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
ax.set_xticks(x)
ax.set_xticklabels(index)
ax.legend()



plt.ylim([0,90])

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}%'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=15)


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()












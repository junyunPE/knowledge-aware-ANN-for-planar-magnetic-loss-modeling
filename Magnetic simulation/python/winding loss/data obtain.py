import pandas as pd
import math 
import numpy as np

## file read
frequency=list()
acloss=list()
dcloss=list()
airgap=list()
dis_hor=list()
dis_ver=list()
thickness=list()
current=list()
################################
path='D:\\file\\ECCE\\Magnetic simulation\\python\\winding loss\\loss.txt'
path1='D:\\file\\ECCE\\Magnetic simulation\\python\\winding loss\\loss2.txt'
fhand=open(path)
for line in fhand:
    line=line.rstrip()
    if '#' in line:
        continue
    p1=line.find('.')
    p2=line.find('.',p1+1)
    acloss.append(float(line[p2-1:]))

fhand=open(path)
for line in fhand:
    line=line.rstrip()
    if '#' in line:
        continue
    p1=line.find('.')
    p2=line.find('.',p1+1)
    frequency.append(float(line[:p2-1]))
    
fhand=open(path)
for line in fhand:
    line=line.rstrip()
    if not 'Parameters' in line:
        continue
    p1=line.find('airgap')
    p2=line.find(';',p1+1)
    airgap.append(float(line[p1+7:p2]))
    p1=line.find('dis_hor') 
    p2=line.find(';',p1+1)
    dis_hor.append(float(line[p1+8:p2]))
    p1=line.find('thickness') 
    p2=line.find(';',p1+1)
    thickness.append(float(line[p1+10:p2]))
    p1=line.find('dis_ver') 
    p2=line.find(';',p1+1)
    dis_ver.append(float(line[p1+8:p2]))
    p1=line.find('current') 
    p2=line.find(';',p1+1)
    current.append(float(line[p1+8:p2]))

fhand=open(path1)
index=0
while index<len(thickness):
    if (abs(thickness[index]-0.035)<=0.01)&(abs(current[index]-1)<=0.01):
        dcloss.append(0.0099988998099160)
    elif (abs(thickness[index]-0.07)<=0.01)&(abs(current[index]-1)<=0.01):
        dcloss.append(0.0050069301825926)
    elif (abs(thickness[index]-0.105)<=0.01)&(abs(current[index]-1)<=0.01):
        dcloss.append(0.0033373675533874)
    ##
    elif (abs(thickness[index]-0.035)<=0.01)&(abs(current[index]-3.25)<=0.01):
        dcloss.append(0.10561337924336)
    elif (abs(thickness[index]-0.07)<=0.01)&(abs(current[index]-3.25)<=0.01):
        dcloss.append(0.052885700053234)
    elif (abs(thickness[index]-0.105)<=0.01)&(abs(current[index]-3.25)<=0.01):
        dcloss.append(0.035250944782574)
    ##
    elif (abs(thickness[index]-0.035)<=0.01)&(abs(current[index]-5.5)<=0.01):
        dcloss.append(0.30246671925195)
    elif (abs(thickness[index]-0.07)<=0.01)&(abs(current[index]-5.5)<=0.01):
        dcloss.append(0.15145963802422)  
    elif (abs(thickness[index]-0.105)<=0.01)&(abs(current[index]-5.5)<=0.01):
        dcloss.append(0.10095536848986)  
    ##
    elif (abs(thickness[index]-0.035)<=0.01)&(abs(current[index]-7.75)<=0.01):
        dcloss.append(0.60055891983758)
    elif (abs(thickness[index]-0.07)<=0.01)&(abs(current[index]-7.75)<=0.01):
        dcloss.append(0.30072874409134)  
    elif (abs(thickness[index]-0.105)<=0.01)&(abs(current[index]-7.75)<=0.01):
        dcloss.append(0.20045063867472) 
    ##
    elif (abs(thickness[index]-0.035)<=0.01)&(abs(current[index]-10)<=0.01):
        dcloss.append(0.99988998099252)
    elif (abs(thickness[index]-0.07)<=0.01)&(abs(current[index]-10)<=0.01):
        dcloss.append(0.50069301825825)  
    else:
        dcloss.append(0.33379452968902)
    index=index+1

acloss=np.array(acloss)
dcloss=np.array(dcloss)
airgap=np.array(airgap)/1000
frequency=np.array(frequency)*1000
thickness=np.array(thickness)/1000
dis_ver=np.array(dis_ver)/1000
current=np.array(current)

#dowell model calculation
def sd(fs):   #skin depth calculation
    mu=1    #relative permeability
    mu0=4*math.pi*1e-7  #permeability of vacuum
    p=1/58000000    #electrical resistance
    temp=np.zeros(np.size(fs))
    for i in range(np.size(fs)):
        temp[i]=math.sqrt(p/(math.pi*fs[i]*mu*mu0))
    return temp

ratio1=np.zeros(np.size(frequency))
for i in range(np.size(frequency)):
    ratio1[i]=acloss[i]/dcloss[i]

sd=sd(frequency)
N=1
ratio2=np.zeros(np.size(frequency))
for i in range(np.size(frequency)):
    temp=thickness[i]/sd[i]
    ratio2[i]=temp*((math.sinh(2*temp)+math.sin(2*temp))/(math.cosh(2*temp)-math.cos(2*temp))+\
    2*(N*N-1)/3*(math.sinh(temp)-math.sin(temp))/(math.cosh(temp)+math.cos(temp)))
################################

print(np.size(frequency))
print(np.size(acloss))
print(np.size(dis_ver))

data=pd.DataFrame({'airgap':airgap,'frequency':frequency,'thickness':thickness,'dis_ver':dis_ver,'current':current,\
'analytical ac/dc':ratio2,'measured ac/dc':ratio1,'dcloss':dcloss,'acloss':acloss,'error':(ratio1-ratio2)/ratio1,'abs error':abs((ratio1-ratio2)/ratio1)})

data.to_csv('D:\\file\\ECCE\\Magnetic simulation\\python\\winding loss\\data.csv')


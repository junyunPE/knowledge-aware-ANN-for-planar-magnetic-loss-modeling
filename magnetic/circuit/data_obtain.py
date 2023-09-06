import pandas as pd
import math 
import numpy as np

#winding loss
def data_processing(input_file_path, output_file_path):
    # Read input data from CSV file
    temp1 = pd.read_csv(input_file_path, header=0)
    data = np.array(temp1)
    dw = data[:,0]/1e3
    dc = data[:,1]/1e3
    da = data[:,2]/1e3
    dgap1 = data[:,3]/1e3
    dgap2 = data[:,4]/1e3
    fs = data[:,5]
    thickness=data[:,6]/1e3
    DC = data[:,7]  ###  DC resistance

    R11=np.zeros(np.size(fs))
    ZL11=np.zeros(np.size(fs))
    R21=np.zeros(np.size(fs))
    ZL21=np.zeros(np.size(fs))
    R12=np.zeros(np.size(fs))
    ZL12=np.zeros(np.size(fs))
    R22=np.zeros(np.size(fs))
    ZL22=np.zeros(np.size(fs))
    for i in range(np.size(fs)):
        R11[i]=np.complex(data[:,8][i]).real
        ZL11[i]=np.complex(data[:,8][i]).imag
        R21[i]=np.complex(data[:,9][i]).real
        ZL21[i]=np.complex(data[:,9][i]).imag
        R22[i]=np.complex(data[:,10][i]).real
        ZL22[i]=np.complex(data[:,10][i]).imag
    R12=R21
    ZL12=ZL21

    # Define helper functions
    def my_sqrt(x):
        return np.sqrt(x)

    def sd(fs):   #skin depth calculation
        mu=1    #relative permeability
        p=1/5.998e7    #electrical resistance ratio
        temp=np.zeros(np.size(fs))
        for i in range(np.size(fs)):
            temp[i]=math.sqrt(p/(math.pi*fs[i]*mu*mu0))
        return temp

    # Calculate component data
    mu0 = 4 * np.pi * 1e-7  # permeability of vacuum
    sd = sd(fs)
    F = np.zeros(np.size(fs))
    G = np.zeros(np.size(fs))
    for i in range(np.size(fs)):
        temp = thickness[i]/ sd[i]
        F[i] = (np.sinh(temp) + np.sin(temp)) / (np.cosh(temp) - np.cos(temp))
        G[i] = (np.sinh(temp) - np.sin(temp)) / (np.cosh(temp) + np.cos(temp))

    Ra11 = np.zeros(np.size(fs))
    Ra12 = np.zeros(np.size(fs))
    Ra21 = np.zeros(np.size(fs))
    Ra22 = np.zeros(np.size(fs))
    temp = thickness / sd
    Ra11 = temp / 2 * (4 * F + 52 * G) * DC
    Ra22 = temp / 2 * (2 * F + 2 * G) * DC
    Ra12 = temp / 4 * (16 * G) * DC
    Ra21 = Ra12

    RL = dgap2 / (mu0 * da * da ) 
    Rm = dgap1 / (mu0 * dc/2 * dc/2 * np.pi) 
    Np = 4
    Ns = 2
    Lk = Np * Np / RL
    Lm = Np * Np / Rm
    k = Lm / (Lk + Lm)
    L11 = Lm / k
    L22 = L11 / (Np / Ns * Np / Ns)
    L12 = k * np.vectorize(my_sqrt)(L11 * L22)
    L21 = L12

    Za11 = L11 * 2 * math.pi * fs
    Za22 = L22 * 2 * math.pi * fs
    Za21 = L21 * 2 * math.pi * fs
    Za12 = Za21

    if output_file_path==output_file_path1:
        Ra,Rt=Ra11,R11
    elif output_file_path==output_file_path2:
        Ra,Rt=Za11,ZL11
    elif output_file_path==output_file_path3:
        Ra,Rt=Ra21,R21
    elif output_file_path==output_file_path4:
        Ra,Rt=Za21,ZL21
    elif output_file_path==output_file_path5:
        Ra,Rt=Ra22,R22
    else:
        Ra,Rt=Za22,ZL22

    # Create output data frame
    data = pd.DataFrame({
        'The width of the copper windings': dw,
        'The length of the auxiliary leg': da,
        'The diameter of the circular leg': dc,
        'The airgap length of the circular leg': dgap1,
        'The airgap length of the auxiliary':dgap2,
        'Switching frequency':fs,
        'analytical':Ra,'traning':Rt})
    data.to_csv(output_file_path)
    

input_file_path='C:\\Users\\junyun_deng\\Desktop\\odd and open.csv'

#generating R11 data
output_file_path1='C:\\Users\\junyun_deng\\Desktop\\data\\R11.csv'
data_processing(input_file_path, output_file_path1)

#generating Z11 data
output_file_path2='C:\\Users\\junyun_deng\\Desktop\\data\\Z11.csv'
data_processing(input_file_path, output_file_path2)

#generating R21 data
output_file_path3='C:\\Users\\junyun_deng\\Desktop\\data\\R21.csv'
data_processing(input_file_path, output_file_path3)

#generating Z21 data
output_file_path4='C:\\Users\\junyun_deng\\Desktop\\data\\Z21.csv'
data_processing(input_file_path, output_file_path4)

#generating R22 data
output_file_path5='C:\\Users\\junyun_deng\\Desktop\\data\\R22.csv'
data_processing(input_file_path, output_file_path5)

#generating Z22 data
output_file_path6='C:\\Users\\junyun_deng\\Desktop\\data\\Z22.csv'
data_processing(input_file_path, output_file_path6)

##core loss


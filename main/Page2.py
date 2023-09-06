import tkinter as tk
import numpy as np
from PIL import ImageTk, Image
from tkinter import ttk
import sys
sys.path.append('D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\transistor_loss')
sys.path.append('D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\transistor_loss\\library')
sys.path.append('D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\controlloop')
import transistor_selections
import loss
import LTspice
import PLECS
import controlloop
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

Np=4
Ns=2

def create_frame1 (parent_frame,update_callback):
    frame = tk.Frame(parent_frame, borderwidth=2, relief="groove")
    label1 = tk.Label(frame, text="Resonant frequency fs (MHz):")
    label1.grid(row=0, column=0)
    scale1 = tk.Scale(frame, from_=0.5, to=1, resolution=0.5, orient=tk.HORIZONTAL)
    scale1.set(1)
    scale1.grid(row=0, column=1)

    label2 = tk.Label(frame, text="Input voltage (V):")
    label2.grid(row=1, column=0)
    scale2 = tk.Scale(frame, from_=380, to=440, resolution=5, orient=tk.HORIZONTAL)
    scale2.set(400)
    scale2.grid(row=1, column=1)

    label3 = tk.Label(frame, text="Output voltage (V):")
    label3.grid(row=2, column=0)
    scale3 = tk.Scale(frame, from_=24, to=48, resolution=24, orient=tk.HORIZONTAL)
    scale3.set(48)
    scale3.grid(row=2, column=1)

    label4 = tk.Label(frame, text="Transferred power (W):")
    label4.grid(row=3, column=0)
    scale4 = tk.Scale(frame, from_=500, to=1000, resolution=100, orient=tk.HORIZONTAL)
    scale4.set(500)
    scale4.grid(row=3, column=1)

    label5 = tk.Label(frame, text="Switching dead time (ns):")
    label5.grid(row=4, column=0)
    scale5 = tk.Scale(frame, from_=80, to=150, resolution=10, orient=tk.HORIZONTAL)
    scale5.set(100)
    scale5.grid(row=4, column=1)

    label6 = tk.Label(frame, text="Inductance value ratio k:")
    label6.grid(row=5, column=0)
    scale6 = tk.Scale(frame, from_=0.1, to=0.5, resolution=0.05, orient=tk.HORIZONTAL)
    scale6.set(0.1)
    scale6.grid(row=5, column=1)

    label7 = tk.Label(frame, text="Turn-on gate resistance:")
    label7.grid(row=6, column=0)
    scale7 = tk.Scale(frame, from_=0, to=10, resolution=1, orient=tk.HORIZONTAL)
    scale7.set(5)
    scale7.grid(row=6, column=1)

    label8 = tk.Label(frame, text="Turn-off gate resistance:")
    label8.grid(row=7, column=0) 
    scale8 = tk.Scale(frame, from_=0, to=10, resolution=1, orient=tk.HORIZONTAL)
    scale8.set(1)
    scale8.grid(row=7, column=1)

    label9 = tk.Label(frame, text="Turn-on gate voltage:")
    label9.grid(row=8, column=0) 
    scale9 = tk.Scale(frame, from_=4, to=6, resolution=1, orient=tk.HORIZONTAL)
    scale9.set(6)
    scale9.grid(row=8, column=1)

    label10 = tk.Label(frame, text="Negative turn-off gate voltage(V):")
    label10.grid(row=9, column=0)
    scale10 = tk.Scale(frame, from_=0, to=5, resolution=1, orient=tk.HORIZONTAL)
    scale10.set(0)
    scale10.grid(row=9, column=1)

    label11 = tk.Label(frame, text="Commutation loop inductance value (nH):")
    label11.grid(row=10, column=0)
    scale11 = tk.Scale(frame, from_=0, to=5, resolution=1, orient=tk.HORIZONTAL)
    scale11.set(2)
    scale11.grid(row=10, column=1)

    label12 = tk.Label(frame, text="     Primary switching transistor:")
    label12.grid(row=11, column=0)
    values1 = ["GS0650111L", "GS66508B"]
    combobox1 = ttk.Combobox(frame, values=values1, state="readonly")
    combobox1.grid(row=11, column=1)
    combobox1.set(values1[0])

    label13 = tk.Label(frame, text="     Secondary synchronous transistor:")
    label13.grid(row=12, column=0)
    values2 = ["EPC2031"]
    combobox2 = ttk.Combobox(frame, values=values2, state="readonly")
    combobox2.grid(row=12, column=1)
    combobox2.set(values2[0])

    '''
    label14 = tk.Label(frame, text="     Secondary Schottky diodes:")
    label14.grid(row=13, column=0)
    values3 = ["SS2FH6"]
    combobox3 = ttk.Combobox(frame, values=values3, state="readonly")
    combobox3.grid(row=13, column=1)
    combobox3.set(values3[0])
    ''' 
    label15 = tk.Label(frame, text="     PCB layer number:")
    label15.grid(row=0, column=2)
    values4 = [4,6]
    combobox4 = ttk.Combobox(frame, values=values4, state="readonly")
    combobox4.grid(row=0, column=3)
    combobox4.set(values4[1])

    label16 = tk.Label(frame, text="Thickness of single copper layer (mm):")
    label16.grid(row=1, column=2)
    scale16 = tk.Scale(frame, from_=0.035, to=0.07, resolution=0.035, orient=tk.HORIZONTAL)
    scale16.set(0.035)
    scale16.grid(row=1, column=3)

    label17 = tk.Label(frame, text="Total thickness of PCB (mm):")
    label17.grid(row=2, column=2)
    scale17 = tk.Scale(frame, from_=1.6, to=3, resolution=0.1, orient=tk.HORIZONTAL)
    scale17.set(1.6)
    scale17.grid(row=2, column=3)

    label18 = tk.Label(frame, text="The area of thermal pad (mm^2):")
    label18.grid(row=3, column=2)
    scale18 = tk.Scale(frame, from_=40, to=70, resolution=5, orient=tk.HORIZONTAL)
    scale18.set(60)
    scale18.grid(row=3, column=3)

    label19 = tk.Label(frame, text="The width of heat sink (mm):")
    label19.grid(row=4, column=2)
    scale19 = tk.Scale(frame, from_=15, to=30, resolution=1, orient=tk.HORIZONTAL)
    scale19.set(25)
    scale19.grid(row=4, column=3)

    label20 = tk.Label(frame, text="The length of heat sink (mm):")
    label20.grid(row=5, column=2)
    scale20 = tk.Scale(frame, from_=30, to=80, resolution=1, orient=tk.HORIZONTAL)
    scale20.set(50)
    scale20.grid(row=5, column=3)

    label21 = tk.Label(frame, text="The thickness of heat sink base (mm):")
    label21.grid(row=6, column=2)
    scale21 = tk.Scale(frame, from_=3, to=8, resolution=1, orient=tk.HORIZONTAL)
    scale21.set(5)
    scale21.grid(row=6, column=3)

    label22 = tk.Label(frame, text="The thickness of fins (mm):")
    label22.grid(row=7, column=2)
    scale22 = tk.Scale(frame, from_=1, to=2, resolution=0.2, orient=tk.HORIZONTAL)
    scale22.set(1.2)
    scale22.grid(row=7, column=3)

    label23 = tk.Label(frame, text="The height of fins (mm):")
    label23.grid(row=8, column=2)
    scale23 = tk.Scale(frame, from_=10, to=40, resolution=1, orient=tk.HORIZONTAL)
    scale23.set(25)
    scale23.grid(row=8, column=3)

    label24 = tk.Label(frame, text="The number of fins:")
    label24.grid(row=9, column=2)
    scale24 = tk.Scale(frame, from_=10, to=20, resolution=1, orient=tk.HORIZONTAL)
    scale24.set(10)
    scale24.grid(row=9, column=3)

    label25 = tk.Label(frame, text="Crossover frequency (kHz):")
    label25.grid(row=10, column=2)
    scale25 = tk.Scale(frame, from_=10, to=100, resolution=5, orient=tk.HORIZONTAL)
    scale25.set(50)
    scale25.grid(row=10, column=3)

    label26 = tk.Label(frame, text="Phase margin (degree):")
    label26.grid(row=11, column=2)
    scale26 = tk.Scale(frame, from_=40, to=70, resolution=5, orient=tk.HORIZONTAL)
    scale26.set(45)
    scale26.grid(row=11, column=3)

    # Load the image
    image_path = "D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\picture\\topology.jpg"
    img = Image.open(image_path)
    img = img.resize((760, 250))
    photo = ImageTk.PhotoImage(img)
    # Create the label to display the second image
    label_image = tk.Label(frame, image=photo)
    label_image.photo = photo
    label_image.grid(row=14, column=0, pady=10,columnspan=4)

    fr = scale1.get()*1e6
    Vin = scale2.get()
    Vout=scale3.get()
    P = scale4.get()
    td = scale5.get()/1e9
    k = scale6.get()
    Rgon = scale7.get()
    Rgoff = scale8.get()
    Vdon = scale9.get()
    Vdoff = scale10.get()
    Lp = scale11.get()/1e9
    transistor_pri = combobox1.get()
    transistor_sec=combobox2.get()
    #Schottky_diode=combobox3.get()
    PCB_layer=combobox4.get()
    tcu=scale16.get()/1e3
    tPCB = scale16.get()/1e3
    area_thermal = scale18.get()/1e6
    W = scale19.get()/1e3
    L=scale20.get()/1e3
    tb = scale21.get()/1e3
    tf = scale22.get()/1e3
    hf = scale23.get()/1e3
    N_fin= scale24.get()
    fc= scale25.get()*1e3
    phase_margin= scale26.get()
    
    def confirm_callback():
        fr = scale1.get()*1e6
        Vin = scale2.get()
        Vout=scale3.get()
        P = scale4.get()
        td = scale5.get()/1e9
        k = scale6.get()
        Rgon = scale7.get()
        Rgoff = scale8.get()
        Vdon = scale9.get()
        Vdoff = scale10.get()
        Lp = scale11.get()/1e9
        transistor_pri = combobox1.get()
        transistor_sec=combobox2.get()
        #Schottky_diode=combobox3.get()
        PCB_layer=combobox4.get()
        tcu=scale16.get()/1e3
        tPCB = scale16.get()/1e3
        area_thermal = scale18.get()/1e6
        W = scale19.get()/1e3
        L=scale20.get()/1e3
        tb = scale21.get()/1e3
        tf = scale22.get()/1e3
        hf = scale23.get()/1e3
        N_fin= scale24.get()
        fc= scale25.get()*1e3
        phase_margin= scale26.get()
        update_callback(fr, Vin, Vout, P, td, k, Rgon, Rgoff, Vdon, Vdoff, Lp, transistor_pri, transistor_sec, PCB_layer, tcu, tPCB, area_thermal, W, L, tb, tf, hf, N_fin, fc, phase_margin)

    confirm_button = tk.Button(frame, text="Parameters Update", command=confirm_callback)
    confirm_button.grid(row=12, column=2, pady=10, sticky='n')

    return frame, fr, Vin, Vout, P, td, k, Rgon, Rgoff, Vdon, Vdoff, Lp, transistor_pri, transistor_sec, PCB_layer, tcu, tPCB, area_thermal, W, L, tb, tf, hf, N_fin, fc, phase_margin

def create_frame2 (parent_frame, fr, Vin, Vout, P, td, k, Rgon, Rgoff, Vdon, Vdoff, Lp, transistor_pri, transistor_sec, PCB_layer, tcu, tPCB, area_thermal, W, L, tb, tf, hf, N_fin, fc, phase_margin):
    frame = tk.Frame(parent_frame, borderwidth=2, relief="groove")

    for component in [transistor_selections.GS0650111L, transistor_selections.EPC2031]:
        transistor_p = 0
        if component['name'] == transistor_pri:
            transistor_p=component
            break
    
    for component in [transistor_selections.GS0650111L, transistor_selections.EPC2031]:
        transistor_s = 0
        if component['name'] == transistor_sec:
            transistor_s=component
            break
    '''
    for component in [transistor_selections.SS2FH6]:
        Schottky_d = 0
        if component['name'] == Schottky_diode:
            Schottky_d=component
            break
    '''
    time_array,ip,ir,imos,conduction_loss,fs=PLECS.Run_plecs(Vin, P, Vout, td, k, transistor_p['PlECS model'], transistor_p['name'])
    Ioff=PLECS.turn_off_current(imos)
    Vsw,Isw,time,Eoff=LTspice.Turn_off_loss(Rgon,Rgoff,Vin,Ioff,Lp,transistor_p['LTspice model'], transistor_p['name'])
    time=np.array(time)
 
    def plot1(time_array,ip,ir):
        fig, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(time_array*1e9, ip, label='Primary side current', color='red')
        ax2 = ax1.twinx()
        ax2.plot(time_array*1e9, ir, label='Secondary side current', color='blue')

        #ax1.set_xlim(0, 1/fs*1e6)
        ax1.set_xlabel('Time [ns]', fontsize=8)
        ax1.set_ylabel('Primary-side resonant current [A]', fontsize=8, color='red')
        ax2.set_ylabel('Secondary-side diode current [A]', fontsize=8, color='blue')
        ax1.set_title('Steady-state waveforms', fontsize=8)
        ax1.tick_params(axis='both', which='major', direction='in', length=2, width=1, colors='black', labelsize=8)
        ax1.tick_params(axis='both', which='minor', direction='in', length=2, width=1, colors='black', labelsize=8)
        ax2.tick_params(axis='both', which='major', direction='in', length=2, width=1, colors='black', labelsize=8)
        ax2.tick_params(axis='both', which='minor', direction='in', length=2, width=1, colors='black', labelsize=8)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=4)
    plot1(time_array,ip,ir)

    def plot2(Vsw,Isw,time):
        fig, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(time*1e9, Isw, label='Primary side current', color='red')
        ax2 = ax1.twinx()
        ax2.plot(time*1e9, Vsw, label='Secondary side current', color='blue')

        ax1.set_xlabel('Time [ns]', fontsize=8)
        ax1.set_ylabel('Transistor turn-off current [A]', fontsize=8, color='red')
        ax2.set_ylabel('Transistor turn-off voltage [V]', fontsize=8, color='blue')
        ax1.set_title('Turn-off transistion waveforms', fontsize=8)
        ax1.tick_params(axis='both', which='major', direction='in', length=2, width=1, colors='black', labelsize=8)
        ax1.tick_params(axis='both', which='minor', direction='in', length=2, width=1, colors='black', labelsize=8)
        ax2.tick_params(axis='both', which='major', direction='in', length=2, width=1, colors='black', labelsize=8)
        ax2.tick_params(axis='both', which='minor', direction='in', length=2, width=1, colors='black', labelsize=8)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, columnspan=4)
    plot2(Vsw,Isw,time)
    
    tnp=2 #the number of primary transistors
    tns=2 #the number of secondary transistors
    Pnorm = np.arange(100, 501, 100)
    loss_pri=  np.zeros_like(Pnorm)
    loss_sec = np.zeros_like(Pnorm)  
    for i, p in enumerate(Pnorm):
        loss_pri[i],loss_sec[i]=loss.loss(transistor_p, transistor_s, Vin, int(Pnorm[i]), Vout, td, k, Rgon, Rgoff, Lp, tnp, tns)
    def plot3(Pnorm,loss_pri,loss_sec):
        fig, ax1 = plt.subplots(figsize=(5, 3))
        x = range(len(loss_pri))
        ax1.bar(x, height=loss_pri, width=0.15, alpha=0.8, color='red', label="Primary loss")
        ax1.bar(x, height=loss_sec, width=0.15, color='green', label="Secondary loss", bottom=loss_pri)
        #ax1.bar(Pnorm, loss_pri+loss_sec, color='red', width=1)
        ax1.set_xlabel('Transferred power [W]', fontsize=8)
        ax1.set_ylabel('Total power loss [W]', fontsize=8, color='red')
        ax1.set_title('Power loss under different transferred power', fontsize=8)
        ax1.set_xticks(x)
        ax1.legend()
        ax1.set_xticklabels(Pnorm)
        ax1.tick_params(axis='both', which='major', direction='in', length=2, width=1, colors='black', labelsize=8)
        ax1.tick_params(axis='both', which='minor', direction='in', length=2, width=1, colors='black', labelsize=8)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=0, columnspan=4)
    plot3(Pnorm,loss_pri,loss_sec)

    
    def plot4(fc,phase_margin,path,k,v1,v2,f1,f2):
        f_eval, gain_eval, mag, phase_eval, phase=controlloop.compensated_tf(fc,phase_margin,path,k,v1,v2,f1,f2)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
        ax1.semilogx(f_eval,gain_eval, label='Uncompensated control loop')
        ax1.semilogx(f_eval, mag+gain_eval, label='Compensated control loop')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.grid(True)

        ax2.semilogx(f_eval, phase_eval, label='Uncompensated control loop')
        ax2.semilogx(f_eval, phase+phase_eval, label='Compensated control loop')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.grid(True)

        fig.suptitle('Bode Plot')
        ax1.legend()
        ax2.legend()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=5, columnspan=4)

    path4='D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\controlloop\\SIMPLIS\\Heavy load and low input voltage.csv'
    plot4(fc,phase_margin,path4,1e6,1,2,1.5e6,0.68e6)
    return frame

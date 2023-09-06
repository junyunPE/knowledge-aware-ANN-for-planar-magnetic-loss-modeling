import tkinter as tk
import numpy as np
from PIL import ImageTk, Image
from tkinter import ttk
from tkinter import font
import sys
sys.path.append('D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\magnetic\\circuit')
sys.path.append('D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\magnetic\\core')
sys.path.append('D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\magnetic\\thermal')
import equivalent_circuit
import core_loss
import heat
import functions_core
import functions_MOO
import MOO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

Np=4
Ns=2

def create_frame1 (parent_frame,update_callback):
    frame = tk.Frame(parent_frame, borderwidth=2, relief="groove")

    custom_font = font.Font(family="Times New Roman", size=20)
    label0 = tk.Label(frame, text=" Design Inputs", font=custom_font)
    label0.grid(row=0, columnspan=4)

    label1 = tk.Label(frame, text="Switching frequency fs (MHz):")
    label1.grid(row=1, column=0)
    scale1 = tk.Scale(frame, from_=0.5, to=1, resolution=0.5, orient=tk.HORIZONTAL)
    scale1.set(1)
    scale1.grid(row=1, column=1)

    label2 = tk.Label(frame, text="The width of the copper windings dw (mm):")
    label2.grid(row=2, column=0)
    scale2 = tk.Scale(frame, from_=5, to=40, resolution=5, orient=tk.HORIZONTAL)
    scale2.set(5)
    scale2.grid(row=2, column=1)

    label3 = tk.Label(frame, text="The thickness of the copper layer (mm):")
    label3.grid(row=3, column=0)
    scale3 = tk.Scale(frame, from_=0.07, to=0.07, resolution=0, orient=tk.HORIZONTAL)
    scale3.set(0.07)
    scale3.grid(row=3, column=1)

    label4 = tk.Label(frame, text="The length of the auxiliary leg da (mm):")
    label4.grid(row=4, column=0)
    scale4 = tk.Scale(frame, from_=1, to=5, resolution=1, orient=tk.HORIZONTAL)
    scale4.set(1)
    scale4.grid(row=4, column=1)

    label5 = tk.Label(frame, text="The diameter of the circular leg dc (mm):")
    label5.grid(row=5, column=0)
    scale5 = tk.Scale(frame, from_=5, to=20, resolution=1, orient=tk.HORIZONTAL)
    scale5.set(15)
    scale5.grid(row=5, column=1)

    label6 = tk.Label(frame, text="The vertical distance between core and PCB dv (mm):")
    label6.grid(row=6, column=0)
    scale6 = tk.Scale(frame, from_=0.1, to=0.4, resolution=0.1, orient=tk.HORIZONTAL)
    scale6.set(0.1)
    scale6.grid(row=6, column=1)

    label7 = tk.Label(frame, text="The airgap length of the circular leg dgap1 (mm):")
    label7.grid(row=7, column=0) 
    scale7 = tk.Scale(frame, from_=0.1, to=3, resolution=0.05, orient=tk.HORIZONTAL)
    scale7.set(0.25)
    scale7.grid(row=7, column=1)

    label8 = tk.Label(frame, text="The airgap length of the auxiliary leg dgap2 (mm):")
    label8.grid(row=8, column=0)
    scale8 = tk.Scale(frame, from_=0.1, to=2, resolution=0.1, orient=tk.HORIZONTAL)
    scale8.set(1)
    scale8.grid(row=8, column=1)

    label9 = tk.Label(frame, text="   Ambient temperature Ta (⁰C):")
    label9.grid(row=9, column=0)
    scale9 = tk.Scale(frame, from_=25, to=50, resolution=5, orient=tk.HORIZONTAL)
    scale9.set(25)
    scale9.grid(row=9, column=1)

    label10 = tk.Label(frame, text="Layer number of primary windings Np:")
    label10.grid(row=1, column=2)
    label10 = tk.Label(frame, text="4")
    label10.grid(row=1, column=3)

    label10 = tk.Label(frame, text="Layer number of secondary windings Ns:")
    label10.grid(row=2, column=2)
    label10 = tk.Label(frame, text="2")
    label10.grid(row=2, column=3)

    label11 = tk.Label(frame, text="     Core materials:")
    label11.grid(row=3, column=2)
    label11 = tk.Label(frame, text="TP5E")
    label11.grid(row=3, column=3)

    global combobox
    label12 = tk.Label(frame, text="     Voltage exciation waveforms:")
    label12.grid(row=4, column=2)
    values = ["Sinusoidal", "Rectangular"]
    combobox = ttk.Combobox(frame, values=values, state="readonly")
    combobox.grid(row=4, column=3)
    combobox.set(values[1])

    label13 = tk.Label(frame, text="     Peak value of sinusoidal excitation")
    label13.grid(row=5, column=2)
    scale13 = tk.Scale(frame, from_=400, to=400, resolution=0, orient=tk.HORIZONTAL)
    scale13.set(400)
    scale13.grid(row=5, column=3)

    label14 = tk.Label(frame, text="     Positive value of rectangular excitation")
    label14.grid(row=6, column=2)
    scale14 = tk.Scale(frame, from_=400, to=400, resolution=0, orient=tk.HORIZONTAL)
    scale14.set(400)
    scale14.grid(row=6, column=3)

    label15 = tk.Label(frame, text="     Duty_cycle of rectangular excitation")
    label15.grid(row=7, column=2)
    scale15 = tk.Scale(frame, from_=0.1, to=0.9, resolution=0.1, orient=tk.HORIZONTAL)
    scale15.set(0.5)
    scale15.grid(row=7, column=3)

    label16 = tk.Label(frame, text="     Peak value of primary-side current")
    label16.grid(row=8, column=2)
    scale16 = tk.Scale(frame, from_=1, to=9, resolution=0.1, orient=tk.HORIZONTAL)
    scale16.set(1)
    scale16.grid(row=8, column=3)
    
    # Load the image
    image_path = "D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\picture\\geometry.jpg"
    img = Image.open(image_path)
    img = img.resize((520, 520))
    photo = ImageTk.PhotoImage(img)
    label_image = tk.Label(frame, image=photo)
    label_image.photo = photo
    label_image.grid(row=10, column=0, pady=10,columnspan=4)

    fs = scale1.get()*1e6
    dw = scale2.get()/1e3
    thickness=scale3.get()/1e3
    da = scale4.get()/1e3
    dc = scale5.get()/1e3
    dv = scale6.get()/1e3
    dgap1=scale7.get()/1e3
    dgap2=scale8.get()/1e3
    Ta=scale9.get()+273.15
    Vps=scale13.get()
    Vpr=scale14.get()
    duty_cycle=scale15.get()
    waveform=combobox.get()
    Ip=scale16.get()
    
    def confirm_callback():
        fs = scale1.get()*1e6
        dw = scale2.get()/1e3
        thickness=scale3.get()/1e3
        da = scale4.get()/1e3
        dc = scale5.get()/1e3
        dv = scale6.get()/1e3
        dgap1=scale7.get()/1e3
        dgap2=scale8.get()/1e3
        Ta=scale9.get()+273.15
        Vps=scale13.get()
        Vpr=scale14.get()
        duty_cycle=scale15.get()
        waveform=combobox.get()
        Ip=scale16.get()
        update_callback(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip)

    confirm_button = tk.Button(frame, text="Parameters Update", command=confirm_callback)
    confirm_button.grid(row=9, column=2, pady=10, sticky='n')

    return frame,fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle,Ip

def create_frame2 (parent_frame,fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip):
    frame = tk.Frame(parent_frame, borderwidth=2, relief="groove")
    custom_font = font.Font(family="Times New Roman", size=20)
    label0 = tk.Label(frame, text=" Winding Resistance Evaluation", font=custom_font)
    label0.grid(row=0, columnspan=4)
    label_result1 = tk.Label(frame, text="Z11: ")
    label_result1.grid(row=2, column=1, pady=10)
    label_result2 = tk.Label(frame, text="Z11: ")
    label_result2.grid(row=2, column=3, pady=10)
    label_result3 = tk.Label(frame, text="Z21: ")
    label_result3.grid(row=3, column=1, pady=10)
    label_result4 = tk.Label(frame, text="Z22: ")
    label_result4.grid(row=3, column=3, pady=10)
    label_result5 = tk.Label(frame, text="Primary winding loss: ")
    label_result5.grid(row=4, column=1, pady=10)
    label_result6 = tk.Label(frame, text="Secondary winding loss: ")
    label_result6.grid(row=4, column=3, pady=10)

    def clear():
        label_result1.config(text="Z11:")
        label_result2.config(text="Z12:")
        label_result3.config(text="Z21:")
        label_result4.config(text="Z22:")
        label_result5.config(text="Primary winding loss:")
        label_result6.config(text="Secondary winding loss:")

    def string_output(label1,label2,label3,label4,label5,label6,Z11,Z12,Z21,Z22,Ppri,Psec):
        label1.config(text="Z11: " + str(Z11) + "\u03A9")
        label2.config(text="Z12: " + str(Z12) + "\u03A9")
        label3.config(text="Z21: " + str(Z21) + "\u03A9")
        label4.config(text="Z22: " + str(Z22) + "\u03A9")
        label5.config(text="Primary winding loss: " + str(Ppri) + "W")
        label6.config(text="Secondary winding loss: " + str(Psec) + "W")
        
    Is=Ip*Np/Ns
    def simulate():
        clear()
        Z11,Z12,Z21,Z22=equivalent_circuit.simulate(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta)
        Ppri_sim=0.5*Z11.real*Ip*Ip+0.5*Z12.real*Ip*Is
        Psec_sim=0.5*Z21.real*Ip*Is+0.5*Z22.real*Is*Is
        string_output(label_result1,label_result2,label_result3,label_result4,label_result5,label_result6,Z11,Z12,Z21,Z22,Ppri_sim,Psec_sim)

    def calculate():
        clear()
        Z11,Z12,Z21,Z22=equivalent_circuit.calculate(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta)
        Ppri_cal=0.5*Z11.real*Ip*Ip+0.5*Z12.real*Ip*Is
        Psec_cal=0.5*Z21.real*Ip*Is+0.5*Z22.real*Is*Is
        string_output(label_result1,label_result2,label_result3,label_result4,label_result5,label_result6,Z11,Z12,Z21,Z22, Ppri_cal,Psec_cal)

    def AI():
        clear()
        Z11,Z12,Z21,Z22=equivalent_circuit.AI(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta)
        Ppri_AI=0.5*Z11.real*Ip*Ip+0.5*Z12.real*Ip*Is
        Psec_AI=0.5*Z21.real*Ip*Is+0.5*Z22.real*Is*Is
        string_output(label_result1,label_result2,label_result3,label_result4,label_result5,label_result6,Z11,Z12,Z21,Z22,Ppri_AI,Psec_AI)

    # Add the button and result label to frame1
    button = tk.Button(frame, text="Simulate", command=simulate)
    button.grid(row=5, column=1, pady=10)
    button = tk.Button(frame, text="Calculate", command=calculate)
    button.grid(row=5, column=2, pady=10)
    button = tk.Button(frame, text="Physical-informed AI", command=AI)
    button.grid(row=5, column=3, pady=10)

    image_path4 = "D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\picture\\coupled.jpg"
    img4 = Image.open(image_path4)
    img4 = img4.resize((520, 240))
    photo4 = ImageTk.PhotoImage(img4)
    label_image4 = tk.Label(frame, image=photo4)
    label_image4.photo4 = photo4
    label_image4.grid(row=1, column=1,columnspan=4, pady=10) 
    
    return frame

def create_frame3 (parent_frame,fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle):
    frame = tk.Frame(parent_frame, borderwidth=2, relief="groove")
    custom_font = font.Font(family="Times New Roman", size=20)
    label0 = tk.Label(frame, text=" Core Loss Evaluation", font=custom_font)
    label0.grid(row=0, columnspan=4)
    def plot(waveform,fs,Vps,Np,dc,Vpr):
        fig, ax1 = plt.subplots(figsize=(5.2, 3))
        if waveform=="Sinusoidal":
            time,Voltage,B=functions_core.flux_waveform_sin(fs,Vps,Np,np.pi*(dc/2)**2)
            ax1.plot(time*1e6, Voltage, label='Voltage', color='red')
            ax2 = ax1.twinx()
            ax2.plot(time*1e6, B*1e3, label='Flux density', color='blue')
        else:
            time,Voltage,B=functions_core.flux_waveform_rectangular(fs,Vpr,duty_cycle,Np,np.pi*(dc/2)**2)
            ax1.plot(time*1e6, Voltage, label='Voltage', color='red')
            ax2 = ax1.twinx()
            ax2.plot(time*1e6, B*1e3, label='Flux density', color='blue')
        ax1.set_xlim(0, 1/fs*1e6)
        ax1.set_xlabel('Time [us]', fontsize=8)
        ax1.set_ylabel('Voltage [V]', fontsize=8, color='red')
        ax2.set_ylabel('Flux density [mT]', fontsize=8, color='blue')
        ax1.set_title('Excitation and Flux Density Visualization', fontsize=8)
        ax1.tick_params(axis='both', which='major', direction='in', length=2, width=1, colors='black', labelsize=8)
        ax1.tick_params(axis='both', which='minor', direction='in', length=2, width=1, colors='black', labelsize=8)
        ax2.tick_params(axis='both', which='major', direction='in', length=2, width=1, colors='black', labelsize=8)
        ax2.tick_params(axis='both', which='minor', direction='in', length=2, width=1, colors='black', labelsize=8)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, columnspan=4)
    plot(waveform,fs,Vps,Np,dc,Vpr)

    label1_result = tk.Label(frame, text="Core loss: ")
    label1_result.grid(row=2, column=0, pady=10, columnspan=4, sticky='nsew')

    def clear():
        label1_result.config(text="Core loss: ")

    def string_output(label1,Pcv):
        label1.config(text="Core loss: " + str(Pcv) + "W")

    def simulate_sin():
        clear()
        Pcv_sim_sin=core_loss.simulate_sin(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Vps)
        string_output(label1_result,Pcv_sim_sin)

    def calculate_sin():
        clear()
        Pcv_cal_sin=core_loss.calculate_sin(fs, dw, da, dc, dv, dgap1, dgap2,  Vps)
        string_output(label1_result,Pcv_cal_sin)

    def AI_sin():
        clear()
        Pcv_AI_sin=core_loss.AI_sin(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Vps)
        string_output(label1_result,Pcv_AI_sin)

    def simulate_rect():
        clear()
        Pcv_sim_rect=core_loss.simulate_rect(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Vpr, duty_cycle)
        string_output(label1_result,Pcv_sim_rect)
    
    def calculate_rect():
        clear()
        Pcv_cal_rect=core_loss.calculate_rect(fs, dw, da, dc, dv, dgap1, dgap2, Vpr, duty_cycle)
        string_output(label1_result,Pcv_cal_rect)
    
    def AI_rect():
        clear()
        Pcv_AI_rect=core_loss.AI_rect(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Vpr, duty_cycle)
        string_output(label1_result,Pcv_AI_rect)

    if  waveform=="Sinusoidal":
        button = tk.Button(frame, text="Simulate", command=simulate_sin)
        button.grid(row=3, column=1, pady=10, sticky='w')
        button = tk.Button(frame, text="Calculate", command=calculate_sin)
        button.grid(row=3, column=2, pady=10, sticky='w')
        button = tk.Button(frame, text="Physical-informed AI", command=AI_sin)
        button.grid(row=3, column=3, pady=10, sticky='w')
    else:
        button = tk.Button(frame, text="Simulate", command=simulate_rect)
        button.grid(row=3, column=1, pady=10, sticky='w')
        button = tk.Button(frame, text="Calculate", command=calculate_rect)
        button.grid(row=3, column=2, pady=10, sticky='w')
        button = tk.Button(frame, text="Physical-informed AI", command=AI_rect)
        button.grid(row=3, column=3, pady=10, sticky='w')
        
    return frame

def create_frame4 (parent_frame,fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip):
    frame = tk.Frame(parent_frame, borderwidth=2, relief="groove")
    custom_font = font.Font(family="Times New Roman", size=20)
    label0 = tk.Label(frame, text=" Temperature Evaluation", font=custom_font)
    label0.grid(row=0, columnspan=4)
    label_result1 = tk.Label(frame, text="Tmax,core: ")
    label_result1.grid(row=2, column=1, pady=10)
    label_result2 = tk.Label(frame, text="Tmax,pri: ")
    label_result2.grid(row=2, column=2, pady=10)
    label_result3 = tk.Label(frame, text="Tmax,sec: ")
    label_result3.grid(row=2, column=3, pady=10)

    def clear():
        label_result1.config(text="Tmax,core:")
        label_result2.config(text="Tmax,pri:")
        label_result3.config(text="Tmax,sec:")

    def string_output(label1,label2,label3,Tmax_core,Tmax_pri,Tmax_sec):
        label1.config(text="Tmax,core: " + str(Tmax_core) + "℃")
        label2.config(text="Tmax,pri: " + str(Tmax_pri) + "℃")
        label3.config(text="Tmax,sec: " + str(Tmax_sec) + "℃")

    def plot():
        image_path = "C:\\Users\\junyun_deng\\Desktop\\heat.png"
        img = Image.open(image_path)
        img = img.resize((520, 300))
        photo = ImageTk.PhotoImage(img)
        label_image = tk.Label(frame, image=photo)
        label_image.photo = photo
        label_image.grid(row=1, column=1,columnspan=4, pady=10) 
    plot()

    Is=Ip*Np/Ns
    def simulate():
        clear()
        Z11,Z12,Z21,Z22=equivalent_circuit.simulate(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta)
        Ppri_sim=0.5*Z11.real*Ip*Ip+0.5*Z12.real*Ip*Is
        Psec_sim=0.5*Z21.real*Ip*Is+0.5*Z22.real*Is*Is
        if  waveform=="Sinusoidal":
            Pc=core_loss.simulate_sin(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Vps)
        else:
            Pc=core_loss.simulate_rect(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Vpr, duty_cycle)
        Tmax_core,Tmax_pri,Tmax_sec=heat.simulate(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Ppri_sim, Psec_sim, Pc)
        Tmax_core=Tmax_core-273.15
        Tmax_pri=Tmax_pri-273.15
        Tmax_sec=Tmax_sec-273.15
        string_output(label_result1,label_result2,label_result3,np.round(Tmax_core, 4),np.round(Tmax_pri, 4),np.round(Tmax_sec, 4))
        plot()
    
    def calculate():
        clear()
        Ppri_cal,Psec_cal,Pc=functions_MOO.loss_sum_cal(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip)
        Tmax_core,Tmax_pri,Tmax_sec=functions_MOO.temperature(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip)
        string_output(label_result1,label_result2,label_result3,np.round(Tmax_core, 4),np.round(Tmax_pri, 4),np.round(Tmax_sec, 4))
        
    button = tk.Button(frame, text="Simulate", command=simulate)
    button.grid(row=3, column=1, pady=10)
    button = tk.Button(frame, text="Calculate", command=calculate)
    button.grid(row=3, column=2, pady=10)
    button = tk.Button(frame, text="Phsyical-informed AI", command=calculate)
    button.grid(row=3, column=3, pady=10)

    return frame

def create_frame5 (parent_frame,fs, Ta, waveform, Vps, Vpr, duty_cycle, thickness, Ip):
    frame = tk.Frame(parent_frame, borderwidth=2, relief="groove")
    fs=500e3
    Ta=25+273.15
    waveform="Rectangular"
    Vps=400
    Vpr=400
    duty_cycle=0.5
    thickness=0.07/1e3
    Ip=5
    def calculate():
        def plot_results(optimal_solutions,solutions):
            fig, ax = plt.subplots(figsize=(5, 4))
            x = [sol[1][0] for sol in optimal_solutions]
            y = [sol[1][1]*1e6 for sol in optimal_solutions]
            ax.plot(x, y, '-o', color='blue', label='Pareto front')
            for sol in solutions:
                ax.scatter(sol[1][0], sol[1][1]*1e6, color='red')
            ax.set_xlabel('Total power loss [W]')
            ax.set_ylabel('Total volume [cm^3]')
            ax.set_title('Objective Space')
            canvas = FigureCanvasTkAgg(plt.gcf(), frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, columnspan=4)
        solutions_filtered,optimal_solutions_filtered=MOO.analytical(fs, Ta, waveform, Vps, Vpr, duty_cycle, thickness, Ip)
        plot_results(optimal_solutions_filtered, solutions_filtered)

    label_result1 = tk.Label(frame, text="Switching frequency: " + str(fs/1e6) + "MHz")
    label_result1.grid(row=1, column=0, pady=10)

    label_result2 = tk.Label(frame, text="Primary-side peak current: " + str(Ip) + "A")
    label_result2.grid(row=1, column=1, pady=10)

    button = tk.Button(frame, text="Calculate", command=calculate)
    button.grid(row=2, column=0, pady=10)
    
    button = tk.Button(frame, text="Physical-informed AI", command=calculate)
    button.grid(row=2, column=1, pady=10)

    return frame

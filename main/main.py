import mph
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import ttk
import cmath
import numpy as np
import math 
import pandas as pd

import Page1
#import Page2
#import Page3

root = tk.Tk()
root_width = 2000
root_height = 1100
root.geometry(f"{root_width}x{root_height}")
title_style = ttk.Style()
title_style.configure("Title.TLabel", font=("Times New Roman", 24))
root.title("Physical-informed Artifical Neural Network for Magnetics")

style = ttk.Style()
style.configure("TNotebook.Tab", font=("Times New Roman", 14))
notebook = ttk.Notebook(root)
page1 = ttk.Frame(notebook)
page2 = ttk.Frame(notebook)
page3 = ttk.Frame(notebook)
notebook.add(page1, text='Magnetic design')
notebook.add(page2, text='Circuit design')
notebook.add(page3, text='Global optimization')
notebook.pack()


def update_frames1(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip):
    global frame2, frame3, frame4, frame5

    frame2.destroy()
    frame3.destroy()
    frame4.destroy()
    #frame5.destroy()

    frame2 = Page1.create_frame2(page1, fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip)
    frame3 = Page1.create_frame3(page1, fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle)
    frame4 = Page1.create_frame4(page1, fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip)
    #frame5 = Page1.create_frame5(page1, fs, Ta, waveform, Vps, Vpr, duty_cycle, thickness, Ip)

    frame2.grid(row=0, column=1, padx=10, pady=10, sticky='n')
    frame3.grid(row=0, column=2, padx=10, pady=10, sticky='n')
    frame4.grid(row=1, column=1, padx=10, pady=10, sticky='n')
    #frame5.grid(row=1, column=2, padx=10, pady=10, sticky='n')

frame1, fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip=Page1.create_frame1(page1,update_frames1)
frame2=Page1.create_frame2(page1, fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip)
frame3=Page1.create_frame3(page1, fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle)
frame4=Page1.create_frame4(page1, fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip)
#frame5=Page1.create_frame5(page1, fs, Ta, waveform, Vps, Vpr, duty_cycle, thickness, Ip)

frame1.grid(row=0, column=0, padx=10, pady=10, rowspan=2)
frame2.grid(row=0, column=1, padx=10, pady=10, sticky='n')
frame3.grid(row=1, column=1, padx=10, pady=10, sticky='n')
frame4.grid(row=0, column=2, padx=10, pady=10, sticky='n')



#frame5.grid(row=1, column=2, padx=10, pady=10, sticky='n')
'''
def update_frames2(fr, Vin, Vout, P, td, k, Rgon, Rgoff, Vdon, Vdoff, Lp, transistor_pri, transistor_sec, PCB_layer, tcu, tPCB, area_thermal, W, L, tb, tf, hf, N_fin, fc, phase_margin):
    global frame7
    frame7.destroy()
    frame7=Page2.create_frame2(page2, fr, Vin, Vout, P, td, k, Rgon, Rgoff, Vdon, Vdoff, Lp, transistor_pri, transistor_sec, PCB_layer, tcu, tPCB, area_thermal, W, L, tb, tf, hf, N_fin, fc, phase_margin)
    frame7.grid(row=0, column=1, padx=10, pady=10, sticky='n')


frame6, fr, Vin, Vout, P, td, k, Rgon, Rgoff, Vdon, Vdoff, Lp, transistor_pri, transistor_sec, PCB_layer, tcu, tPCB, area_thermal, W, L, tb, tf, hf, N_fin, fc, phase_margin = Page2.create_frame1(page2,update_frames2)
frame7=Page2.create_frame2(page2, fr, Vin, Vout, P, td, k, Rgon, Rgoff, Vdon, Vdoff, Lp, transistor_pri, transistor_sec, PCB_layer, tcu, tPCB, area_thermal, W, L, tb, tf, hf, N_fin, fc, phase_margin)
frame6.grid(row=0, column=0, padx=10, pady=10, rowspan=2)
frame7.grid(row=0, column=1, padx=10, pady=10, rowspan=2)

frame8 = Page3.create_frame1(page3)
frame8.grid(row=0, column=0, padx=10, pady=10, rowspan=2)
'''

# Load the image in page1
image_path = "D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\picture\\logo.jpg"
img = Image.open(image_path)
img = img.resize((520, 235))
photo = ImageTk.PhotoImage(img)
image_frame = tk.Frame(page1, borderwidth=2, relief="groove")
image_frame.grid(row=1, column=2, pady=5, sticky=tk.W)
label_image = tk.Label(image_frame, image=photo)
label_image.photo = photo
label_image.pack()


root.mainloop()
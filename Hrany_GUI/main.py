import tkinter as tk
import cv2
import filtry
from tkinter.filedialog import askopenfilename

# Vytvorenie GUI okna
window = tk.Tk()

# Nastavenie veľkosti GUI a nadpis okna GUI
window.title('Voľba filtrov')
window.geometry('350x550')

# Pridanie jednotlivých checkboxov a ich názvov
labels = ['Gausov sum', 'Laplacian', 'Sobel X', 'Sobel Y', 'Sobel XY', 'Prewitt X', 'Prewitt Y', 'Prewitt XY','Robinson', 'Canny', 'Jednoduche prahovanie - Canny', 'Jednoduche prahovanie - Prewitt', 'Jednoduche prahovanie - Sobel ','Jednoduche prahovanie - Laplacian', 'Adpativne prahovanie']


# Ukladanie hodnot o checkboxoch a ich vypis do okna.
var_gauss = tk.BooleanVar()
cb_gauss = tk.Checkbutton(window, text="Gauss", variable=var_gauss)
cb_gauss.pack(anchor='w', padx=5, pady=5)

var_lap = tk.BooleanVar()
cb_lap = tk.Checkbutton(window, text="Laplacian", variable=var_lap)
cb_lap.pack(anchor='w', padx=5, pady=5)

var_sobel_x = tk.BooleanVar()
cb_sobel_x = tk.Checkbutton(window, text="Sobel X", variable=var_sobel_x)
cb_sobel_x.pack(anchor='w', padx=5, pady=5)

var_sobel_y = tk.BooleanVar()
cb_sobel_y = tk.Checkbutton(window, text="Sobel Y", variable=var_sobel_y)
cb_sobel_y.pack(anchor='w', padx=5, pady=5)

var_sobel_xy = tk.BooleanVar()
cb_sobel_xy = tk.Checkbutton(window, text="Sobel XY", variable=var_sobel_xy)
cb_sobel_xy.pack(anchor='w', padx=5, pady=5)

var_prewitt_x = tk.BooleanVar()
cb_prewitt_x = tk.Checkbutton(window, text="Prewitt X", variable=var_prewitt_x)
cb_prewitt_x.pack(anchor='w', padx=5, pady=5)

var_prewitt_y = tk.BooleanVar()
cb_prewitt_y = tk.Checkbutton(window, text="Prewitt Y", variable=var_prewitt_y)
cb_prewitt_y.pack(anchor='w', padx=5, pady=5)

var_prewitt_xy = tk.BooleanVar()
cb_prewitt_xy = tk.Checkbutton(window, text="Prewitt XY", variable=var_prewitt_xy)
cb_prewitt_xy.pack(anchor='w', padx=5, pady=5)

var_robinson = tk.BooleanVar()
cb_robinson = tk.Checkbutton(window, text="Robinson", variable=var_robinson)
cb_robinson.pack(anchor='w', padx=5, pady=5)

var_canny = tk.BooleanVar()
cb_canny= tk.Checkbutton(window, text="Canny", variable=var_canny)
cb_canny.pack(anchor='w', padx=5, pady=5)

var_simple_thresh = tk.BooleanVar()
cb_thresh= tk.Checkbutton(window, text="Jednoduche prahovanie", variable=var_simple_thresh)
cb_thresh.pack(anchor='w', padx=5, pady=5)

var_adaptive_thresh = tk.BooleanVar()
cb_adaptive_thresh= tk.Checkbutton(window, text="Adaptivne prahovanie", variable=var_adaptive_thresh)
cb_adaptive_thresh.pack(anchor='w', padx=5, pady=5)


label = tk.Label(window, text="Spodna hranica jednoducheho thresholdu: 50")
label.pack()

# Min a max hodnoty slideru.
slider = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL)
slider.pack()

# Funkcia definujuca slider, pre spodnu hranicu jednoducheho thresholdu.
def update_label(val):
    # Uvodny text.
    label.config(text=f"Spodna hranica jednoducheho thresholdu: {slider.get()}")
    # Priradenie hodnoty slideru do globalnej premnnej threshold_low_value z triedy filtry.
    filtry.threshold_low_value = slider.get()
slider.config(command=update_label)

# Funkcia pre moznost zadania vlastneho obrazka.
def show_file_picker():
    filename = askopenfilename()
    print(filename)
    if(filename != ""):
        filtry.SetPicture(filename)

# Vytvorenie bloku tlačítka s nápisom
button_file_picker = tk.Button(window, text='Zvol obrazok', command=show_file_picker)
# Vlozenie buttonu do okna
button_file_picker.pack()

# Kontrola, ktory checkbox bol zakliknuty a na zaklade toho sa zavola prislusna funkcia z triedy filtry.
# Vid vysvetlenie na prvom if-e.
def button_choice():
    # Ak bol checkbox zaskrtnuty, tak je hodnota rovna True
    if(var_gauss.get() == True):
        # Volanie prislusnej fcie.
        img = filtry.GausSum()
        # Zobrazenie vystupneho obrazku pricom sa okno bude volat ako label na indexi 0 v poli labels.
        cv2.imshow(labels[0], img)

    if (var_lap.get() == True):
        img = filtry.laplacian()
        cv2.imshow(labels[1], img)

    if (var_sobel_x.get() == True):
        img = filtry.sobel_X()
        cv2.imshow(labels[2], img)

    if (var_sobel_y.get() == True):
        img = filtry.sobel_Y()
        cv2.imshow(labels[3], img)

    if (var_sobel_xy.get() == True):
        img = filtry.sobel_XY()
        cv2.imshow(labels[4], img)

    if (var_prewitt_x.get() == True):
        img = filtry.prewitt_x()
        cv2.imshow(labels[5], img)

    if (var_prewitt_y.get() == True):
        img = filtry.prewitt_y()
        cv2.imshow(labels[6], img)

    if (var_prewitt_xy.get() == True):
        img = filtry.prewitt_xy()
        cv2.imshow(labels[7], img)

    if (var_robinson.get() == True):
        img = filtry.robinson()
        cv2.imshow(labels[8], img)

    if (var_canny.get() == True):
        img = filtry.canny_detect()
        cv2.imshow(labels[9], img)

    if (var_simple_thresh.get() == True):
        img, img_2, img_3, img_4 = filtry.simple_thresholding()
        cv2.imshow(labels[10], img)
        cv2.imshow(labels[11], img_2)
        cv2.imshow(labels[12], img_3)
        cv2.imshow(labels[13], img_4)

    if (var_adaptive_thresh.get() == True):
        img = filtry.adaptive_thresh()
        cv2.imshow(labels[14], img)

# Vytvorenie bloku tlačítka s nápisom
button = tk.Button(window, text='Potvrď !', command=button_choice)
button.pack()

window.mainloop()
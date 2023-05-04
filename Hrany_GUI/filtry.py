import cv2
import numpy as np

# Pomocou funkcie IMREAD_GRAYSCALE sa obrázok prevedie do šedého formátu, ktorý
# sa následne bude používať pri prahovaní (thresholding)
img_gray = cv2.imread('obrazok_hrany.jpg', cv2.IMREAD_GRAYSCALE)
threshold_low_value = 0

def SetPicture(path):
    global img_gray
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Použitie funkcie GaussainBlur na obrázok, z dôvodu zníženia šumu v obrázku. Funkcia zjemní vysoké výkyvy intenzít pixelov,
# zníži šum a zlepší tak identifikáciu okrajov. V argumentoch funkcie sa nachádza obrázok, veľkosť Gaussoveského jadra
# (3x3) a hodnota 0 znamená, že štandardná odchýlka sa vypočíta automaticky na základe veľkosti jadra.
# Zjemnenený obrázok sa použije u Laplaceovho operátora, nakoľko Laplaceov operátor je citlivý na akýkoľvek šum.
def GausSum():
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    return img_blur

# Typy operátorov pre detekciu hran
# Laplaceov operátor
# Do funkcie je zadadný vstupný obrázok img_blur a druhý parameter hovorí o tom v ako formáte má byť výstupný obrázok (64bit float).
# Ako obrázok bol použitý obrázok, na ktorý už bolo použité Gaussove rozostrenie.
def laplacian():
    laplacian = cv2.Laplacian(GausSum(), cv2.CV_64F)

    return laplacian

# Sobelov operator
# Detekcia hran sa realizuje pomocou dvoch jadier (kernelov) - jedno pre osu X a jedno pres osu Y. Tieto jadrá sa
# konvoluujú so vstupným obrázkom a výstupom konvolúcie sú gradienty.
# Do funkcie sa zadáva vstupný obrázok v šedom tóne, násldne sa zadáva dátový typ výstupného obrázku (64 bit float),
# následne je prvý parameter dx, to znamená, že ak je rovno 1, tak je určené, že sa počíta prvá derivácia v smere x,
# ak je rovno 0, tak sa nepočíta deriváacia v smere x. To isté platí analogicky pre posledný parameter dy.
# Pre sobel_xy platí, že sa počíta prvá derivácia aj v smere x aj v smere y, preto je dx=1 aj dy=1.
def sobel_X():
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1,0)
    return sobel_x

def sobel_Y():
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0,1)
    return sobel_y

def sobel_XY():
    sobel_xy = cv2.Sobel(img_gray, cv2.CV_64F, 1,1)
    return  sobel_xy


# Prewittov operator X
# Ako sa prvé sa určia masky pre os X a pre os Y, ktoré už sú zadefinované v literatúre.
def prewitt_x():
    mask_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_x = cv2.filter2D(img_gray, -1, mask_prewitt_x)

    return prewitt_x
# Prewittov operator Y
def prewitt_y():
    mask_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_y = cv2.filter2D(img_gray, -1, mask_prewitt_y)

    return prewitt_y

# Prewittov operator XY
def prewitt_xy():
    mask_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    mask_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_x = cv2.filter2D(img_gray, -1, mask_prewitt_x)
    prewitt_y = cv2.filter2D(img_gray, -1, mask_prewitt_y)

    # Pomocou funkie filter2D sa masky skonvoluujú s obrázkom. Opäť sa do funkcie zadáva obrázok img_gray, -1 hovorí o tom,
    # že výstupný obrázok má mať rovnaký dátový typ a nakoniec sa vkladá maska (kernel) pre smer X a smer Y.
    prewitt_xy = cv2.bitwise_or(prewitt_x, prewitt_y)

    return prewitt_xy

# Robinsonov operator
# Nastavenie definovanej masky Robinsonovho operátora a následná konvolúcia masky so vstupným obrázkom.

def robinson():
    mask_robinson = np.array([[1,1,1], [1, -2, 1], [-1, -1, -1]])
    conv_robinson = cv2.filter2D(img_gray, -1, mask_robinson)

    return  conv_robinson

# Cannyho hranový detektor
# Do funkcie je zadaný vstupný obrázok, 120 a 220 sú prahové hodnoty, pričom ak je hodnota pixelu nad spodnou prahovou
# hodnotou, tak je braný ako hrana, ak je hodnota pixelu väčšia ako horná prahová hranica, tak je braný ako silná hrana.
# Hodnota 3 hovorí veľkosti Sobelovho jadra, ktoré je použité pre detekciu hrán (3x3).
def canny_detect():
    canny = cv2.Canny(img_gray, 120, 220, 3)

    return canny

# Prahovanie (Threshold)
# Jednoduché prahovanie - ak je hodnota pixelu menšia ako je nastavená dolná prahovacia hodnota, tak je pixel označený ako
#                         čierny. V prípade, že je jeho intenzita vyššia, tak bude pixel označený v obrázku ako biely.
#                       - So správnym nastavením prahovacích hodnôt je možné dostať vysokú kvalitu. Čím vyššie bude
#                         nastavená dolná prahovacia hodnota, tým vyššia je pravdepodobnosť, že bude väčšina pixelov vo
#                         výslednom obrázku označená ako čierna.

def simple_thresholding():
    # Nastavenie spodnej prahovacej hodnoty na 100.
    global threshold_low_value
    # First je premenná, do ktorej sa ukladá prahovacia hodnota, ktorá bola použitá pri operácii. Do premennej thresh_XXXX,
    # sa ukladá binárny obrázok, ktorý je výsledkom funkcie threshold. Ďalej do funkcie threshold vstupujú obrázky, ktoré
    # vznikli na výstupe jednoltivých operátorov. Ďalej sa zadáva spodná prahovacia hodnota, následne horná prahovacia hodnota
    # a ako posledné sa volá funkcia THRESH_BINARY, ktorá zabezpečí to, že hodnoty pixelov pod spodnou hranicou budú nastavené
    # na 0 (čierne pixely) a hodnoty pixelov nad spodnou hranicou budú nastavené na max hodnotu, to znamená na 255.
    first,thresh_canny = cv2.threshold(canny_detect(), threshold_low_value, 255, cv2.THRESH_BINARY)
    first,thresh_prewitt = cv2.threshold(prewitt_xy(), threshold_low_value, 255, cv2.THRESH_BINARY)
    first,thresh_sobel = cv2.threshold(sobel_XY(), threshold_low_value, 255, cv2.THRESH_BINARY)
    first,thresh_laplacian = cv2.threshold(laplacian(), threshold_low_value, 255, cv2.THRESH_BINARY)

    return thresh_sobel, thresh_laplacian, thresh_prewitt, thresh_canny

# Adaptívne prahovanie (Adaptívny thresholding) - adaptívne prahovanie si zistí prahovú hodnotu podľa algoritmu, ktorý
# zvolíme. Parametre sú obrázok, maximálna prahovacia hodnota (255), algoritmus, ktorý bude použitý pre výpočet optimálnej
# prahovej hodnoty, veľkosť bloku (11x11) respektíve to, akú veľkú plochu pixelov algoritmus analyzuje. Posledná sa
# zadáva konštanta (2), ktorá sa odpočítava od strednej hodnoty susedných pixelov čím sa zabezpečí lepšia prahová hodnota
# pre rôzne svetelné podmienky. Experimentálne bolo zistené, že najlepšia hodnota konštanty je 2.
def adaptive_thresh():
    adaptive_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return adaptive_thresh
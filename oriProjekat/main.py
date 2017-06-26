# coding: utf-8

import cv2                                      # import OpenCV biblioteke
import numpy as np                              # NumPy biblioteka, "np" je sinonim koji se koristi dalje u kodu kada se koriste funkcije ove biblioteke
import matplotlib.pyplot as plt                 # biblioteka za plotovanje, tj. crtanje grafika, slika... "plt" je sinonim
import collections

# k-means
from sklearn.cluster import KMeans

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD


import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12       # za prikaz većih slika i plotova

def ucitavanje_slike(putaja):
    ''' Funckija za ucitavanje slike sa diska pomocu openCV, koja kao parametar prima putanju do slike.
    Ucitana slika je numPy matrica. OpenCv ucitava sliku kao BGR, pa se ona koncertuje u RGB.
    Matrica za ovakvu sliku ima 3 dimenzije, gde je prva visina, druga sirina, a treca predstavlja boju'''
    slika = cv2.imread(putaja)
    slika = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
    return slika

def konvertovanje_slike_u_sivo(slika):
    ''' Funkcija za konvertovanje slike u sivo. ndarray je n-dimenzionalna matrica. Grayscale slika nema RGB
    vec samo intenzitet piksela(0 je crno, a 255 belo, sve ostalo je nijansa sive).
    '''
    visina = slika.shape[0]
    sirina = slika.shape[1]
    siva_slika = np.ndarray((visina, sirina), dtype=np.uint8)
    for i in range(0, visina):
        for j in range(0, sirina):
            pix = slika[i, j]
            r = pix[0]
            g = pix[1]
            b = pix[2]
            siva = float(r) * .2989 + float(g) * .587 + float(b) * .114
            siva_slika[i, j] = siva

    # ili image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return siva_slika

def binarizacija_slike(siva_slika):
    ''' Funckija vrsi binarizaciju slike, tj izdvajanje sadrzaja od pozadine.
    Prvo se odabere prag, a zatim svi pikseli koji imaju intenzitet manji od tog praga dobijaju
    vr.0 (postaju crni), a svi pikseli koji imaju intenzitet veci od tog praga dobijaju vr.255 (postaju beli).
    Ovde je prag 127.'''
    visina = siva_slika.shape[0]
    sirina = siva_slika.shape[1]
    image_bin = np.ndarray((visina, sirina), dtype=np.uint8)

    for i in range(0, visina):
        for j in range(0, sirina):
            if siva_slika[i,j] > 127:
                image_bin[i,j] = 255
            else:
                image_bin[i,j] = 0
    # ili ret, image_bin = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def dilatacija(image):
    ''' Funkcija koja vrsi dilataciju slike. To je morfoloska operacija i koristi se za uklanjanje raznih sumova i smetnji
    koji su posledice obrade slike tj.konvertovanja, binarizacije. Vrsi se nad binarnom slikom.
    Menja sliku pomocu kernela, koji je 3x3 matrica. Kernel se postavlja preko svakog piksela na slici i poredi
    se sa susednim el.koje zahvata.'''
    kernel = np.ones((3,3))
    return cv2.dilate(image, kernel, iterations=1)

def erozija(image):
    ''' Funkcija koja vrsi dilataciju slike. To je morfoloska operacija i koristi se za uklanjanje raznih sumova i smetnji
    koji su posledice obrade slike tj.konvertovanja, binarizacije. Vrsi se nad binarnom slikom.
    Menja sliku pomocu kernela, koji je 3x3 matrica. Kernel se postavlja preko svakog piksela na slici i poredi
    se sa susednim el.koje zahvata.'''
    kernel = np.ones((3,3))
    return cv2.erode(image, kernel, iterations=1)

def resize(region):
    ''' Funkcija koja transformise sliku u sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

class Region(object):
    x = 0
    y = 0
    w = 0
    h = 0
    red = 0

    def __init__(self, x, y, w, h, red):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.red = red


def selektovanje_regiona(image_orig, image_bin):
    '''Funkcija koja rucno odredjuje regione na originalnoj slici. Za svaki region se
    pravi posebna slika dimenzije 28x28.
    Povratna vrednost su oriinalna slika sa obelezenim regionima i niz slika region koje su
    sortirane po rastucoj vrednosti x-ose
    '''

    konture = []
    visina = image_bin.shape[0]
    sirina = image_bin.shape[1]
    w=0
    listax=[]
    nadjen = 1
    pocetakVisina=-1
    pomocni = -1
    ponovo=0
    noviRed=0
    red =0
    while(ponovo==0):
        red+=1
        ponovo=1
        pocetakVisina=pomocni
        print "Tuuu", pocetakVisina
        pamti=0
        for s in range(0,sirina):
            t = 0

            if(nadjen==0):
                w+=1
            moze=1
            for v in range(pocetakVisina,visina):

                if(image_bin[v][s]==255 ):
                    listax.append(v)
                    t=1
                    nadjen=0
                    moze=0
                nr=0
                if moze==0 :
                    for ss in range(0, sirina):
                        if (image_bin[v][ss] == 255):
                            nr=1

                if nr==0 and moze==0 :
                    pamti=v
                    break
                if pamti!=0:
                    if v > pamti:
                        break
            if(t==0 and nadjen==0):
                x=s-w
                minimum = 1000000000
                maximum = 0

                listax.sort()
                for i, value in enumerate(listax):
                    if value < minimum:
                        minimum = value
                    if value > maximum:
                        maximum = value
                y=minimum

                h=maximum-minimum
                listax=[]
                nadjen=1
                st = Region(x,y,w,h,red)
                konture.append(st)
                ponovo=0
                if y+h+1>pomocni:
                    pomocni=y+h+1
                h = 0
                w = 0

    regioni = {}
    regioni2 = {}
    for kontura in konture:
        x=getattr(kontura,"x")
        y=getattr(kontura,"y")
        w=getattr(kontura,"w")
        h=getattr(kontura,"h")
        red=getattr(kontura,"red")
        region = image_bin[y:y + h + 1, x:x + w + 1];
        regioni[x+y*red*1000] = [resize(region), (x, y, w, h, red)]
        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    sortirani_regioni = collections.OrderedDict(sorted(regioni.items()))
    lista_sortiranih_regiona = np.array(sortirani_regioni.values())

    sortirane_regije = lista_sortiranih_regiona[:, 1]

    razmaci = [-sortirane_regije[0][0] - sortirane_regije[0][2]]
    pom=sortirane_regije[0][4]
    dodaj=0
    razmak=0
    for x, y, w, h, red in lista_sortiranih_regiona[1:-1, 1]:
        if red!=pom:
            pom=red
            razmaci[-1] += dodaj
            razmaci[-1]+=razmak
        else:
            razmaci[-1] += x
            pom=red
        dodaj=x+w
        print razmaci[-1],razmak
        if razmaci[-1]>razmak:
            razmak=razmaci[-1]
        razmaci.append(-x - w )
    razmaci[-1] += sortirane_regije[-1][0]
    for i in razmaci:
        print i
    return image_orig, lista_sortiranih_regiona[:, 0], razmaci

def skaliranje(image):
    ''' Funkcija koja sve elemente matrice koja ima vr. 0 ili 255
    skalira na vr.od 0 do 1
    '''
    return image / 255

def matrix_to_vector(image):
    ''' Funkcija koja sliku koja je matrica 28x28 transformise u vektor sa 784 elementa'''
    return image.flatten()


def spremi_za_nm(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Funkcija skalira elemente regiona na [0,1] i transformise ga u vektor od 784 elementa
    '''
    ready_for_ann = []
    for region in regions:
        scale = skaliranje(region)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann

def konvertovanje_izlaza(alphabet):
    '''Funkcija koja konvertuje alfabet u niz pogodan za obučavanje NM,
        odnosno niz čiji su svi elementi 0 osim elementa čiji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def odredjivanje_pobednika(output): # output je vektor sa izlaza neuronske mreze
    '''pronaći i vratiti indeks neurona koji je najviše pobuđen'''
    return max(enumerate(output), key=lambda x: x[1])[0]

def prikaz_rezultata(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    '''
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[odredjivanje_pobednika(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[odredjivanje_pobednika(output)]
    return result

def kreiraj_model():
    ''' Funkcija koja implementira veštačku neuronsku mrežu sa 784 neurona na ulaznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid na izlazu
        a relu na ulaznom sloju.
    '''
    model = Sequential()

    model.add(Dense(128, input_dim=784, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(26, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def treniraj(ann, X_train, y_train):
    ''' Funkcija koja vrsi obucavanje vestacke neuronske mreze'''
    X_train = np.array(X_train, np.float32)                 # dati ulazi
    y_train = np.array(y_train, np.float32)                 # zeljeni izlazi za date ulaze

                                                            # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=2000, batch_size=40, verbose=0, shuffle=False)

    return ann


def prikaz_slike(image, color= False):
    ''' Funkcija za prikaz slike'''
    if color:
        print ("color")
        plt.imshow(image)
        plt.show()
    else:
        print ("gray")
        plt.imshow(image, 'gray')
        plt.show()

def test_ucitavanje_slike():
    path = "images/train/alphabet.png"
    image = ucitavanje_slike(path)
    print image.shape
    prikaz_slike(image)
    return image

def test_konvertovanje_slike_u_sivo():
    image = test_ucitavanje_slike()
    image_gray = konvertovanje_slike_u_sivo(image)
    print image_gray
    plt.imshow(image_gray, 'gray')
    plt.show()

def test_binarizacije_slike():
    image = test_ucitavanje_slike()
    image_gray = konvertovanje_slike_u_sivo(image)
    image_bin = binarizacija_slike(image_gray)
    print image_bin
    plt.imshow(image_bin, 'gray')
    plt.show()

def test_binarizacije_slike2():
    image = test_ucitavanje_slike()
    image_gray = konvertovanje_slike_u_sivo(image)
    image_bin = binarizacija_slike2(image_gray)
    print image_bin
    plt.imshow(image_bin, 'gray')
    plt.show()

def test_dilatacija_erozija():
    image = test_ucitavanje_slike()
    image_gray = konvertovanje_slike_u_sivo(image)
    image_bin = binarizacija_slike(image_gray)
    d = dilatacija(image_bin)
    plt.imshow(d, 'gray')
    plt.show()
    e = erozija(image_bin)
    plt.imshow(e, 'gray')
    plt.show()

def test_pronalazenje_regiona():
    image = test_ucitavanje_slike()
    image_gray = konvertovanje_slike_u_sivo(image)
    image_bin = binarizacija_slike2(image_gray)
    contours = pronalazenje_regiona(image_bin)
    print image_bin
    plt.imshow(image_bin, 'gray')
    plt.show()
    img = image.copy()
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    plt.imshow(img)
    plt.show()

def test_resize():
    img = test_ucitavanje_slike()
    ref = (28, 28)
    res = resize(img).shape[0:2]
    print ref
    print 'Test rezise:',res == ref

def test_skaliranje():
    matrix = np.array([[0, 255], [51, 153]], dtype='float')
    ref = np.array([[0., 1.], [0.2, 0.6]], dtype='float')
    res = skaliranje(matrix)
    print 'Test skaliranja:', np.array_equal(res, ref)

def test_matrix_to_vector():
    image = np.ndarray((28, 28))
    ref = (784L,)
    res = matrix_to_vector(image).shape
    print 'Test matrix to vector:', res == ref

def test_konvertovanja_izlaza():
    alphabet = [0, 1, 2]
    ref = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float')
    res = konvertovanje_izlaza(alphabet).astype('float')
    print 'Test konvertovanja izlaza:', np.array_equal(res, ref)

def test_odredjivanje_pobednika():
    output = [0., 0.2, 0.3, 0.95]
    ref = 3
    res = odredjivanje_pobednika(output)
    print 'Test odredjivanja pobednika:', res == ref

def test_prikaz_rezultata():
    alphabet = np.array(['a', 'b', 'c'])
    outputs = np.array([[0.8, 0.1, 0.1], [0.2, 0.8, 0], [0.1, 0.9, 0], [0.1, 0, 0.9]], np.float32)
    from unittest.util import namedtuple
    KMeansMock = namedtuple('KMeansMock', ['labels_', 'cluster_centers_'])
    k_means_mock = KMeansMock(labels_=[0, 1, 0], cluster_centers_=[1, 20])
    result = prikaz_rezultata(outputs, alphabet, k_means_mock)
    assert result == 'ab bc', 'string "' + result + '" nije jednak očekivanom "ab bc"'
    print 'SUCCESS'

def ucitaj_za_train():
    '''slika2 = ucitavanje_slike('images/test/test2.png')
        slika_siva2 = konvertovanje_slike_u_sivo(slika2)
        slika_bin2 = binarizacija_slike(slika_siva2)

        selected_regions2, letters2, region_distances2 = selektovanje_regiona(slika2.copy(), slika_bin2)
        prikaz_slike(selected_regions2)
        print 'Broj prepoznatih regiona:', len(letters2)

        abeceda2 = ['V','O','D','A','J','E','P','L','A','V','A']
        inputs2 = prepare_for_ann(letters2)
        outputs2 = konvertovanje_izlaza(abeceda2)'''





    #result = results[odredjivanje_pobednika(results[0])]

def obradi_sliku():
    slika = ucitavanje_slike('images/train/abeceda.png')
    slika_siva = konvertovanje_slike_u_sivo(slika)
    slika_bin = binarizacija_slike(slika_siva)
    return slika, slika_bin

def selektuj_regione(slika, slika_bin):
    regioni, slova, razmaci = selektovanje_regiona(slika.copy(), slika_bin)
    prikaz_slike(regioni)
    print 'Broj prepoznatih regiona:', len(slova)
    return regioni, slova, razmaci

def istreniraj(slova):
    abeceda = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    ulazi = spremi_za_nm(slova)
    izlazi = konvertovanje_izlaza(abeceda)
    model = kreiraj_model()
    model = treniraj(model, ulazi, izlazi)
    return model, abeceda

def testiraj(model, abeceda):
    image_color = ucitavanje_slike('images/test/test1.png')
    img = binarizacija_slike(konvertovanje_slike_u_sivo(image_color))
    regioni, slova, razmaci = selektovanje_regiona(image_color.copy(), img)
    prikaz_slike(regioni)
    print 'Broj prepoznatih regiona:', len(slova)

    razmaci = np.array(razmaci).reshape(len(razmaci), 1)
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    k_means.fit(razmaci)

    inputs = spremi_za_nm(slova)
    results = model.predict(np.array(inputs, np.float32))
    print prikaz_rezultata(results, abeceda, k_means)

    abeceda2 = ['V', 'O', 'D', 'A', 'J', 'E', 'P', 'L', 'A', 'V', 'A']
    brPogodjenih=0
    index=0;
    for idx, output in enumerate(results[0:, :]):
        if abeceda[odredjivanje_pobednika(output)]==abeceda2[index]:
            brPogodjenih+=1
        index+=1
    procenat = brPogodjenih/len(abeceda2)*100
    print "Ukupno pogodjenih: ",brPogodjenih, "od: ", len(abeceda2)
    print ("a to je: %.2f%%" %procenat)

def main():
    slika, slika_bin = obradi_sliku()
    regioni, slova, razmaci = selektuj_regione(slika, slika_bin)
    model, abeceda = istreniraj(slova)
    testiraj(model, abeceda)
if __name__ == "__main__":
    main()
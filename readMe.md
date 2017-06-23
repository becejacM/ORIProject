@ivan7792
Becejac Milana, SW10/2014 (grupa 2)

Opis problema:
Data je slika na kojoj se nalazi tekst u vise redova. Potrebno je razdvojiti redove, a zatim svaki red razdvojiti na slova. Slova se, potom, grupisu u reci(stringove) koje se izgovaraju.

Algoritmi koji ce se koristiti:

    1.Za prepoznavanje slova sa slike:
    - Convolution Neural Network
    2.Za razdvajanje redova:
    - K means

Metrike za poredjenje performansi algoritma:

    1. procenat tacnosti pogadjanja slova sa slike na osnovu rucno napravljenih podataka ili na osnovu podataka iz dataset-a sa interneta
    2. procenat tacno izgovorenih reci. Procenat se odredjuje tako sto postoji testni skup sa pravilno napisanim recima iz teksta koji se poredi sa skupom reci koje su detektovane iz teksta. Ovo ujedno predstvalja nacin validacije resenja.

Dataset-ovi za treniranje i testiranje ce biti rucno napravljeni za pocetak dok sve ne proradi kako treba, a kasnije ce treniranje i testiranje biti izvrseno na dataset-u sa neta koji sadrzi 335690 primera za treniranje i 92977 primera za testiranje. Dataset-ovi ce imati slike velikih slova enegleske abecede.
Dataset sa interneta:
http://publications.idiap.ch/index.php/publications/show/709
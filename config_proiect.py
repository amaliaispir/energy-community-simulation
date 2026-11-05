"""
=============================================================================
CONFIGURARE COMUNA A PROIECTULUI - un singur loc pentru cifrele impartite
=============================================================================

DE CE EXISTA ACEST FISIER:
  Pana acum, aceleasi cifre erau scrise de mana in trei scripturi diferite
  (modelare_pv.py, pv_comunitate.py, simulare_multiagent.py) si incepusera
  sa NU mai coincida: modelare_pv.py rula centrala cu 4 panouri, iar
  simulare_multiagent.py cu 7; unul analiza casa 2000905, celalalt "prima
  casa valida". Rezultatul: doua rulari despre acelasi lucru dadeau cifre
  diferite, imposibil de comparat intre ele.

  De acum, cifrele care trebuie sa fie ACELEASI peste tot stau AICI, iar
  cele trei scripturi le importa. Daca vrei sa schimbi casa analizata sau
  tipul de panou, schimbi INTR-UN SINGUR LOC (aici) si toate scripturile se
  aliniaza automat.

CE NU STA AICI:
  Parametrii care sunt, prin natura lor, specifici unei singure etape (ex.
  banda de toleranta a semaforului sau intervalul orar al recomandarilor,
  care exista doar in simularea multi-agent) raman in scriptul lor - nu are
  sens sa fie "comuni" cand doar un script ii foloseste.
=============================================================================
"""

# =========================================================================
# CASA ANALIZATA (analizele pe o SINGURA casa)
# =========================================================================
# ID-ul casei folosite in modelare_pv.py (Pasul 5: productie vs. consum) si
# in simulare_multiagent.py (comunitatea de o casa). Trebuie sa fie ACEEASI
# in ambele, altfel cifrele celor doua etape nu se pot compara intre ele.
#
# Casele valide gasite de pipeline-ul din MAIN_corectat.py sunt:
#   2000901, 2000904, 2000905, 2000906, 2000907, 2000908, 2000909,
#   2000911, 2000913
# Daca pui aici un ID inexistent sau invalid, scripturile se opresc cu o
# eroare care iti arata lista completa a caselor valide.
CASA_IMPLICITA = 2000901


# =========================================================================
# PANOUL FOTOVOLTAIC (acelasi model real in toate etapele)
# =========================================================================
# Panou real de pe piata: Trina Solar Vertex S 425W.
# Verificare rapida a numelui (425W):
#   0.213 (randament) x (1.13 x 1.76) m2 x 1000 W/m2 (iradianta STC)
#   = ~423.6 W ~= 425W - se potriveste cu fisa tehnica reala.
NUME_PANOU = "Trina Solar Vertex S 425W"

PARAMETRI_PANOU = dict(
    pv_efficiency=0.213,          # randamentul panoului (21.3%)
    panel_width_m=1.13,           # latimea unui panou
    panel_height_m=1.76,          # inaltimea unui panou
    distance_between_arrays_m=3.0,  # distanta intre siruri (sa nu se umbreasca)
)


# =========================================================================
# DIMENSIUNEA CENTRALEI - stabilita la Etapa 2, refolosita la Etapa 3
# =========================================================================
# La Etapa 2 (pv_comunitate.py) am dimensionat "de mana", uitandu-ne la
# graficele productie vs. consum cumulat si la tabelul de capacitati
# candidate: 50 de panouri pentru comunitatea de 9 case.
#
# DE CE 50 SI NU MAI PUTINE: cu 50 de panouri (21.2 kWp) productia acopera
# ~89% din consumul anual cumulat al celor 9 case - aproape de neutralitate
# energetica pe an, care e tinta fireasca a unei comunitati energetice. La
# 40 de panouri acoperirea ar scadea la ~72%, lasand un sfert din consum
# descoperit fara un castig real in schimb.
# DE CE NU MAI MULTE: peste ~54 de panouri productia anuala depaseste
# consumul, iar surplusul de vara (deja 170% din consumul din Iunie-August)
# ar creste fara sa mai acopere nimic din consumul de iarna.
PANOURI_COMUNITATE = 50
CASE_IN_COMUNITATE = 9

# Cate panouri revin, in medie, unei singure case: 50 / 9 = 5.56 panouri
# (~2.35 kWp pe casa, ordinul de marime al unei instalatii rezidentiale
# reale). Vezi panouri_pentru() de mai jos pentru motivatie.


def panouri_pentru(numar_case):
    """
    Cate panouri ii revin unei comunitati de `numar_case` case, pastrand
    ACEEASI densitate de panouri pe casa ca la dimensionarea de la Etapa 2.

    DE CE "COTA-PARTE" SI NU UN NUMAR ALES SEPARAT:
      La Etapa 2 am ales, de mana, 40 de panouri pentru 9 case - o densitate
      de 40/9 = 4.44 panouri pe casa, echilibrata intre iarna si vara. La
      Etapa 3, comunitatea simulata are (deocamdata) o singura casa. Daca i-am
      da centrala INTREAGA a comunitatii (40 de panouri), productia ar depasi
      de ~9 ori consumul unei singure case si agentul de recomandari ar da
      practic acelasi semnal toata ziua - simularea nu ar mai arata nimic
      interesant. Daca, invers, am alege un numar rupt din context (ex. 7),
      Etapa 3 nu ar mai avea nicio legatura numerica cu dimensionarea facuta
      la Etapa 2.

      Solutia: ii dam casei COTA EI PARTE din centrala comunitatii - adica
      exact proportia dimensionata la Etapa 2. Cu o casa: round(40 * 1/9) = 4
      panouri. Cand extinzi comunitatea la toate cele 9 case, aceeasi formula
      da inapoi round(40 * 9/9) = 40 de panouri, adica fix centrala de la
      Etapa 2 - fara sa schimbi nimic in cod.

    Rotunjim la intreg (nu poti monta 4.44 panouri fizice) si garantam
    minim un panou.
    """
    return max(1, round(PANOURI_COMUNITATE * numar_case / CASE_IN_COMUNITATE))


# Numarul de panouri folosit in analizele pe o singura casa (modelare_pv.py
# si simularea multi-agent cu o casa in comunitate): 4 panouri.
PANOURI_O_CASA = panouri_pentru(1)

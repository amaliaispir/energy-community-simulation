"""
=============================================================================
MECANISM DE STIMULARE PRIN RECOMANDARI + ALOCARE PONDERATA A BATERIEI
COMUNITARE PE BAZA DE SCOR
=============================================================================

Etapa noua a proiectului, dupa simularea multi-agent cu O SINGURA casa
(simulare_multiagent.py) si simularea COMUNITATII cu baterie DAR FARA
comportament diferentiat pe casa (simulare_comunitate_baterie.py).

ACEST SCRIPT (Pasul 1 al etapei noi) construieste DOAR:
  (a) modulul de comportament PROBABILISTIC al caselor (o casa nu mai
      respecta AUTOMAT recomandarea, ca la Etapa anterioara - o respecta
      cu o probabilitate p_follow_k, diferita de la o casa la alta), si
  (b) scorul DINAMIC al fiecarei case, pe o FEREASTRA MOBILA de 60 de zile
      (1440 ore), care masoara cat de "disciplinata" a fost casa in a
      urma recomandarile primite in ultimele 60 de zile.
ACEST SCRIPT, la Pasul 2 (adaugat acum, peste Pasul 1 deja validat),
adauga si ALGORITMUL DE DESCARCARE SI ALOCARE PONDERATA A BATERIEI
COMUNITARE (50 kWh) pe baza scorului dinamic calculat la Pasul 1:
  - in surplus solar comunitar, bateria se INCARCA din excedent (fiecare
    casa isi acopera integral consumul direct din solar);
  - in deficit comunitar, fiecare casa primeste intai o cota din solarul
    disponibil, PROPORTIONALA cu propriul consum, apoi deficitul ramas al
    fiecarei case e acoperit din bateria comunitara CU O COTA PROPORTIONALA
    CU SCORUL EI curent (casele mai "disciplinate" - care au urmat mai
    multe recomandari in ultimele 60 de zile - primesc prioritate la
    energia stocata), iar ce nu poate fi acoperit nici asa vine din retea.
  - vezi aloca_baterie_ponderat_pe_scor() si ModelComunitateScoruri.
    _pas_baterie() pentru detalii si formule exacte.

CE REFOLOSESTE acest script (nimic nu se re-implementeaza de la zero):
  - din modelare_pv.py (care la randul lui invaluie pipeline-ul din
    MAIN_corectat.py - citire IRISE, curatare, marcarea caselor valide):
    ruleaza_pana_la_marcarea_caselor, pregateste_consum_orar_casa
    (completeaza_golurile + filtreaza_valorile_exagerate +
    calculeaza_consum_orar), construieste_date_meteo,
    construieste_model_solar, orientare_optima, construieste_centrala_pv,
    calculeaza_productie_pv - EXACT aceleasi functii ca in
    simulare_comunitate_baterie.py, pentru consecventa intre etape.
  - din config_proiect.py: panoul real (Trina Solar Vertex S 425W,
    PARAMETRI_PANOU) si dimensiunea centralei comunitare (PANOURI_COMUNITATE
    = 50 de panouri, dimensionata la Etapa 2 si NErediscutata aici).
  - din simulare_comunitate_baterie.py: selecteaza_case_valide_comunitate
    (cele 9 case cu Status == "Valid") si determina_interval_comunitate.
  - din simulare_multiagent.py: regula semaforului de recomandari (banda de
    toleranta, interval orar 7:00-22:00, ROSU_LA_DEFICIT) si constanta
    PROCENT_AJUSTARE - ACELEASI valori, ca semnalele sa insemne acelasi
    lucru in toate etapele proiectului. Cadrul Mesa (AgentRecomandare /
    AgentCasa / Model, activare in ordine fixa: intai recomandarea, apoi
    casele) e extins aici de la 1 casa la cele 9 case ale comunitatii, cu
    doua completari fata de Etapa anterioara: decizia caselor devine
    PROBABILISTICA (nu mai respecta automat) si fiecare casa isi tine un
    SCOR dinamic pe fereastra mobila.

CE E NOU FATA DE simulare_multiagent.py:
  1. Comunitatea intreaga (9 case), nu doar CASA_IMPLICITA.
  2. AgentCasa nu mai respecta AUTOMAT recomandarea: la fiecare pas cu
     semnal activ ("rosu"/"verde"), arunca un zar (random() < p_follow_k):
       - daca-l respecta: se ajusteaza consumul ca la Etapa anterioara
         (+-PROCENT_AJUSTARE) SI se contorizeaza recomandare_urmarita = 1;
       - daca nu: consumul ramane cel estimat, recomandare_urmarita = 0.
     "galben" (sau in afara intervalului orar) NU e o recomandare primita -
     nu intra deloc in calculul scorului (nici la numarator, nici la numitor).
  3. Scorul dinamic Score_k(t) = (recomandari URMATE in ultimele 1440 ore) /
     (recomandari PRIMITE in ultimele 1440 ore), calculat eficient cu o
     fereastra glisanta (collections.deque, O(1) pe pas, nu O(fereastra)) -
     cu 8760+ ore x 9 case, o recalculare naiva pe toata fereastra la
     fiecare pas ar fi mult prea lenta.
       - primele 7 zile (168 ore) de simulare: istoric insuficient -> scorul
         e valoarea implicita 0.50 (nu se calculeaza raportul, care oricum
         ar fi instabil pe foarte putine date).
       - dupa 7 zile, daca fereastra nu contine NICIO recomandare primita
         (posibil doar in situatii degenerate, ex. banda de toleranta prea
         larga) -> ramane tot 0.50, ca sa nu impartim la zero.
       - limita inferioara: Score_k(t) >= 0.05, aplicata DOAR pe scorul
         calculat din date reale (nu schimba valoarea implicita 0.50).
  4. Cele 3 scenarii cerute de profesor (probabilitati fixe, NU se
     schimba in timpul simularii): Pasiv (0.20 pentru toate), Activ (0.80
     pentru toate), Eterogen/Realist (probabilitati diferite per casa).

REPRODUCTIBILITATE: fiecare scenariu ruleaza cu ACELASI seed implicit
(SEED_IMPLICIT), astfel incat cele 3 rulari consuma secventa de numere
aleatoare in ACEEASI ordine (semnalul e determinist, deci "cine arunca
zarul si cand" e identic intre scenarii) - doar pragul p_follow_k difera.
Comparatia dintre scenarii devine astfel mai curata: diferentele din
tabelul final vin din probabilitati, nu din zgomotul aleator.
=============================================================================
"""

from collections import deque

import numpy as np
import pandas as pd
import mesa
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modelare_pv import (
    ruleaza_pana_la_marcarea_caselor,
    construieste_date_meteo,
    construieste_model_solar,
    orientare_optima,
    construieste_centrala_pv,
    calculeaza_productie_pv,
    pregateste_consum_orar_casa,
)
from config_proiect import PARAMETRI_PANOU, NUME_PANOU, PANOURI_COMUNITATE
from simulare_comunitate_baterie import (
    selecteaza_case_valide_comunitate,
    determina_interval_comunitate,
    Q_NOM_ESS_COMUNITATE_KWH,
)
from simulare_multiagent import (
    BANDA_TOLERANTA,
    INTERVAL_ORAR_RECOMANDARI,
    PROCENT_AJUSTARE,
    ROSU_LA_DEFICIT,
    _elimina_timestampuri_duplicate,
)
from model_baterie import SOC_MIN, SOC_MAX, SOC_INITIAL, DELTA_SOC_LIM, DT_H, calculeaza_SC_SS


# =============================================================================
# ZONA DE CONFIGURARE
# =============================================================================

# Cele 9 case valide ale comunitatii (Status == "Valid" in baza IRISE - vezi
# config_proiect.py si selecteaza_case_valide_comunitate). Le tinem si
# hardcodate aici (in ordinea data de profesor) ca sa putem defini direct
# probabilitatile Scenariului 3 pe ID de casa - pregateste_orizont_comunitate
# VERIFICA la rulare ca lista gasita in baza de date coincide cu aceasta,
# si se opreste cu o eroare clara daca nu coincide (nu continua in tacere
# cu o lista gresita).
CASE_VALIDE = [
    2000901, 2000904, 2000905, 2000906,
    2000907, 2000908, 2000909, 2000911, 2000913,
]

# --- Fereastra mobila de scor: 60 de zile = 1440 de ore ---------------------
FEREASTRA_SCOR_ORE = 60 * 24          # 1440
ISTORIC_MINIM_ORE = 7 * 24            # 168 - sub atat, scorul e implicit
SCOR_IMPLICIT = 0.50
SCOR_MINIM = 0.05

SEED_IMPLICIT = 42

# --- Cele 3 scenarii cerute de profesor (p_follow_k per casa) --------------
SCENARIUL_1_PASIV = {id_casa: 0.20 for id_casa in CASE_VALIDE}
SCENARIUL_2_ACTIV = {id_casa: 0.80 for id_casa in CASE_VALIDE}
SCENARIUL_3_ETEROGEN = {
    2000901: 0.15, 2000904: 0.35, 2000905: 0.50,
    2000906: 0.65, 2000907: 0.80, 2000908: 0.85,
    2000909: 0.90, 2000911: 0.40, 2000913: 0.75,
}

SCENARII = {
    "Scenariul 1 (Pasiv)": SCENARIUL_1_PASIV,
    "Scenariul 2 (Activ)": SCENARIUL_2_ACTIV,
    "Scenariul 3 (Eterogen)": SCENARIUL_3_ETEROGEN,
}

# --- o culoare distincta per casa, folosita consecvent in toate graficele --
CULOARE_CASA = {
    2000901: "#1f77b4", 2000904: "#ff7f0e", 2000905: "#2ca02c",
    2000906: "#d62728", 2000907: "#9467bd", 2000908: "#8c564b",
    2000909: "#e377c2", 2000911: "#7f7f7f", 2000913: "#bcbd22",
}


# =============================================================================
# PASUL 1: orizontul comunitar - consumul orar (curatat) al celor 9 case +
# productia orara a centralei comunitare (50 panouri Trina Solar Vertex S)
# =============================================================================
def pregateste_orizont_comunitate(case_valide=CASE_VALIDE, number_of_panels=PANOURI_COMUNITATE):
    """
    Aduna, pe acelasi orizont orar, consumul ESTIMAT (curatat) al fiecareia
    din cele 9 case si productia ESTIMATA a centralei PV comunitare -
    exact aceleasi surse ca in simulare_comunitate_baterie.py, dar aici
    fiecare casa isi PASTREAZA propria coloana de consum (nu se insumeaza
    intr-un singur P_Load,comunitate - agentul de recomandari are nevoie de
    totalul comunitatii, dar fiecare AgentCasa are nevoie de consumul LUI).

    Returneaza (orizont, pv_system):
      orizont   - DataFrame indexat pe timp (index Ora), cu coloanele
                  "Productie_Estimata_Wh" si, pentru fiecare casa,
                  f"Consum_Estimat_Wh_{id_casa}" - DOAR pentru orele in care
                  productia SI consumul TUTUROR celor 9 case exista
                  (intersectia tuturor intervalelor).
      pv_system - obiectul PVplant al centralei comunitare (util pentru
                  rezumate - numar de panouri, putere de varf etc.)
    """
    print("\n=== ORIZONT COMUNITAR: Pasul 1 - identificam casele valide ===")
    df_consum, status_case = ruleaza_pana_la_marcarea_caselor()
    case_gasite = selecteaza_case_valide_comunitate(status_case)
    if sorted(case_gasite) != sorted(case_valide):
        raise ValueError(
            "Lista caselor valide gasita ACUM in baza IRISE difera de cea "
            "asteptata (cele 9 case hardcodate in CASE_VALIDE, pentru care "
            "sunt definite si probabilitatile Scenariului 3):\n"
            f"  Asteptat: {sorted(case_valide)}\n"
            f"  Gasit:    {sorted(case_gasite)}\n"
            "Actualizati CASE_VALIDE (si SCENARIUL_3_ETEROGEN) daca lista s-a "
            "schimbat intentionat."
        )
    print(f"Case valide confirmate ({len(case_valide)}): {case_valide}")

    data_inceput, data_sfarsit = determina_interval_comunitate(df_consum, case_valide)
    print(f"Interval comunitate (cerut BATEM): {data_inceput} -> {data_sfarsit}")

    # --- productia centralei PV comunitare (50 panouri, Etapa 2) -----------
    print("\n=== ORIZONT COMUNITAR: Pasul 2 - productia centralei PV comunitare ===")
    weather_data = construieste_date_meteo(data_inceput=data_inceput, data_sfarsit=data_sfarsit)
    solar_model = construieste_model_solar(weather_data)
    best_exposure, best_slope = orientare_optima(
        solar_model, initial_slope_deg=150, initial_exposure_deg=0,
    )
    pv_system = construieste_centrala_pv(
        solar_model, best_exposure, best_slope,
        number_of_panels=number_of_panels, **PARAMETRI_PANOU,
    )
    pv_power_Wh = calculeaza_productie_pv(pv_system)
    print(
        f"Centrala PV comunitara: {pv_system.number_of_panels} panouri "
        f"{NUME_PANOU}, {pv_system.peak_power_kW:.2f} kWp, orientare optima "
        f"{best_exposure:.0f} grade expunere / {best_slope:.0f} grade inclinare"
    )

    productie = pd.Series(
        pv_power_Wh, index=pd.DatetimeIndex(solar_model.datetimes),
        name="Productie_Estimata_Wh",
    )
    productie = _elimina_timestampuri_duplicate(productie, "Productie_Estimata_Wh")

    # --- consumul orar curatat al fiecarei case -----------------------------
    print("\n=== ORIZONT COMUNITAR: Pasul 3 - consumul orar curatat al fiecarei case ===")
    serii = [productie]
    for id_casa in case_valide:
        consum_orar_casa = pregateste_consum_orar_casa(df_consum, id_casa)
        coloana = f"Consum_Estimat_Wh_{id_casa}"
        s = consum_orar_casa.set_index("Ora")["Consum_Orar_Casa_Wh"].rename(coloana)
        s = _elimina_timestampuri_duplicate(s, coloana)
        serii.append(s)
        print(f"  Casa {id_casa}: {len(s)} ore procesate, "
              f"consum total {s.sum() / 1000:.0f} kWh pe interval")

    orizont = pd.concat(serii, axis=1, join="inner").sort_index()
    orizont.index.name = "Ora"
    if orizont.empty:
        raise ValueError(
            "Intersectia orelor cu productie si cu consumul TUTUROR celor 9 "
            "case e goala - verificati fusul orar / intervalul comunitatii."
        )

    zile = (orizont.index[-1] - orizont.index[0]).total_seconds() / 86400
    print(
        f"\n=== ORIZONT COMUNITAR FINAL: {len(orizont)} ore (~{zile:.0f} zile), "
        f"de la {orizont.index[0]} pana la {orizont.index[-1]} ==="
    )
    return orizont, pv_system


# =============================================================================
# PASUL 2: agentii - AgentRecomandare (semafor comunitar) + AgentCasa
# (comportament probabilistic + scor dinamic pe fereastra mobila)
# =============================================================================
class AgentRecomandareComunitate(mesa.Agent):
    """
    Agentul de recomandari al COMUNITATII (9 case) - un singur exemplar.
    Regula semaforului e IDENTICA cu cea din simulare_multiagent.py (banda
    de toleranta BANDA_TOLERANTA, interval orar activ INTERVAL_ORAR_
    RECOMANDARI, conventia ROSU_LA_DEFICIT) - doar ca "consumul total
    estimat" e acum suma consumurilor estimate ale TUTUROR celor 9 case
    (nu al uneia singure).
    """

    def __init__(self, model, banda, interval_orar, rosu_la_deficit):
        super().__init__(model)
        self.banda = banda
        self.ora_inceput, self.ora_sfarsit = interval_orar
        self.rosu_la_deficit = rosu_la_deficit
        self.istoric = []

    def _in_intervalul_activ(self, ora_din_zi):
        return self.ora_inceput <= ora_din_zi < self.ora_sfarsit

    def step(self):
        pas = self.model.pas_curent
        ora = self.model.orizont.index[pas]
        productie = float(self.model.orizont["Productie_Estimata_Wh"].iloc[pas])
        consum_total_estimat = self.model.consum_total_estimat(pas)

        if not self._in_intervalul_activ(ora.hour):
            semnal = "galben"
        else:
            prag_sus = productie * (1 + self.banda)
            prag_jos = productie * (1 - self.banda)

            if consum_total_estimat > prag_sus:
                # DEFICIT comunitar (consum > productie + banda) => comunitatea
                # trebuie sa reduca consumul.
                semnal = "rosu" if self.rosu_la_deficit else "verde"
            elif consum_total_estimat < prag_jos:
                # SURPLUS comunitar (productie > consum + banda) => comunitatea
                # poate consuma local, din solar.
                semnal = "verde" if self.rosu_la_deficit else "rosu"
            else:
                # In banda de toleranta => nu se emite nicio actiune.
                semnal = "galben"

        self.model.semnal_curent = semnal
        self.model.ultima_productie_estimata = productie
        self.model.ultimul_consum_total_estimat = consum_total_estimat
        if semnal in ("rosu", "verde"):
            self.model.total_recomandari[semnal] += 1

        self.istoric.append({
            "Ora": ora,
            "Productie_Estimata_Wh": productie,
            "Consum_Total_Estimat_Wh": consum_total_estimat,
            "semnal": semnal,
        })


class AgentCasaProbabilistica(mesa.Agent):
    """
    O casa a comunitatii. La fiecare pas cu semnal ACTIV ("rosu"/"verde"),
    decide daca respecta recomandarea aruncand un zar impotriva probabilitatii
    proprii p_follow (self.random - RNG-ul modelului, ca reproductibilitatea
    sa depinda doar de seed-ul modelului):

      random() < p_follow  -> RESPECTA: consumul se ajusteaza +-PROCENT_
                               AJUSTARE (ca la Etapa anterioara) si se
                               contorizeaza recomandare_urmarita = 1.
      altfel                -> NU respecta: consumul ramane cel estimat,
                               recomandare_urmarita = 0.

    "galben" (sau in afara intervalului orar) NU e o recomandare primita -
    nu se arunca zarul, nu intra in scor.

    SCOR DINAMIC (fereastra mobila de FEREASTRA_SCOR_ORE ore = 60 de zile):
        Score_k(t) = (recomandari urmate in ultimele 1440h) /
                     (recomandari primite in ultimele 1440h)
    Calculat incremental cu un deque(maxlen=FEREASTRA_SCOR_ORE) - la fiecare
    pas se adauga (primit, urmat) al pasului curent si, daca fereastra era
    deja plina, se scade din sume intrarea care tocmai iese din fereastra -
    O(1) pe pas, nu O(fereastra) (esential pentru ~8760 ore x 9 case).
    """

    def __init__(
        self, model, id_casa, p_follow, procent_ajustare,
        fereastra_ore=FEREASTRA_SCOR_ORE, istoric_minim_ore=ISTORIC_MINIM_ORE,
        scor_implicit=SCOR_IMPLICIT, scor_minim=SCOR_MINIM,
    ):
        super().__init__(model)
        self.id_casa = id_casa
        self.p_follow = p_follow
        self.procent_ajustare = procent_ajustare
        self.coloana_consum = f"Consum_Estimat_Wh_{id_casa}"

        self.istoric_minim_ore = istoric_minim_ore
        self.scor_implicit = scor_implicit
        self.scor_minim = scor_minim

        self._fereastra = deque(maxlen=fereastra_ore)  # (primit, urmat) per ora
        self._suma_primite = 0
        self._suma_urmate = 0

        self.ultimul_consum_estimat = None
        self.ultimul_consum_simulat = None
        self.ultimul_semnal = None
        self.ultima_recomandare_primita = 0
        self.ultima_recomandare_urmarita = None  # None cand nu s-a primit nimic
        self.ultimul_scor = scor_implicit
        self.istoric = []

    def _actualizeaza_fereastra_si_scor(self, pas, primit, urmat):
        if len(self._fereastra) == self._fereastra.maxlen:
            primit_iesit, urmat_iesit = self._fereastra[0]  # va fi scos automat de append
            self._suma_primite -= primit_iesit
            self._suma_urmate -= urmat_iesit
        self._fereastra.append((primit, urmat))
        self._suma_primite += primit
        self._suma_urmate += urmat

        if pas < self.istoric_minim_ore:
            return self.scor_implicit
        if self._suma_primite == 0:
            return self.scor_implicit
        scor = self._suma_urmate / self._suma_primite
        return max(scor, self.scor_minim)

    def step(self):
        pas = self.model.pas_curent
        ora = self.model.orizont.index[pas]
        consum_estimat = float(self.model.orizont[self.coloana_consum].iloc[pas])
        semnal = self.model.semnal_curent

        primit = 1 if semnal in ("rosu", "verde") else 0
        urmat = 0
        recomandare_urmarita = None

        if primit:
            respecta = self.random.random() < self.p_follow
            recomandare_urmarita = 1 if respecta else 0
            if respecta:
                urmat = 1
                if semnal == "rosu":
                    consum_simulat = consum_estimat * (1 - self.procent_ajustare)
                else:  # "verde"
                    consum_simulat = consum_estimat * (1 + self.procent_ajustare)
            else:
                consum_simulat = consum_estimat
        else:
            consum_simulat = consum_estimat

        scor = self._actualizeaza_fereastra_si_scor(pas, primit, urmat)

        self.ultimul_consum_estimat = consum_estimat
        self.ultimul_consum_simulat = consum_simulat
        self.ultimul_semnal = semnal
        self.ultima_recomandare_primita = primit
        self.ultima_recomandare_urmarita = recomandare_urmarita
        self.ultimul_scor = scor

        self.istoric.append({
            "Ora": ora,
            "id_casa": self.id_casa,
            "semnal": semnal,
            "Consum_Estimat_Wh": consum_estimat,
            "Consum_Simulat_Wh": consum_simulat,
            "recomandare_primita": primit,
            "recomandare_urmarita": recomandare_urmarita,
            "Scor": scor,
        })


# =============================================================================
# PASUL 2bis: alocarea PONDERATA a bateriei comunitare pe baza de scor
# (functie PURA - fara stare, fara Mesa - usor de testat independent)
# =============================================================================
def aloca_baterie_ponderat_pe_scor(deficit_k, scor_k, putere_disponibila, epsilon=1e-9):
    """
    Imparte `putere_disponibila` (kW - deja limitata la puterea maxima de
    DESCARCARE pe care o poate da bateria in ora curenta, vezi _pas_baterie)
    intre casele care mai au deficit (deficit_k: dict id_casa -> kW),
    PROPORTIONAL cu scorul curent al fiecarei case (scor_k: dict id_casa ->
    scor in [0.05, 1]) - scor mai mare => cota mai mare din baterie.

        w_k(t) = Score_k(t) / suma(Score_j(t))
        P_bat_cota,k(t) = w_k(t) * putere_disponibila

    O casa NU poate primi din baterie mai mult decat propriul deficit
    (P_bat,k = min(P_deficit,k, P_bat_cota,k)) - daca scorul ei ii aduce o
    cota mai mare decat are nevoie, DIFERENTA NEUTILIZATA se redistribuie,
    proportional cu scorul, intre celelalte case care inca au deficit
    neacoperit ("water-filling"): se repeta pana cand fie s-a acoperit tot
    deficitul, fie s-a epuizat toata puterea disponibila. Se termina mereu
    in cel mult len(deficit_k) runde (la fiecare runda cu redistribuire,
    cel putin o casa isi satureaza deficitul si iese definitiv din calcul).

    Returneaza dict id_casa -> P_bat,k (kW), cu proprietatile:
      0 <= P_bat,k <= deficit_k[k]  (bateria nu "supra-alimenteaza" o casa)
      sum(P_bat,k) <= putere_disponibila, cu egalitate daca deficitul total
      >= puterea disponibila (bateria descarca tot ce poate).
    """
    alocat = {id_casa: 0.0 for id_casa in deficit_k}
    ramas = dict(deficit_k)
    activi = {id_casa for id_casa, d in deficit_k.items() if d > epsilon}
    putere_ramasa = putere_disponibila

    while activi and putere_ramasa > epsilon:
        suma_scor = sum(scor_k[id_casa] for id_casa in activi)
        if suma_scor <= epsilon:
            break  # nu ar trebui sa se intample - scorurile sunt mereu >= SCOR_MINIM

        saturate = set()
        putere_folosita = 0.0
        for id_casa in activi:
            cota = putere_ramasa * scor_k[id_casa] / suma_scor
            luat = min(cota, ramas[id_casa])
            alocat[id_casa] += luat
            ramas[id_casa] -= luat
            putere_folosita += luat
            if ramas[id_casa] <= epsilon:
                saturate.add(id_casa)

        putere_ramasa -= putere_folosita
        if not saturate:
            # nimeni nu s-a saturat -> toata puterea a fost deja distribuita
            # proportional in aceasta runda, nu mai e nimic de redistribuit
            break
        activi -= saturate

    return alocat


# =============================================================================
# PASUL 3: modelul (comunitatea de 9 case) - discret in pasi, un pas = o ora
# =============================================================================
class ModelComunitateScoruri(mesa.Model):
    """
    Modelul multi-agent al comunitatii (9 case), cu comportament
    probabilistic si scor dinamic. La fel ca ModelComunitateEnergetica din
    simulare_multiagent.py: un AgentRecomandareComunitate + o lista de
    AgentCasaProbabilistica, activati in ordine FIXA (NU shuffle_do) -
    intai recomandarea, apoi casele, pentru ca fiecare casa reactioneaza la
    semnalul calculat DEJA in pasul curent.
    """

    def __init__(
        self,
        orizont,
        id_case,
        probabilitati_follow,
        banda=BANDA_TOLERANTA,
        interval_orar=INTERVAL_ORAR_RECOMANDARI,
        procent_ajustare=PROCENT_AJUSTARE,
        rosu_la_deficit=ROSU_LA_DEFICIT,
        fereastra_ore=FEREASTRA_SCOR_ORE,
        istoric_minim_ore=ISTORIC_MINIM_ORE,
        scor_implicit=SCOR_IMPLICIT,
        scor_minim=SCOR_MINIM,
        seed=SEED_IMPLICIT,
        Q_nom_kWh=Q_NOM_ESS_COMUNITATE_KWH,
        soc_min=SOC_MIN,
        soc_max=SOC_MAX,
        soc_initial=SOC_INITIAL,
        delta_soc_lim=DELTA_SOC_LIM,
        dt=DT_H,
    ):
        super().__init__(seed=seed)
        self.orizont = orizont
        self.n_pasi = len(orizont)
        self.pas_curent = 0
        self.semnal_curent = None
        self.ultima_productie_estimata = None
        self.ultimul_consum_total_estimat = None
        self.total_recomandari = {"rosu": 0, "verde": 0}

        # --- starea bateriei comunitare (Pasul 2) - ACEEASI baterie mare a
        # comunitatii, ACEEASI constrangeri de protectie ca in model_baterie.py
        # / simulare_comunitate_baterie.py (SOC_min=0.20, SOC_max=0.90,
        # SOC_initial=0.20, Delta_SOC_lim=0.35, Q_nom=50 kWh) - importate
        # direct de acolo, nu redefinite, ca sa nu diverga intre etape.
        self.Q_nom_kWh = Q_nom_kWh
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.dt = dt
        self.soc = soc_initial
        self.putere_limita_kw = delta_soc_lim * Q_nom_kWh / dt  # limita FIXA de putere (kW)
        self.ultimul_p_ess = None
        self.ultimul_soc_inceput = None
        self.ultimul_solar_k = {}
        self.ultimul_bat_k = {}
        self.ultimul_grid_k = {}

        self.case = [
            AgentCasaProbabilistica(
                self, id_casa=id_casa, p_follow=probabilitati_follow[id_casa],
                procent_ajustare=procent_ajustare, fereastra_ore=fereastra_ore,
                istoric_minim_ore=istoric_minim_ore, scor_implicit=scor_implicit,
                scor_minim=scor_minim,
            )
            for id_casa in id_case
        ]
        self.recomandare = AgentRecomandareComunitate(
            self, banda=banda, interval_orar=interval_orar,
            rosu_la_deficit=rosu_la_deficit,
        )

        reporters = {
            "Semnal": lambda m: m.semnal_curent,
            "Productie_Estimata_Wh": lambda m: m.ultima_productie_estimata,
            "Consum_Total_Estimat_Wh": lambda m: m.ultimul_consum_total_estimat,
            "P_ESS_kW": lambda m: m.ultimul_p_ess,
            "SOC": lambda m: m.ultimul_soc_inceput,
            "SOC_next": lambda m: m.soc,
        }
        for casa in self.case:
            reporters[f"Consum_Estimat_Wh_{casa.id_casa}"] = (
                lambda m, c=casa: c.ultimul_consum_estimat
            )
            reporters[f"Consum_Simulat_Wh_{casa.id_casa}"] = (
                lambda m, c=casa: c.ultimul_consum_simulat
            )
            reporters[f"Recomandare_Urmata_{casa.id_casa}"] = (
                lambda m, c=casa: c.ultima_recomandare_urmarita
            )
            reporters[f"Scor_{casa.id_casa}"] = (
                lambda m, c=casa: c.ultimul_scor
            )
            reporters[f"Solar_kW_{casa.id_casa}"] = (
                lambda m, c=casa: m.ultimul_solar_k.get(c.id_casa)
            )
            reporters[f"Bat_kW_{casa.id_casa}"] = (
                lambda m, c=casa: m.ultimul_bat_k.get(c.id_casa)
            )
            reporters[f"Grid_kW_{casa.id_casa}"] = (
                lambda m, c=casa: m.ultimul_grid_k.get(c.id_casa)
            )
        self.datacollector = mesa.DataCollector(model_reporters=reporters)

    def consum_estimat_casa(self, id_casa, pas):
        return float(self.orizont[f"Consum_Estimat_Wh_{id_casa}"].iloc[pas])

    def consum_total_estimat(self, pas):
        return float(sum(self.consum_estimat_casa(c.id_casa, pas) for c in self.case))

    def _pas_baterie(self):
        """
        Algoritmul de descarcare si alocare PONDERATA a bateriei comunitare
        (Pasul 2), rulat DUPA ce toate casele si-au calculat deja, in
        ACEEASI ora, consumul SIMULAT (Consum_Simulat_Wh, dupa ce au
        reactionat probabilistic la recomandare) si scorul curent
        (Scor, pe fereastra mobila de 60 de zile) - vezi AgentCasaProbabilistica.

        P_Load,comunitate(t) = suma consumurilor SIMULATE ale celor 9 case
        (NU a celor estimate - acelea sunt folosite DOAR de agentul de
        recomandari, ca sa decida semnalul; fluxul REAL de energie se face
        pe ce a consumat de fapt fiecare casa).

        Caz SURPLUS (P_PV >= P_Load,comunitate):
          fiecare casa isi acopera consumul integral din solar; surplusul
          comunitar incarca bateria (limitat de puterea maxima de incarcare
          la SOC-ul curent si de Delta_SOC_lim).

        Caz DEFICIT (P_PV < P_Load,comunitate):
          1. fiecare casa primeste o cota de solar PROPORTIONALA cu propriul
             consum: P_solar,k = P_PV * P_sim,k / P_Load,comunitate;
          2. deficitul ramas per casa: P_deficit,k = P_sim,k - P_solar,k;
          3. bateria descarca cel mult min(deficit total, putere maxima de
             descarcare la SOC-ul curent);
          4. acea putere se imparte intre case PROPORTIONAL CU SCORUL lor
             curent (aloca_baterie_ponderat_pe_scor - vezi docstring-ul ei
             pentru redistribuirea surplusului neutilizat de casele saturate);
          5. ce nu se acopera nici din solar, nici din baterie, vine din
             retea: P_Grid,k = P_deficit,k - P_bat,k (>= 0 prin constructie).

        Actualizeaza SOC (Ecuatia 10, ca in model_baterie.py):
          SOC(t+1) = SOC(t) - P_ESS(t) * dt / Q_nom   (p_ESS > 0 = descarcare,
          p_ESS < 0 = incarcare - ACEEASI conventie ca in model_baterie.py).
        """
        soc_inceput = self.soc
        p_pv_kw = self.ultima_productie_estimata / 1000.0  # Wh/ora -> kW
        consum_kw = {c.id_casa: c.ultimul_consum_simulat / 1000.0 for c in self.case}
        scoruri = {c.id_casa: c.ultimul_scor for c in self.case}
        p_load_com_kw = sum(consum_kw.values())

        if p_pv_kw >= p_load_com_kw:
            # --- SURPLUS solar: fiecare casa acoperita integral din solar ---
            solar_k = dict(consum_kw)
            bat_k = {id_casa: 0.0 for id_casa in consum_kw}
            grid_k = {id_casa: 0.0 for id_casa in consum_kw}

            delta_p = p_pv_kw - p_load_com_kw
            putere_max_incarcare_soc = max(0.0, (self.soc_max - soc_inceput) * self.Q_nom_kWh / self.dt)
            putere_max_incarcare = min(self.putere_limita_kw, putere_max_incarcare_soc)
            p_ess = -min(delta_p, putere_max_incarcare)
        else:
            # --- DEFICIT: solar proportional cu consumul + baterie ponderata pe scor ---
            if p_load_com_kw > 1e-12:
                solar_k = {
                    id_casa: p_pv_kw * p_sim / p_load_com_kw
                    for id_casa, p_sim in consum_kw.items()
                }
            else:
                solar_k = {id_casa: 0.0 for id_casa in consum_kw}
            deficit_k = {
                id_casa: max(0.0, consum_kw[id_casa] - solar_k[id_casa])
                for id_casa in consum_kw
            }
            deficit_total = sum(deficit_k.values())

            putere_max_descarcare_soc = max(0.0, (soc_inceput - self.soc_min) * self.Q_nom_kWh / self.dt)
            putere_max_descarcare = min(self.putere_limita_kw, putere_max_descarcare_soc)
            p_ess_discharge = min(deficit_total, putere_max_descarcare)

            bat_k = aloca_baterie_ponderat_pe_scor(deficit_k, scoruri, p_ess_discharge)
            grid_k = {id_casa: deficit_k[id_casa] - bat_k[id_casa] for id_casa in consum_kw}
            p_ess = p_ess_discharge

        soc_urmator = soc_inceput - p_ess * self.dt / self.Q_nom_kWh
        self.soc = min(max(soc_urmator, self.soc_min), self.soc_max)

        self.ultimul_p_ess = p_ess
        self.ultimul_soc_inceput = soc_inceput
        self.ultimul_solar_k = solar_k
        self.ultimul_bat_k = bat_k
        self.ultimul_grid_k = grid_k

    def step(self):
        """Un pas de model = o ora: recomandarea, apoi casele, apoi bateria."""
        self.recomandare.step()
        for casa in self.case:
            casa.step()
        self._pas_baterie()
        self.datacollector.collect(self)
        self.pas_curent += 1

    def ruleaza(self, n_pasi=None):
        n = self.n_pasi if n_pasi is None else min(n_pasi, self.n_pasi)
        for _ in range(n):
            self.step()
        rezultate = self.datacollector.get_model_vars_dataframe()
        rezultate.index = self.orizont.index[:len(rezultate)]
        rezultate.index.name = "Ora"
        return rezultate


# =============================================================================
# PASUL 4: rularea celor 3 scenarii + verificare in consola
# =============================================================================
def ruleaza_toate_scenariile(orizont, case_valide=CASE_VALIDE, scenarii=SCENARII, seed=SEED_IMPLICIT):
    """
    Ruleaza modelul o data per scenariu (probabilitatile din SCENARII),
    pastrand ACELASI orizont (nu se reia pipeline-ul IRISE/BATEM) si ACELASI
    seed (vezi nota despre reproductibilitate din docstring-ul fisierului).

    Returneaza (total_recomandari, scoruri_medii, rezultate_per_scenariu):
      total_recomandari   - dict {"rosu": n, "verde": n} - identic intre
                             scenarii, pentru ca semnalul foloseste consumul
                             ESTIMAT (nu cel simulat, care depinde de
                             comportamentul probabilistic al caselor).
      scoruri_medii        - dict {nume_scenariu: {id_casa: scor mediu}}
      rezultate_per_scenariu - dict {nume_scenariu: DataFrame rezultate}
    """
    total_recomandari_per_scenariu = {}
    scoruri_medii = {}
    rezultate_per_scenariu = {}

    for nume_scenariu, probabilitati in scenarii.items():
        print(f"\n=== Rulare {nume_scenariu} ===")
        model = ModelComunitateScoruri(
            orizont=orizont, id_case=case_valide,
            probabilitati_follow=probabilitati, seed=seed,
        )
        rezultate = model.ruleaza()
        rezultate_per_scenariu[nume_scenariu] = rezultate
        total_recomandari_per_scenariu[nume_scenariu] = dict(model.total_recomandari)

        scoruri_medii[nume_scenariu] = {
            id_casa: float(rezultate[f"Scor_{id_casa}"].mean()) for id_casa in case_valide
        }
        print(
            f"  Recomandari emise: rosu={model.total_recomandari['rosu']}, "
            f"verde={model.total_recomandari['verde']}"
        )

    # Semnalul e determinist (depinde doar de consumul ESTIMAT si de
    # productia PV, nu de comportamentul probabilistic al caselor) - deci
    # numarul de recomandari TREBUIE sa fie identic intre cele 3 scenarii.
    # Verificam explicit, nu presupunem in tacere.
    valori_unice = {tuple(sorted(v.items())) for v in total_recomandari_per_scenariu.values()}
    if len(valori_unice) != 1:
        print(
            "ATENTIE: numarul de recomandari difera intre scenarii - nu ar "
            "trebui, verificati logica AgentRecomandareComunitate.\n"
            f"  {total_recomandari_per_scenariu}"
        )
        total_recomandari = None
    else:
        total_recomandari = next(iter(total_recomandari_per_scenariu.values()))

    return total_recomandari, scoruri_medii, rezultate_per_scenariu


def afiseaza_total_recomandari(total_recomandari, n_ore):
    print(f"\n=== RECOMANDARI EMISE PE AN ({n_ore} ore simulate, ~{n_ore / 24:.0f} zile) ===")
    if total_recomandari is None:
        print("  (numarul de recomandari a diferit intre scenarii - vezi avertismentul de mai sus)")
        return
    total = total_recomandari["rosu"] + total_recomandari["verde"]
    print(f"  Semnal rosu  (deficit, comunitatea trebuie sa reduca consumul): {total_recomandari['rosu']:6d} ore")
    print(f"  Semnal verde (surplus, comunitatea poate consuma local):       {total_recomandari['verde']:6d} ore")
    print(f"  TOTAL recomandari active (rosu + verde):                      {total:6d} ore "
          f"({100 * total / n_ore:.1f}% din orele simulate)")


def afiseaza_tabel_comparativ_scoruri(scoruri_medii, case_valide=CASE_VALIDE):
    nume_scenarii = list(scoruri_medii.keys())
    print("\n=== SCORUL MEDIU PE CASA, COMPARAT PE CELE 3 SCENARII ===")
    header = f"{'Casa':>10s}"
    for nume in nume_scenarii:
        header += f"{nume:>26s}"
    print(header)
    for id_casa in case_valide:
        linie = f"{id_casa:>10d}"
        for nume in nume_scenarii:
            linie += f"{scoruri_medii[nume][id_casa]:>26.4f}"
        print(linie)

    print(f"\n{'Medie comunitate':>10s}", end="")
    for nume in nume_scenarii:
        medie = float(np.mean(list(scoruri_medii[nume].values())))
        print(f"{medie:>26.4f}", end="")
    print()


# =============================================================================
# PASUL 5: verificarea bilantului energetic + raportul de baterie (Pasul 2)
# =============================================================================
def afiseaza_verificare_bilant_energetic(rezultate, case_valide=CASE_VALIDE, dt=DT_H, toleranta_kw=1e-6):
    """
    Verifica, pentru FIECARE ora simulata, ca:
      suma_k (P_solar,k + P_bat,k + P_Grid,k) == suma_k P_sim,k
    (bilantul de putere al comunitatii - trebuie sa fie o IDENTITATE, nu o
    aproximare: fiecare Wh consumat de o casa vine din exact una din cele
    trei surse). Verifica separat si ca SOC ramane STRICT in [soc_min,
    soc_max] pe tot anul, si ca nicio casa nu a primit din baterie mai mult
    decat propriul deficit (P_Grid,k >= 0, altfel ar insemna ca am "inventat"
    energie).
    """
    consum_simulat_total_kw = sum(
        rezultate[f"Consum_Simulat_Wh_{id_casa}"] for id_casa in case_valide
    ) / 1000.0
    acoperire_totala_kw = sum(
        rezultate[f"Solar_kW_{id_casa}"] + rezultate[f"Bat_kW_{id_casa}"] + rezultate[f"Grid_kW_{id_casa}"]
        for id_casa in case_valide
    )
    diferenta = (acoperire_totala_kw - consum_simulat_total_kw).abs()
    abatere_maxima_kw = float(diferenta.max())

    grid_min = min(float(rezultate[f"Grid_kW_{id_casa}"].min()) for id_casa in case_valide)

    soc_min_atins = float(rezultate["SOC_next"].min())
    soc_max_atins = float(rezultate["SOC_next"].max())
    soc_in_limite = (soc_min_atins >= SOC_MIN - 1e-9) and (soc_max_atins <= SOC_MAX + 1e-9)

    print("\n=== VERIFICARE BILANT ENERGETIC (Solar + Baterie + Retea == Consum simulat) ===")
    print(f"Abaterea maxima (peste toate orele si casele), in kW: {abatere_maxima_kw:.2e}")
    print(
        "  -> OK, bilantul se inchide EXACT (in limita erorilor de rotunjire in virgula mobila)."
        if abatere_maxima_kw <= toleranta_kw else
        "  !!! ATENTIE: bilantul NU se inchide - verificati _pas_baterie()."
    )
    print(f"\nImportul minim din retea (per casa/ora), kW: {grid_min:.2e}")
    print(
        "  -> OK, nicio casa nu a primit vreodata mai multa baterie decat deficitul ei."
        if grid_min >= -toleranta_kw else
        "  !!! ATENTIE: import negativ din retea - alocarea bateriei a depasit deficitul unei case."
    )
    print(f"\nSOC (stare de incarcare baterie): min={soc_min_atins * 100:.2f}%, max={soc_max_atins * 100:.2f}% "
          f"(limite impuse: [{SOC_MIN * 100:.0f}%, {SOC_MAX * 100:.0f}%])")
    print(
        "  -> OK, SOC ramane STRICT in limite pe tot anul."
        if soc_in_limite else
        "  !!! ATENTIE: SOC a iesit din limitele impuse - verificati _pas_baterie()."
    )


def afiseaza_energie_descarcata_pe_casa(rezultate, case_valide=CASE_VALIDE, dt=DT_H):
    """Energia totala (kWh, pe tot anul) descarcata din bateria comunitara, per casa."""
    print("\n=== ENERGIE DESCARCATA DIN BATERIE PE CASA (Scenariul 3, an intreg) ===")
    print(f"{'Casa':>10s}{'Energie baterie (kWh)':>24s}{'Scor mediu':>14s}{'% din total bat.':>20s}{'% acoperit din consum':>24s}")

    energii = {
        id_casa: float(rezultate[f"Bat_kW_{id_casa}"].sum()) * dt
        for id_casa in case_valide
    }
    total_baterie_kWh = sum(energii.values())

    for id_casa in case_valide:
        scor_mediu = float(rezultate[f"Scor_{id_casa}"].mean())
        procent_bat = 100 * energii[id_casa] / total_baterie_kWh if total_baterie_kWh > 0 else 0.0
        
        # Extragem consumul simulat total al casei pentru a calcula metrica relativa
        consum_total_casa_kWh = float(rezultate[f"Consum_Simulat_Wh_{id_casa}"].sum()) / 1000.0
        procent_din_consum = 100 * energii[id_casa] / consum_total_casa_kWh if consum_total_casa_kWh > 0 else 0.0
        
        print(f"{id_casa:>10d}{energii[id_casa]:>24.2f}{scor_mediu:>14.4f}{procent_bat:>19.1f}%{procent_din_consum:>23.1f}%")

    print(f"{'TOTAL':>10s}{total_baterie_kWh:>24.2f}")

    energie_descarcata_model_kWh = float(rezultate.loc[rezultate["P_ESS_kW"] > 0, "P_ESS_kW"].sum()) * dt
    print(
        f"\nVerificare incrucisata: suma pe case = {total_baterie_kWh:.2f} kWh vs. "
        f"suma P_ESS_kW>0 la nivel de model = {energie_descarcata_model_kWh:.2f} kWh "
        f"(diferenta: {abs(total_baterie_kWh - energie_descarcata_model_kWh):.2e} kWh)"
    )


# =============================================================================
# PASUL 6: indicatorii de performanta (SS/SC) - individuali si comunitari
# =============================================================================
def calculeaza_indicatori_performanta(rezultate, case_valide=CASE_VALIDE, dt=DT_H):
    """
    Calculeaza indicatorii de performanta ai Pasului 3, pentru UN SINGUR
    scenariu (rezultate = DataFrame-ul intors de model.ruleaza(), care
    contine deja coloanele per casa Solar_kW_/Bat_kW_/Grid_kW_/
    Consum_Simulat_Wh_ si cele de comunitate Productie_Estimata_Wh/
    P_ESS_kW, populate de _pas_baterie la Pasul 2).

    Per casa (E_x,k in kWh, pe tot orizontul simulat):
      E_load,k   = suma Consum_Simulat_Wh,k / 1000
      E_solar,k  = suma Solar_kW,k * dt
      E_bat,k    = suma Bat_kW,k * dt
      E_grid,k   = suma Grid_kW,k * dt
      E_pv_cota,k = suma [P_PV(t) * P_sim,k(t)/P_Load,comunitate(t)] * dt -
                    "cota" NOMINALA a casei k din productia TOTALA a
                    centralei comunitare, proportionala cu propriul consum
                    la fiecare ora (aceeasi regula de "cota-parte" folosita
                    peste tot in proiect - vezi panouri_pentru() din
                    config_proiect.py) - DIFERITA de Solar_k (care e doar
                    energia REAL primita direct, plafonata la consumul
                    propriu in orele de deficit): E_pv_cota,k conteaza si
                    partea "virtuala" din surplusul orelor de soare, chiar
                    daca acel surplus s-a dus, de fapt, in baterie.
      SS_k = 100 * (E_solar,k + E_bat,k) / E_load,k     (autonomie individuala)
      SC_k = 100 * (E_solar,k + E_bat,k) / E_pv_cota,k  (autoconsum atribuit)

    La nivel de comunitate, SS_com/SC_com se calculeaza cu calculeaza_SC_SS
    din model_baterie.py (Ecuatiile 1-2 din articol) - ACELASI cod deja
    folosit si in simulare_comunitate_baterie.py, aplicat pe productia si
    consumul TOTALE ale comunitatii si pe P_ESS_kW deja simulat la Pasul 2.

    Returneaza (indicatori_per_casa, indicatori_comunitate):
      indicatori_per_casa   - DataFrame indexat pe id_casa, cu coloanele
        E_load_kWh, E_solar_kWh, E_bat_kWh, E_grid_kWh, E_pv_cota_kWh,
        SS_procent, SC_procent.
      indicatori_comunitate - dict cu SS_com_procent, SC_com_procent,
        E_load_com_MWh, E_pv_com_MWh, E_grid_com_MWh.
    """
    p_pv_kw = rezultate["Productie_Estimata_Wh"].to_numpy() / 1000.0
    p_load_com_kw = sum(
        rezultate[f"Consum_Simulat_Wh_{id_casa}"] for id_casa in case_valide
    ).to_numpy() / 1000.0
    p_ess_kw = rezultate["P_ESS_kW"].to_numpy()

    randuri = []
    for id_casa in case_valide:
        consum_k_kw = rezultate[f"Consum_Simulat_Wh_{id_casa}"].to_numpy() / 1000.0
        solar_k_kw = rezultate[f"Solar_kW_{id_casa}"].to_numpy()
        bat_k_kw = rezultate[f"Bat_kW_{id_casa}"].to_numpy()
        grid_k_kw = rezultate[f"Grid_kW_{id_casa}"].to_numpy()

        E_load = float(np.sum(consum_k_kw) * dt)
        E_solar = float(np.sum(solar_k_kw) * dt)
        E_bat = float(np.sum(bat_k_kw) * dt)
        E_grid = float(np.sum(grid_k_kw) * dt)

        cota_pv_k_kw = np.divide(
            p_pv_kw * consum_k_kw, p_load_com_kw,
            out=np.zeros_like(p_pv_kw), where=p_load_com_kw > 1e-9,
        )
        E_pv_cota = float(np.sum(cota_pv_k_kw) * dt)

        SS_k = 100 * (E_solar + E_bat) / E_load if E_load > 0 else float("nan")
        SC_k = 100 * (E_solar + E_bat) / E_pv_cota if E_pv_cota > 0 else float("nan")

        # Indicator SUPLIMENTAR (nu cerut explicit de formulele profesorului,
        # dar necesar ca sa se VADA clar efectul scorului): cate kWh de
        # BATERIE a primit casa k, raportat STRICT la propriul ei consum
        # anual (nu la E_load - E_solar, care amesteca inca un factor
        # proportional cu consumul). SS_k amesteca DOUA lucruri care nu au
        # legatura cu scorul - marimea casei (E_load la numitor) si cota
        # solara directa (E_solar, proportionala cu consumul, nu cu scorul)
        # - motiv pentru care compararea BRUTA a SS_k intre case de marimi
        # foarte diferite (ex. o casa cu 6864 kWh/an langa una cu 1133
        # kWh/an) poate ascunde complet efectul scorului. Indicatorul de mai
        # jos izoleaza DOAR contributia bateriei, per kWh de consum propriu -
        # comparabil corect intre case de marimi diferite.
        Bat_per_Load = 100 * E_bat / E_load if E_load > 0 else float("nan")

        randuri.append({
            "id_casa": id_casa,
            "E_load_kWh": E_load, "E_solar_kWh": E_solar,
            "E_bat_kWh": E_bat, "E_grid_kWh": E_grid,
            "E_pv_cota_kWh": E_pv_cota,
            "SS_procent": SS_k, "SC_procent": SC_k,
            "Bat_per_Load_procent": Bat_per_Load,
        })

    indicatori_per_casa = pd.DataFrame(randuri).set_index("id_casa")

    SC_com, SS_com = calculeaza_SC_SS(p_pv_kw, p_load_com_kw, p_ess_kw, dt=dt)
    indicatori_comunitate = {
        "SS_com_procent": 100 * SS_com,
        "SC_com_procent": 100 * SC_com,
        "E_load_com_MWh": float(indicatori_per_casa["E_load_kWh"].sum()) / 1000.0,
        "E_pv_com_MWh": float(np.sum(p_pv_kw) * dt) / 1000.0,
        "E_grid_com_MWh": float(indicatori_per_casa["E_grid_kWh"].sum()) / 1000.0,
    }
    return indicatori_per_casa, indicatori_comunitate


def calculeaza_rata_urmarire(rezultate, case_valide=CASE_VALIDE):
    """
    Rata EMPIRICA de urmare a recomandarilor (%), agregata pe toate cele 9
    case si pe tot orizontul simulat:
        rata = 100 * (total recomandari urmate) / (total recomandari primite)
    DISTINCTA de scorul mediu (Scor_k mediu in timp): scorul e o medie a
    valorilor instantanee ale ferestrei mobile (care porneste implicit la
    0.50 in primele 7 zile ale fiecarei case), in timp ce rata de aici e
    raportul brut, pe tot anul, fara valoarea implicita.
    """
    urmate, primite = 0, 0
    for id_casa in case_valide:
        col = rezultate[f"Recomandare_Urmata_{id_casa}"]
        primite += int(col.notna().sum())
        urmate += int((col == 1).sum())
    return 100 * urmate / primite if primite > 0 else float("nan")


def calculeaza_toti_indicatorii(rezultate_per_scenariu, case_valide=CASE_VALIDE, dt=DT_H):
    """Aplica calculeaza_indicatori_performanta + calculeaza_rata_urmarire pe fiecare scenariu."""
    indicatori_per_casa_per_scenariu = {}
    indicatori_comunitate_per_scenariu = {}
    rate_urmarire_per_scenariu = {}

    for nume_scenariu, rezultate in rezultate_per_scenariu.items():
        indicatori_per_casa, indicatori_comunitate = calculeaza_indicatori_performanta(
            rezultate, case_valide, dt=dt,
        )
        indicatori_per_casa_per_scenariu[nume_scenariu] = indicatori_per_casa
        indicatori_comunitate_per_scenariu[nume_scenariu] = indicatori_comunitate
        rate_urmarire_per_scenariu[nume_scenariu] = calculeaza_rata_urmarire(rezultate, case_valide)

    return indicatori_per_casa_per_scenariu, indicatori_comunitate_per_scenariu, rate_urmarire_per_scenariu


def afiseaza_tabel_comparativ_scenarii(indicatori_comunitate_per_scenariu, rate_urmarire_per_scenariu):
    print("\n=== TABEL COMPARATIV INTRE CELE 3 SCENARII ===")
    print(
        f"{'Scenariu':24s}{'Rata medie urmarire (%)':>26s}{'SS Comunitar (%)':>20s}"
        f"{'SC Comunitar (%)':>20s}{'Import Retea Total (MWh)':>27s}"
    )
    for nume, ind in indicatori_comunitate_per_scenariu.items():
        print(
            f"{nume:24s}{rate_urmarire_per_scenariu[nume]:>26.2f}"
            f"{ind['SS_com_procent']:>20.2f}{ind['SC_com_procent']:>20.2f}"
            f"{ind['E_grid_com_MWh']:>27.2f}"
        )


def afiseaza_tabel_detaliat_scenariul_3(
    indicatori_per_casa, scoruri_medii_scenariu, probabilitati=SCENARIUL_3_ETEROGEN,
    case_valide=CASE_VALIDE,
):
    print("\n=== TABEL DETALIAT PER CASA - Scenariul 3 (Eterogen) ===")
    print(
        f"{'Casa':>10s}{'p_follow setata (%)':>22s}{'Scor mediu (%)':>18s}"
        f"{'Consum (kWh)':>16s}{'Din baterie (kWh)':>20s}{'Din retea (kWh)':>18s}"
        f"{'Autonomie SS (%)':>20s}{'Baterie/consum (%)':>21s}"
    )
    for id_casa in case_valide:
        ind = indicatori_per_casa.loc[id_casa]
        print(
            f"{id_casa:>10d}{100 * probabilitati[id_casa]:>22.1f}"
            f"{100 * scoruri_medii_scenariu[id_casa]:>18.2f}"
            f"{ind['E_load_kWh']:>16.1f}{ind['E_bat_kWh']:>20.1f}"
            f"{ind['E_grid_kWh']:>18.1f}{ind['SS_procent']:>20.2f}"
            f"{ind['Bat_per_Load_procent']:>21.2f}"
        )
    print(
        "\n(coloana 'Baterie/consum' = cati kWh de baterie a primit fiecare "
        "casa, la fiecare 100 kWh consumati de ea intr-un an - comparatie "
        "corecta intre casele mari si cele mici, spre deosebire de coloanele "
        "brute de mai sus.)"
    )

    # Concluzia ceruta de profesor foloseste SS_k (autonomia bruta), dar SS_k
    # amesteca marimea casei (E_load la numitor) si cota solara (proportionala
    # cu consumul, nu cu scorul) peste efectul bateriei - pe o comunitate cu
    # case foarte inegale ca marime (consumul anual variaza de peste 6 ori
    # intre cea mai mica si cea mai mare casa), asta poate ascunde efectul
    # scorului in SS_k brut. Raportam deci DOUA corelatii:
    #   r_SS       = Scor vs. SS_k (indicatorul CERUT, dar posibil "murdarit"
    #                de marimea casei)
    #   r_baterie  = Scor vs. Bat_per_Load (kWh de baterie primiti la fiecare
    #                100 kWh consumati - comparatie CORECTA intre case mari
    #                si mici, izoleaza mai bine efectul PROPRIU al scorului)
    scoruri_arr = np.array([scoruri_medii_scenariu[c] for c in case_valide])
    ss_arr = np.array([indicatori_per_casa.loc[c, "SS_procent"] for c in case_valide])
    bat_per_load_arr = np.array([indicatori_per_casa.loc[c, "Bat_per_Load_procent"] for c in case_valide])
    r_SS = float(np.corrcoef(scoruri_arr, ss_arr)[0, 1])
    r_baterie = float(np.corrcoef(scoruri_arr, bat_per_load_arr)[0, 1])

    casa_min_scor = min(case_valide, key=lambda c: scoruri_medii_scenariu[c])
    casa_max_scor = max(case_valide, key=lambda c: scoruri_medii_scenariu[c])
    print(
        f"\nConcluzie: casele nu consuma toate la fel (unele consuma de mai "
        f"multe ori mai mult decat altele intr-un an), deci compararea DIRECTA "
        f"a autonomiei (SS) intre case mari si mici e inselatoare - e ca si "
        f"cum ai compara doi angajati dupa salariul TOTAL, nu dupa salariul "
        f"pe ora. Comparatia corecta e cat de multa baterie a primit fiecare "
        f"casa RAPORTAT LA CAT CONSUMA EA (coloana 'Baterie/consum' din "
        f"tabelul de mai sus).\n"
        f"Casa {casa_min_scor} (scorul cel mai MIC, "
        f"{100 * scoruri_medii_scenariu[casa_min_scor]:.0f}%) a primit "
        f"{indicatori_per_casa.loc[casa_min_scor, 'Bat_per_Load_procent']:.1f} "
        f"kWh de baterie la fiecare 100 kWh consumati, iar casa {casa_max_scor} "
        f"(scorul cel mai MARE, {100 * scoruri_medii_scenariu[casa_max_scor]:.0f}%) "
        f"a primit "
        f"{indicatori_per_casa.loc[casa_max_scor, 'Bat_per_Load_procent']:.1f} "
        f"kWh la fiecare 100 kWh consumati.\n"
        f"Corelatia scor - baterie/consum (comparatia corecta): r = {r_baterie:+.3f}"
        f"   |   corelatia scor - autonomie SS bruta: r = {r_SS:+.3f}"
    )
    if r_baterie > 0.2:
        print(
            "Interpretare: dupa ce eliminam efectul marimii casei, se vede "
            "clar ca locuintele cu scor mai mare primesc, proportional cu cat "
            "consuma, mai multa energie din bateria comunitara - mecanismul "
            "de recompensare a cooperarii functioneaza asa cum a fost "
            "proiectat."
        )
    else:
        print(
            "Interpretare: nici dupa ce eliminam efectul marimii casei, "
            "corelatia nu e puternica - pe acest interval, bateria (50 kWh) "
            "pare prea mica fata de cererea celor 9 case ca efectul scorului "
            "sa iasa clar in evidenta la nivel anual, desi mecanismul de "
            "alocare a fost validat exact, ora cu ora, la Pasul 2."
        )


# =============================================================================
# PASUL 7: graficele Plotly (Pasul 3 - sinteza finala)
# =============================================================================
def genereaza_grafic_scoruri_dinamice(
    rezultate_scenariu, case_valide=CASE_VALIDE, nume_scenariu="Scenariul 3 (Eterogen)",
    nume_fisier="grafic_scoruri_dinamice_comunitate.html",
):
    """
    Evolutia in timp a Score_k(t) pentru toate cele 9 case, pe fereastra
    mobila de 60 de zile - o linie per casa, culoare consecventa
    (CULOARE_CASA). Scenariul implicit e Scenariul 3 (Eterogen), cel mai
    ilustrativ: toate scorurile pornesc de la valoarea implicita 0.50 (in
    primele 7 zile) si apoi DIVERG treptat spre propriile probabilitati
    p_follow, pe masura ce se acumuleaza istoric real in fereastra mobila.
    """
    fig = go.Figure()
    for id_casa in case_valide:
        fig.add_trace(go.Scatter(
            x=rezultate_scenariu.index, y=rezultate_scenariu[f"Scor_{id_casa}"],
            name=f"Casa {id_casa}", line=dict(color=CULOARE_CASA.get(id_casa)),
        ))
    fig.add_hline(
        y=SCOR_IMPLICIT, line=dict(color="black", width=1, dash="dot"),
        annotation_text=f"Scor implicit (primele {ISTORIC_MINIM_ORE // 24} zile) = {SCOR_IMPLICIT}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=(
            f"Evolutia scorului dinamic Score_k(t) - fereastra mobila de "
            f"{FEREASTRA_SCOR_ORE // 24} zile ({nume_scenariu})"
        ),
        hovermode="x unified",
        xaxis_title="Data si ora", yaxis_title="Scor (recomandari urmate / primite)",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"\nGrafic scoruri dinamice salvat ca: {nume_fisier}")
    return fig


def genereaza_grafic_comparativ_ss_scor(
    indicatori_per_casa, scoruri_medii_scenariu, case_valide=CASE_VALIDE,
    nume_scenariu="Scenariul 3 (Eterogen)", nume_fisier="grafic_comparativ_ss_per_casa.html",
):
    """
    Doua bar-chart-uri alaturate, per casa:
      (stanga)  Scor mediu (%) vs. Autonomie SS (%) - comparatia CERUTA de
                profesor, calculata exact dupa formula lui.
      (dreapta) Scor mediu (%) vs. Baterie primita / consum propriu (%) -
                ACEEASI comparatie, dar controlata pentru marimea casei (o
                casa care consuma de 6 ori mai mult ca alta nu mai "ascunde"
                efectul scorului): cati kWh de baterie a primit fiecare casa,
                la fiecare 100 kWh consumati de ea - permite compararea
                corecta intre case mari si mici.

    SS_k amesteca marimea casei si cota solara directa (proportionala cu
    consumul, NU cu scorul) peste contributia bateriei - pe o comunitate cu
    case foarte inegale ca marime, panoul din dreapta arata mult mai clar
    efectul scorului decat panoul din stanga (cerut de formula profesorului,
    dar sensibil la marimea casei).
    """
    x = [str(c) for c in case_valide]
    scor_pct = [100 * scoruri_medii_scenariu[c] for c in case_valide]
    ss_pct = [float(indicatori_per_casa.loc[c, "SS_procent"]) for c in case_valide]
    bat_per_load_pct = [float(indicatori_per_casa.loc[c, "Bat_per_Load_procent"]) for c in case_valide]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Scor vs. Autonomie SS (%) - formula profesorului",
            "Scor vs. Baterie primita / consum propriu (%) - corectat pt. marimea casei",
        ),
    )
    fig.add_trace(
        go.Bar(x=x, y=scor_pct, name="Scor mediu (%)", marker_color="#1f77b4", legendgroup="scor"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=x, y=ss_pct, name="Autonomie SS (%)", marker_color="#2ca02c"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=x, y=scor_pct, name="Scor mediu (%)", marker_color="#1f77b4",
               legendgroup="scor", showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=x, y=bat_per_load_pct, name="Baterie/consum (%)", marker_color="#ff7f0e"),
        row=1, col=2,
    )
    fig.update_layout(
        barmode="group",
        title=f"Scor mediu vs. beneficiul primit din baterie, pe casa - {nume_scenariu}",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Casa", row=1, col=1)
    fig.update_xaxes(title_text="Casa", row=1, col=2)
    fig.update_yaxes(title_text="Procent (%)", row=1, col=1)
    fig.update_yaxes(title_text="Procent (%)", row=1, col=2)
    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"Grafic comparativ scor/SS per casa salvat ca: {nume_fisier}")
    return fig


def genereaza_grafic_comparatie_scenarii(
    indicatori_comunitate_per_scenariu, nume_fisier="grafic_comparatie_scenarii_comunitate.html",
):
    """
    Doua bar-chart-uri ALATURATE (subplot-uri, NU o axa Y secundara pe
    acelasi plot): Autonomia comunitara SS_com (%) in stanga, Energia
    importata din retea (MWh) in dreapta.

    DE CE NU axa Y secundara (varianta initiala, corectata acum): cand doua
    serii de BARE sunt puse pe acelasi subplot cu axe Y diferite (SS in
    procente, ~55-65%, vs. import in MWh, ~10-11), Plotly NU le grupeaza
    corect side-by-side (barmode="group" nu se aplica intre trace-uri de pe
    axe Y diferite) - bareLE se suprapun vizual la aceeasi pozitie X, iar
    fiecare bara ocupa aproape toata inaltimea plot-ului RAPORTAT LA PROPRIA
    ei scara (auto-range), dand impresia gresita ca importul din retea "
    "domina" graficul. Doua subplot-uri separate, fiecare cu propria scara,
    elimina complet aceasta ambiguitate.
    """
    nume_scenarii = list(indicatori_comunitate_per_scenariu.keys())
    ss_com = [indicatori_comunitate_per_scenariu[n]["SS_com_procent"] for n in nume_scenarii]
    grid_mwh = [indicatori_comunitate_per_scenariu[n]["E_grid_com_MWh"] for n in nume_scenarii]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Autonomie comunitara SS (%)", "Import din retea (MWh)"),
    )
    fig.add_trace(
        go.Bar(
            x=nume_scenarii, y=ss_com, name="SS comunitar (%)", marker_color="#2ca02c",
            text=[f"{v:.1f}%" for v in ss_com], textposition="outside",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=nume_scenarii, y=grid_mwh, name="Import retea (MWh)", marker_color="#d62728",
            text=[f"{v:.2f}" for v in grid_mwh], textposition="outside",
        ),
        row=1, col=2,
    )
    fig.update_layout(
        title="Comparatie intre scenarii: autonomie comunitara (SS) vs. import din retea",
        showlegend=False,
    )
    fig.update_yaxes(title_text="SS comunitar (%)", row=1, col=1)
    fig.update_yaxes(title_text="Import din retea (MWh)", row=1, col=2)
    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"Grafic comparatie scenarii salvat ca: {nume_fisier}")
    return fig

# =============================================================================
# GRAFICE CLARE PENTRU PREZENTARE
# =============================================================================

def genereaza_grafic_balanta_anuala_comunitate(
    rezultate_scenariu, case_valide=CASE_VALIDE, nume_fisier="grafic_1_balanta_anuala_comunitate.html"
):
    """
    GRAFICUL 1: Totaluri lunare - Consum Estimat vs. Consum Simulat (dupa recomandari)
    si acoperirea energetica (Solar direct + Descarcare Baterie + Import Retea).
    """
    df = rezultate_scenariu.copy()
    
    # Agregam la nivel de comunitate (in kWh)
    consum_est_kWh = sum(df[f"Consum_Estimat_Wh_{c}"] for c in case_valide) / 1000.0
    consum_sim_kWh = sum(df[f"Consum_Simulat_Wh_{c}"] for c in case_valide) / 1000.0
    solar_kWh = sum(df[f"Solar_kW_{c}"] for c in case_valide)
    bat_kWh = sum(df[f"Bat_kW_{c}"] for c in case_valide)
    grid_kWh = sum(df[f"Grid_kW_{c}"] for c in case_valide)
    
    lunar = pd.DataFrame({
        "Consum Estimat (IRISE)": consum_est_kWh,
        "Consum Simulat (Dupa Recomandari)": consum_sim_kWh,
        "Acoperit din Solar": solar_kWh,
        "Acoperit din Baterie": bat_kWh,
        "Importat din Retea": grid_kWh,
    }, index=df.index).resample("MS").sum()
    
    luni_str = [t.strftime("%b %Y") for t in lunar.index]
    
    fig = go.Figure()
    # Bare suprapuse pentru cum a fost acoperit consumul
    fig.add_trace(go.Bar(x=luni_str, y=lunar["Acoperit din Solar"], name="Solar Direct (kWh)", marker_color="#e6a800"))
    fig.add_trace(go.Bar(x=luni_str, y=lunar["Acoperit din Baterie"], name="Baterie Comunitara (kWh)", marker_color="#2ca02c"))
    fig.add_trace(go.Bar(x=luni_str, y=lunar["Importat din Retea"], name="Import Retea (kWh)", marker_color="#d62728"))
    
    # Linii peste bare pentru consumul inainte si dupa recomandari
    fig.add_trace(go.Scatter(
        x=luni_str, y=lunar["Consum Estimat (IRISE)"], name="Consum Initial Estimat (kWh)",
        mode="lines+markers", line=dict(color="#1f77b4", width=3, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=luni_str, y=lunar["Consum Simulat (Dupa Recomandari)"], name="Consum Simulat (kWh)",
        mode="lines+markers", line=dict(color="#000000", width=2)
    ))
    
    fig.update_layout(
        barmode="stack",
        title="Graficul 1: Balanta Energetica Lunara a Comunitatii (Solar + Baterie + Retea vs. Recomandari)",
        xaxis_title="Luna", yaxis_title="Energie Lunara (kWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"\n[OK] Grafic 1 salvat ca: {nume_fisier}")
    return fig


def genereaza_grafic_demonstratie_alocare_48h(
    rezultate_scenariu, casa_activa=2000909, casa_pasiva=2000901,
    start_date="1998-10-15", ore_durata=48,
    nume_fisier="grafic_3_demonstratie_baterie_48h.html"
):
    """
    GRAFICUL 3: Zoom pe 48h - Compara direct Casa Activa (Scor ~90%) vs. Casa Pasiva (Scor ~15%).
    Arata cum la orele de seara bateria merge la casa cu scor mare.
    """
    # Selectam intervalul de 48h
    start_ts = pd.to_datetime(start_date).tz_localize(rezultate_scenariu.index.tz)
    end_ts = start_ts + pd.Timedelta(hours=ore_durata)
    sub = rezultate_scenariu.loc[start_ts:end_ts]
    
    if sub.empty:
        # Fallback daca data e in afara intervalului
        sub = rezultate_scenariu.iloc[2000:2000+ore_durata]
        
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=(
            f"Casa Activa {casa_activa} (Scor Mare ~90%) - Beneficiaza masiv de Baterie",
            f"Casa Pasiva {casa_pasiva} (Scor Mic ~15%) - Fortata sa importe din Retea"
        )
    )
    
    # Casa Activa (Sus)
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub[f"Consum_Simulat_Wh_{casa_activa}"]/1000.0,
        name="Consum Necesar (kW)", line=dict(color="black", width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub[f"Solar_kW_{casa_activa}"],
        name="Din Solar (kW)", line=dict(color="#e6a800"), fill='tozeroy'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub[f"Solar_kW_{casa_activa}"] + sub[f"Bat_kW_{casa_activa}"],
        name="Din Baterie (kW)", line=dict(color="#2ca02c"), fill='tonexty'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub[f"Solar_kW_{casa_activa}"] + sub[f"Bat_kW_{casa_activa}"] + sub[f"Grid_kW_{casa_activa}"],
        name="Din Retea (kW)", line=dict(color="#d62728"), fill='tonexty'
    ), row=1, col=1)

    # Casa Pasiva (Jos)
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub[f"Consum_Simulat_Wh_{casa_pasiva}"]/1000.0,
        name="Consum Necesar (kW)", line=dict(color="black", width=2), showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub[f"Solar_kW_{casa_pasiva}"],
        name="Din Solar (kW)", line=dict(color="#e6a800"), fill='tozeroy', showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub[f"Solar_kW_{casa_pasiva}"] + sub[f"Bat_kW_{casa_pasiva}"],
        name="Din Baterie (kW)", line=dict(color="#2ca02c"), fill='tonexty', showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub[f"Solar_kW_{casa_pasiva}"] + sub[f"Bat_kW_{casa_pasiva}"] + sub[f"Grid_kW_{casa_pasiva}"],
        name="Din Retea (kW)", line=dict(color="#d62728"), fill='tonexty', showlegend=False
    ), row=2, col=1)

    fig.update_layout(
        title="Graficul 3: Demonstratia Alocarii Bateriei in Orele de Seara (48h)",
        hovermode="x unified",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Putere (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Putere (kW)", row=2, col=1)
    fig.update_xaxes(title_text="Data si Ora", row=2, col=1)
    
    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"[OK] Grafic 3 salvat ca: {nume_fisier}")
    return fig

# =============================================================================
# ZONA DE COMENZI
# =============================================================================
if __name__ == "__main__":
    orizont, pv_system = pregateste_orizont_comunitate(CASE_VALIDE, number_of_panels=PANOURI_COMUNITATE)

    total_recomandari, scoruri_medii, rezultate_per_scenariu = ruleaza_toate_scenariile(
        orizont, case_valide=CASE_VALIDE, scenarii=SCENARII, seed=SEED_IMPLICIT,
    )

    afiseaza_total_recomandari(total_recomandari, len(orizont))
    afiseaza_tabel_comparativ_scoruri(scoruri_medii, CASE_VALIDE)

    # Scorurile medii salvate acum - "materia prima" pentru Pasul 2 (alocarea
    # ponderata a bateriei comunitare pe baza acestui scor).
    df_scoruri = pd.DataFrame(scoruri_medii).loc[CASE_VALIDE]
    df_scoruri.index.name = "id_casa"
    nume_csv = "scoruri_medii_case_scenarii.csv"
    df_scoruri.to_csv(nume_csv)
    print(f"\nScorurile medii per casa/scenariu salvate ca: {nume_csv}")

    # --- Pasul 2: algoritmul de descarcare + alocare ponderata a bateriei --
    # Bateria comunitara (50 kWh) a fost simulata pentru TOATE cele 3
    # scenarii (in acelasi model.step(), ca sa fie mereu disponibila si
    # consistenta cu scorurile din tabelul de mai sus), dar verificarea si
    # raportul cerute de profesor se fac aici pe Scenariul 3 (Eterogen) -
    # scenariul cu probabilitati diferentiate, cel mai relevant pentru
    # alocarea PONDERATA (Scenariile 1 si 2 au acelasi p_follow pentru toate
    # casele, deci scorurile lor converg spre aceeasi valoare si ponderea
    # bateriei devine aproape egala intre case - mai putin interesant de
    # verificat).
    print("\n\n" + "=" * 79)
    print("PASUL 2: ALOCAREA PONDERATA A BATERIEI COMUNITARE - Scenariul 3 (Eterogen)")
    print("=" * 79)
    rezultate_scenariul_3 = rezultate_per_scenariu["Scenariul 3 (Eterogen)"]

    afiseaza_verificare_bilant_energetic(rezultate_scenariul_3, CASE_VALIDE)
    afiseaza_energie_descarcata_pe_casa(rezultate_scenariul_3, CASE_VALIDE)

    # --- Pasul 3: indicatori de performanta (SS/SC), sinteza comparativa ---
    print("\n\n" + "=" * 79)
    print("PASUL 3: INDICATORI DE PERFORMANTA (SS/SC) + SINTEZA COMPARATIVA")
    print("=" * 79)

    (indicatori_per_casa_per_scenariu, indicatori_comunitate_per_scenariu,
     rate_urmarire_per_scenariu) = calculeaza_toti_indicatorii(rezultate_per_scenariu, CASE_VALIDE)

    afiseaza_tabel_comparativ_scenarii(indicatori_comunitate_per_scenariu, rate_urmarire_per_scenariu)
    afiseaza_tabel_detaliat_scenariul_3(
        indicatori_per_casa_per_scenariu["Scenariul 3 (Eterogen)"],
        scoruri_medii["Scenariul 3 (Eterogen)"],
        SCENARIUL_3_ETEROGEN, CASE_VALIDE,
    )

    # --- graficele Plotly (Pasul 3) -----------------------------------------
    print("\n=== GENERARE GRAFICE ===")
    genereaza_grafic_scoruri_dinamice(rezultate_scenariul_3, CASE_VALIDE)
    genereaza_grafic_comparativ_ss_scor(
        indicatori_per_casa_per_scenariu["Scenariul 3 (Eterogen)"],
        scoruri_medii["Scenariul 3 (Eterogen)"], CASE_VALIDE,
    )
    genereaza_grafic_comparatie_scenarii(indicatori_comunitate_per_scenariu)

    print("\n=== PASUL 3 COMPLET: cele 3 grafice HTML au fost salvate in directorul curent. ===")

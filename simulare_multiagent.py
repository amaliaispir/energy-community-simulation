"""
=============================================================================
SIMULARE MULTI-AGENT (Mesa) - comunitate energetica cu agent de recomandari
=============================================================================

Etapa noua a proiectului, peste modelarea PV si consumul orar deja facute
(modelare_pv.py, pv_comunitate.py). Aici construim un MODEL MULTI-AGENT cu
biblioteca Mesa (https://mesa.readthedocs.io), dupa tiparul din tutorialul
oficial "Boltzmann Wealth Model"
(https://mesa.readthedocs.io/latest/examples/basic/boltzmann_wealth_model.html):
o clasa Agent (mesa.Agent), o clasa Model (mesa.Model) care tine agentii si
un DataCollector, si un model.step() care avanseaza simularea cu un pas.

IMPORTANT (compatibilitate): Mesa 3.1+ cere Python >= 3.11, dar acest
proiect ruleaza pe Python 3.10 (vezi __pycache__/*.cpython-310.pyc). De
aceea foloseste explicit Mesa 3.0.x, ultima varianta compatibila cu 3.10:
    pip install "mesa==3.0.3"

CE MODELEAZA acest script (asa cum a fost discutat cu profesorul):

  - Fiecare CASA e un agent (AgentCasa). O comunitate completa va avea, in
    viitor, 9 agenti-casa (cele 9 case valide din baza IRISE - vezi
    MAIN_corectat.py). PENTRU ACEASTA ETAPA, comunitatea are DOAR O CASA
    (ID_CASE_COMUNITATE mai jos, cu CASA_IMPLICITA din config_proiect.py -
    ACEEASI casa analizata si in modelare_pv.py) - cadrul e insa scris ca sa
    se extinda direct la 9, doar adaugand ID-urile in ID_CASE_COMUNITATE.

  - CENTRALA folosita e cota-parte a caselor din comunitate din centrala
    dimensionata la Etapa 2 (pv_comunitate.py): 40 de panouri pentru 9 case,
    deci 4 panouri pentru comunitatea de o casa de acum. Cand adaugati toate
    cele 9 case, numarul revine automat la 40 - vezi panouri_pentru() in
    config_proiect.py.

  - Un agent special, AgentRecomandare - un singur exemplar, "creierul"
    comunitatii - se uita, la fiecare pas, la consumul ESTIMAT (din baza
    IRISE, agregat orar) si la productia ESTIMATA a centralei fotovoltaice
    (modelul solar BATEM) si da un semnal de tip semafor (ROSU/GALBEN/
    VERDE). Ambele valori sunt ESTIMARI, nu masuratori live: consumul e
    profilul istoric al casei folosit ca profil asteptat, iar productia e
    cea calculata de modelul solar - simuland exact situatia reala in care
    sistemul de recomandari trebuie sa decida INAINTE sa stie exact ce se
    va consuma/produce.

  - Fiecare AgentCasa, la pasul lui (dupa ce AgentRecomandare a calculat
    DEJA semnalul, in acelasi pas), isi genereaza un CONSUM SIMULAT:
    pentru aceasta etapa consideram cazul IDEAL in care casa respecta
    intotdeauna recomandarea - consumul simulat = consumul estimat +/- un
    procent fix (PROCENT_AJUSTARE), in functie de semnal.

  - Modelul e DISCRET IN PASI, un pas = o ora (aceeasi esantionare orara
    folosita peste tot in proiect). Simularea ruleaza pe TOT intervalul
    orar comun intre seria de consum si seria de productie - in practica,
    aproape un an intreg (se afiseaza exact cate zile, la rulare).

ORDINEA DE ACTIVARE intr-un pas - DELIBERAT NU shuffle_do (spre deosebire
de Boltzmann Wealth Model, unde ordinea agentilor nu conteaza): aici
ordinea CONTEAZA, pentru ca fiecare casa reactioneaza la semnalul calculat
DEJA in pasul curent:
    1. AgentRecomandare.step()   - calculeaza si publica semnalul
    2. AgentCasa.step() (fiecare) - reactioneaza la semnalul de mai sus

REGULA SEMAFORULUI:

  Se compara consumul TOTAL estimat al comunitatii cu productia estimata,
  cu o BANDA DE TOLERANTA de +-10% din productie (BANDA_TOLERANTA):

    consum > productie*(1+banda)  DEFICIT - comunitatea consuma mai mult
                                  decat produce centrala, deci trage din
                                  retea                          -> ROSU
                                  (casa trebuie sa consume mai putin)

    consum < productie*(1-banda)  SURPLUS - centrala produce mai mult decat
                                  consuma comunitatea, deci surplusul se
                                  pierde in retea                -> VERDE
                                  (casa poate consuma mai mult, ca sa
                                  foloseasca energia solara disponibila)

    consum intre cele doua praguri  productia acopera consumul, cu o marja
                                  rezonabila                     -> GALBEN
                                  (nu faceti nimic)

  Recomandarile se dau DOAR intr-un interval orar (INTERVAL_ORAR_RECOMANDARI)
  - in afara lui (ex. noaptea), semnalul e automat GALBEN.

  Regula e controlata de variabila ROSU_LA_DEFICIT din ZONA DE CONFIGURARE.
  Valoarea implicita (True) e exact regula ceruta: "daca consumul e mai mare
  decat productia, inseamna ca acea casa trebuie sa consume mai putin".
  Punand-o pe False obtineti conventia inversa (rosu la surplus), fara sa
  schimbati nimic altceva in cod.
=============================================================================
"""

import numpy as np
import pandas as pd
import mesa
import plotly.graph_objects as go

from modelare_pv import (
    sincronizeaza_interval_cu_irise,
    construieste_date_meteo,
    construieste_model_solar,
    orientare_optima,
    construieste_centrala_pv,
    calculeaza_productie_pv,
    pregateste_consum_orar_casa,
)
from config_proiect import CASA_IMPLICITA, PARAMETRI_PANOU, panouri_pentru


# =============================================================================
# ZONA DE CONFIGURARE - valorile pe care le puteti schimba usor la intalnire
# =============================================================================

# Casele din comunitatea simulata ACUM. Pentru aceasta etapa, o singura casa
# (CASA_IMPLICITA din config_proiect.py - ACEEASI casa analizata si in
# modelare_pv.py, ca cifrele celor doua etape sa fie comparabile).
#
# Cand treceti la toate cele 9 case, adaugati aici ID-urile lor si extindeti
# ZONA DE COMENZI de mai jos ca sa construiasca orizontul pentru fiecare -
# restul codului (agentii, modelul, graficele) NU trebuie schimbat, e scris
# generic pentru o lista de case. Numarul de panouri se recalculeaza singur
# (vezi mai jos).
ID_CASE_COMUNITATE = [CASA_IMPLICITA]

# Casa pentru care se construieste orizontul de simulare in aceasta etapa
# (o singura casa in comunitate => prima si singura din lista).
ID_CASA_DORITA = ID_CASE_COMUNITATE[0]

# Centrala folosita in simulare NU e un numar ales separat: e COTA-PARTE a
# caselor din comunitate din centrala dimensionata "de mana" la Etapa 2
# (pv_comunitate.py): 40 de panouri pentru 9 case. Cu o singura casa in
# comunitate => round(40 * 1/9) = 4 panouri. Daca adaugati toate cele 9 case
# in ID_CASE_COMUNITATE, formula da inapoi exact 40 de panouri, adica fix
# centrala comunitatii de la Etapa 2 - fara sa schimbati nimic aici.
# Vezi panouri_pentru() in config_proiect.py.
NUMBER_OF_PANELS = panouri_pentru(len(ID_CASE_COMUNITATE))

BANDA_TOLERANTA = 0.10  # +-10% din productie => semnal galben (nu faceti nimic)

PROCENT_AJUSTARE = 0.10  # cu cat isi schimba casa consumul cand primeste rosu/verde

# (h_inceput, h_sfarsit) - ore din zi (0-23) in care agentul de recomandari
# e "activ". In afara acestui interval, semnalul e automat galben. Implicit,
# un interval de zi/seara (7:00-22:00) - schimbati dupa ce discutati cu
# profesorul ce interval anume vrea.
INTERVAL_ORAR_RECOMANDARI = (7, 22)

# True  = regula CERUTA (si cea uzuala in auto-consum): daca CONSUMUL e mai
#         mare decat productia (deficit, comunitatea trage din retea) =>
#         ROSU, casa trebuie sa consume mai putin; daca productia e mai mare
#         decat consumul (surplus solar nefolosit) => VERDE, casa poate
#         consuma mai mult.
# False = conventia inversa (rosu la surplus, verde la deficit). Comutati
#         aici daca decideti altfel - restul codului NU se schimba.
ROSU_LA_DEFICIT = True


# =============================================================================
# PASUL 1: orizontul de simulare - consum estimat + productie estimata,
# aliniate pe aceeasi axa orara (o singura casa, deocamdata)
# =============================================================================
def _elimina_timestampuri_duplicate(serie, nume):
    """
    Elimina timestamp-urile duplicate din indexul unei serii (pastreaza
    prima aparitie), cu un mesaj clar despre cate au fost gasite - NU le
    stergem in tacere. Cel mai frecvent motiv: ora ambigua de la trecerea
    la ora de iarna (o data pe an, o ora locala "exista" de doua ori).
    """
    duplicate = serie.index.duplicated(keep="first")
    n_dup = int(duplicate.sum())
    if n_dup:
        print(
            f"ATENTIE: {n_dup} timestamp-uri duplicate gasite in seria "
            f"'{nume}' (posibil de la trecerea ora vara/iarna) - pastram "
            f"prima aparitie a fiecaruia, restul se elimina."
        )
        serie = serie[~duplicate]
    return serie


def pregateste_orizont_simulare(id_casa=None, number_of_panels=NUMBER_OF_PANELS):
    """
    Aduna, pe acelasi orizont orar, consumul ESTIMAT al casei (din baza
    IRISE, vezi modelare_pv.py) si productia ESTIMATA a centralei PV
    (model solar BATEM) - aceleasi surse ca la Pasii 1-5a din
    modelare_pv.py, dar aici le aliniem printr-un JOIN pe ora exacta (nu
    doar le suprapunem vizual, ca in genereaza_grafic_productie_vs_consum):
    simularea are nevoie de exact o valoare de consum SI o valoare de
    productie la FIECARE pas, nu doar de un grafic care le arata alaturi.

    Returneaza (orizont, id_casa_ales, pv_system):
      orizont       - DataFrame indexat pe timp (index Ora), cu coloanele
                      "Productie_Estimata_Wh" si
                      f"Consum_Estimat_Wh_{id_casa_ales}" - DOAR pentru
                      orele in care ambele exista (intersectia celor doua
                      intervale - vezi observatia din modelare_pv.py ca
                      cele doua serii pot avea capete usor diferite).
      id_casa_ales  - ID-ul casei alese (util daca ID_CASA_DORITA=None)
      pv_system     - obiectul PVplant (util pentru rezumate - puterea de
                      varf etc.)
    """
    rezultate_pv = sincronizeaza_interval_cu_irise(id_casa=id_casa)

    weather_data = construieste_date_meteo(
        data_inceput=rezultate_pv["data_inceput"],
        data_sfarsit=rezultate_pv["data_sfarsit"],
    )
    solar_model = construieste_model_solar(weather_data)
    best_exposure, best_slope = orientare_optima(
        solar_model, initial_slope_deg=150, initial_exposure_deg=0,
    )
    pv_system = construieste_centrala_pv(
        solar_model, best_exposure, best_slope,
        number_of_panels=number_of_panels, **PARAMETRI_PANOU,
    )
    pv_power = calculeaza_productie_pv(pv_system)

    id_casa_ales = rezultate_pv["id_casa"]
    consum_orar_casa = pregateste_consum_orar_casa(
        rezultate_pv["df_consum"], id_casa_ales,
    )

    productie = pd.Series(
        pv_power, index=pd.DatetimeIndex(solar_model.datetimes),
        name="Productie_Estimata_Wh",
    )
    coloana_consum = f"Consum_Estimat_Wh_{id_casa_ales}"
    consum = (
        consum_orar_casa.set_index("Ora")["Consum_Orar_Casa_Wh"]
        .rename(coloana_consum)
    )

    # Ambele serii pot avea timestamp-uri DUPLICATE (de obicei de la
    # trecerea ora-de-vara/ora-de-iarna - ora ambigua de toamna, sau un
    # artefact al listei de timp din BATEM) - pd.concat NU accepta un
    # index cu duplicate la reindexare, deci le curatam explicit, cu un
    # mesaj clar despre cate au fost gasite/eliminate (pastram prima
    # aparitie a fiecarui timestamp).
    productie = _elimina_timestampuri_duplicate(productie, "Productie_Estimata_Wh")
    consum = _elimina_timestampuri_duplicate(consum, coloana_consum)

    orizont = pd.concat([productie, consum], axis=1, join="inner").sort_index()
    orizont.index.name = "Ora"

    if orizont.empty:
        raise ValueError(
            "Intersectia orelor cu consum si cu productie e goala - "
            "verificati fusul orar / intervalul din sincronizeaza_interval_cu_irise."
        )

    zile = (orizont.index[-1] - orizont.index[0]).total_seconds() / 86400
    print(f"\n=== ORIZONT DE SIMULARE (casa {id_casa_ales}): {len(orizont)} ore "
          f"(~{zile:.0f} zile) ===")
    print(f"De la {orizont.index[0]} pana la {orizont.index[-1]}")

    return orizont, id_casa_ales, pv_system


# =============================================================================
# PASUL 2: agentii
# =============================================================================
class AgentCasa(mesa.Agent):
    """
    Un agent-casa. Are asociat, din baza de date, un consum ESTIMAT orar
    (coloana f"Consum_Estimat_Wh_{id_casa}" din orizontul modelului) - vine
    din masuratorile reale IRISE, agregate pe ora, dar in simulare joaca
    rolul unui profil de consum ASTEPTAT (nu un consum "adevarat", masurat
    live - de-aia se numeste estimat).

    La fiecare pas, casa GENEREAZA un consum SIMULAT, reactionand la
    semnalul PRIMIT DEJA in acelasi pas de la AgentRecomandare. Pentru
    aceasta etapa consideram cazul IDEAL: casa respecta intotdeauna
    recomandarea.
        semnal rosu   -> consum_simulat = consum_estimat * (1 - procent_ajustare)
        semnal verde  -> consum_simulat = consum_estimat * (1 + procent_ajustare)
        semnal galben -> consum_simulat = consum_estimat (neschimbat)
    """

    def __init__(self, model, id_casa, procent_ajustare):
        super().__init__(model)
        self.id_casa = id_casa
        self.procent_ajustare = procent_ajustare
        self.coloana_consum = f"Consum_Estimat_Wh_{id_casa}"

        self.ultimul_consum_estimat = None
        self.ultimul_consum_simulat = None
        self.ultimul_semnal = None
        self.istoric = []  # o intrare/pas: Ora, semnal, consum estimat/simulat

    def step(self):
        pas = self.model.pas_curent
        ora = self.model.orizont.index[pas]
        consum_estimat = float(self.model.orizont[self.coloana_consum].iloc[pas])
        semnal = self.model.semnal_curent

        if semnal == "rosu":
            consum_simulat = consum_estimat * (1 - self.procent_ajustare)
        elif semnal == "verde":
            consum_simulat = consum_estimat * (1 + self.procent_ajustare)
        else:  # galben
            consum_simulat = consum_estimat

        self.ultimul_consum_estimat = consum_estimat
        self.ultimul_consum_simulat = consum_simulat
        self.ultimul_semnal = semnal
        self.istoric.append({
            "Ora": ora,
            "id_casa": self.id_casa,
            "semnal": semnal,
            "Consum_Estimat_Wh": consum_estimat,
            "Consum_Simulat_Wh": consum_simulat,
        })


class AgentRecomandare(mesa.Agent):
    """
    Agentul inteligent al comunitatii - UN SINGUR exemplar. ACTIONEAZA
    PRIMUL in fiecare pas (nu random, ca la shuffle_do din Boltzmann
    Wealth Model - aici ordinea conteaza: casele au nevoie de semnalul
    calculat DEJA in pasul curent).

    Vezi docstring-ul de la inceputul fisierului pentru regula exacta
    (banda de toleranta + ROSU_LA_DEFICIT) si pentru constrangerea de
    interval orar.
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
                # Consumul e PESTE productie + banda => DEFICIT: comunitatea
                # consuma mai mult decat produce centrala si trage din retea.
                # Regula ceruta: in acest caz casa trebuie sa consume mai
                # putin => ROSU.
                semnal = "rosu" if self.rosu_la_deficit else "verde"
            elif consum_total_estimat < prag_jos:
                # Consumul e SUB productie - banda => SURPLUS: centrala
                # produce mai mult decat consuma comunitatea, iar surplusul
                # s-ar pierde in retea. Casa poate consuma mai mult => VERDE.
                semnal = "verde" if self.rosu_la_deficit else "rosu"
            else:
                # Consumul e in banda de +-10% din productie: productia
                # acopera consumul cu o marja rezonabila => nu faceti nimic.
                semnal = "galben"

        self.model.semnal_curent = semnal
        self.model.ultima_productie_estimata = productie
        self.model.ultimul_consum_total_estimat = consum_total_estimat
        self.istoric.append({
            "Ora": ora,
            "Productie_Estimata_Wh": productie,
            "Consum_Total_Estimat_Wh": consum_total_estimat,
            "semnal": semnal,
        })


# =============================================================================
# PASUL 3: modelul (comunitatea) - discret in pasi, un pas = o ora
# =============================================================================
class ModelComunitateEnergetica(mesa.Model):
    """
    Modelul multi-agent al comunitatii. Contine:
      - un AgentRecomandare (agentul inteligent, un singur exemplar)
      - o lista de AgentCasa (momentan un singur element - vezi
        ID_CASE_COMUNITATE in ZONA DE CONFIGURARE)

    Un PAS = o ORA (esantionarea folosita in tot proiectul). model.step()
    activeaza intai AgentRecomandare, apoi fiecare AgentCasa (ordine fixa,
    NU shuffle_do), si colecteaza datele pasului cu un mesa.DataCollector
    (acelasi tipar ca in tutorialul Boltzmann Wealth Model).
    """

    def __init__(
        self,
        orizont,
        id_case,
        banda=BANDA_TOLERANTA,
        interval_orar=INTERVAL_ORAR_RECOMANDARI,
        procent_ajustare=PROCENT_AJUSTARE,
        rosu_la_deficit=ROSU_LA_DEFICIT,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.orizont = orizont
        self.n_pasi = len(orizont)
        self.pas_curent = 0
        self.semnal_curent = None
        self.ultima_productie_estimata = None
        self.ultimul_consum_total_estimat = None

        self.case = [
            AgentCasa(self, id_casa=id_casa, procent_ajustare=procent_ajustare)
            for id_casa in id_case
        ]
        self.recomandare = AgentRecomandare(
            self, banda=banda, interval_orar=interval_orar,
            rosu_la_deficit=rosu_la_deficit,
        )

        reporters = {
            "Semnal": lambda m: m.semnal_curent,
            "Productie_Estimata_Wh": lambda m: m.ultima_productie_estimata,
            "Consum_Total_Estimat_Wh": lambda m: m.ultimul_consum_total_estimat,
            "Consum_Total_Simulat_Wh": lambda m: sum(
                c.ultimul_consum_simulat for c in m.case
            ),
        }
        # cate o coloana estimat/simulat per casa - functioneaza identic
        # pentru 1 casa (acum) sau 9 case (cand extindeti comunitatea)
        for casa in self.case:
            reporters[f"Consum_Estimat_Wh_{casa.id_casa}"] = (
                lambda m, c=casa: c.ultimul_consum_estimat
            )
            reporters[f"Consum_Simulat_Wh_{casa.id_casa}"] = (
                lambda m, c=casa: c.ultimul_consum_simulat
            )
        self.datacollector = mesa.DataCollector(model_reporters=reporters)

    def consum_estimat_casa(self, id_casa, pas):
        return float(self.orizont[f"Consum_Estimat_Wh_{id_casa}"].iloc[pas])

    def consum_total_estimat(self, pas):
        coloane = [c for c in self.case]
        return float(sum(self.consum_estimat_casa(c.id_casa, pas) for c in coloane))

    def step(self):
        """Un pas de model = o ora: intai recomandarea, apoi casele."""
        self.recomandare.step()
        for casa in self.case:
            casa.step()
        self.datacollector.collect(self)
        self.pas_curent += 1

    def ruleaza(self, n_pasi=None):
        n = self.n_pasi if n_pasi is None else min(n_pasi, self.n_pasi)
        for _ in range(n):
            self.step()
        rezultate = self.datacollector.get_model_vars_dataframe()
        # DataCollector indexeaza dupa numarul pasului (0..n-1); punem
        # indexul real de timp, care e exact orizontul de simulare, in
        # aceeasi ordine - ca sa putem trasa graficele direct pe timp.
        rezultate.index = self.orizont.index[:len(rezultate)]
        rezultate.index.name = "Ora"
        return rezultate


# =============================================================================
# PASUL 4: grafice + rezumat
# =============================================================================
CULOARE_SEMNAL = {"rosu": "#d62728", "galben": "#e6a800", "verde": "#2ca02c"}


def genereaza_grafic_simulare_orara(
    rezultate, id_casa, nume_fisier=None,
):
    """
    Suprapune, pe axa orara, productia estimata, consumul estimat (din
    baza de date) si consumul SIMULAT (dupa ce casa a reactionat la
    recomandari) - la fel ca genereaza_grafic_productie_vs_consum din
    modelare_pv.py, plus a treia serie.
    """
    if nume_fisier is None:
        nume_fisier = f"grafic_simulare_multiagent_{id_casa}.html"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rezultate.index, y=rezultate["Productie_Estimata_Wh"],
        name="Productie PV estimata (Wh)", line=dict(color="#e6a800"),
    ))
    fig.add_trace(go.Scatter(
        x=rezultate.index, y=rezultate[f"Consum_Estimat_Wh_{id_casa}"],
        name=f"Consum estimat casa {id_casa} (Wh)", line=dict(color="#1cc847"),
    ))
    fig.add_trace(go.Scatter(
        x=rezultate.index, y=rezultate[f"Consum_Simulat_Wh_{id_casa}"],
        name=f"Consum SIMULAT casa {id_casa} (Wh, dupa recomandari)",
        line=dict(color="#9467bd", dash="dot"),
    ))

    fig.update_layout(
        title=(
            f"Simulare multi-agent (Mesa): productie PV vs. consum estimat "
            f"vs. consum simulat - casa {id_casa}"
        ),
        hovermode="x unified",
        xaxis_title="Data si ora",
        yaxis_title="Energie pe ora (Wh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"\nGrafic simulare orara salvat ca: {nume_fisier}")
    return fig


def genereaza_grafic_lunar_simulare(rezultate, id_casa, nume_fisier=None):
    """
    Totaluri LUNARE (kWh/luna): productie estimata vs. consum estimat vs.
    consum simulat - varianta usor de citit, la fel ca
    genereaza_grafic_lunar din pv_comunitate.py.
    """
    if nume_fisier is None:
        nume_fisier = f"grafic_simulare_multiagent_lunar_{id_casa}.html"

    lunar = rezultate.resample("MS").sum(numeric_only=True) / 1000  # Wh -> kWh
    luni_str = [t.strftime("%Y-%m") for t in lunar.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=luni_str, y=lunar["Productie_Estimata_Wh"],
        name="Productie PV estimata (kWh)", marker_color="#e6a800",
    ))
    fig.add_trace(go.Bar(
        x=luni_str, y=lunar[f"Consum_Estimat_Wh_{id_casa}"],
        name="Consum estimat (kWh)", marker_color="#1f77b4",
    ))
    fig.add_trace(go.Bar(
        x=luni_str, y=lunar[f"Consum_Simulat_Wh_{id_casa}"],
        name="Consum simulat (kWh)", marker_color="#9467bd",
    ))
    fig.update_layout(
        barmode="group",
        title=f"Totaluri lunare: productie vs. consum estimat vs. consum simulat - casa {id_casa}",
        xaxis_title="Luna", yaxis_title="Energie (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"Grafic lunar simulare salvat ca: {nume_fisier}")
    return fig


def genereaza_grafic_distributie_semnale(rezultate, id_casa, nume_fisier=None):
    """Cate ore a dat agentul de recomandari fiecare semnal (rosu/galben/verde)."""
    if nume_fisier is None:
        nume_fisier = f"grafic_distributie_semnale_{id_casa}.html"

    numar = rezultate["Semnal"].value_counts()
    ordine = [s for s in ("rosu", "galben", "verde") if s in numar.index]

    fig = go.Figure(go.Bar(
        x=ordine, y=[int(numar[s]) for s in ordine],
        marker_color=[CULOARE_SEMNAL[s] for s in ordine],
        text=[int(numar[s]) for s in ordine], textposition="outside",
    ))
    fig.update_layout(
        title=f"Numarul de ore cu fiecare semnal - casa {id_casa}",
        xaxis_title="Semnal", yaxis_title="Numar de ore",
    )
    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"Grafic distributie semnale salvat ca: {nume_fisier}")
    return fig


def afiseaza_rezumat_simulare(rezultate, id_casa):
    n_total = len(rezultate)
    numar = rezultate["Semnal"].value_counts()

    productie_kWh = rezultate["Productie_Estimata_Wh"].sum() / 1000
    consum_estimat_kWh = rezultate[f"Consum_Estimat_Wh_{id_casa}"].sum() / 1000
    consum_simulat_kWh = rezultate[f"Consum_Simulat_Wh_{id_casa}"].sum() / 1000

    print(f"\n=== SIMULARE MULTI-AGENT: rezumat (casa {id_casa}) ===")
    print(f"Pasi simulati (ore):                 {n_total}  (~{n_total / 24:.0f} zile)")
    for semnal in ("rosu", "galben", "verde"):
        n = int(numar.get(semnal, 0))
        print(f"  Semnal {semnal:<6s}: {n:6d} ore  ({100 * n / n_total:5.1f}%)")

    print(f"\nProductie PV estimata (an):          {productie_kWh:9.0f} kWh")
    print(f"Consum ESTIMAT (an, din baza IRISE):  {consum_estimat_kWh:9.0f} kWh")
    print(f"Consum SIMULAT (an, dupa recomandari):{consum_simulat_kWh:9.0f} kWh")
    diferenta_pct = 100 * (consum_simulat_kWh - consum_estimat_kWh) / consum_estimat_kWh
    print(f"Diferenta consum simulat vs. estimat:{diferenta_pct:+9.2f}%")
    print(
        f"Grad de acoperire (productie / consum): "
        f"estimat {100 * productie_kWh / consum_estimat_kWh:.0f}%  ->  "
        f"simulat {100 * productie_kWh / consum_simulat_kWh:.0f}%"
    )


# =============================================================================
# ZONA DE COMENZI
# =============================================================================
if __name__ == "__main__":
    # --- Pasul 1: orizontul de simulare (consum estimat + productie estimata)
    orizont, id_casa_ales, pv_system = pregateste_orizont_simulare(
        id_casa=ID_CASA_DORITA, number_of_panels=NUMBER_OF_PANELS,
    )

    # Comunitatea, deocamdata: o singura casa (ID_CASE_COMUNITATE din ZONA DE
    # CONFIGURARE). Cand extindeti la toate cele 9 case, adaugati-le acolo si
    # construiti orizontul pentru fiecare (acelasi pregateste_orizont_simulare,
    # cate un apel per id_casa), le uniti intr-un singur DataFrame
    # (join="inner" pe toate, plus O SINGURA coloana Productie_Estimata_Wh -
    # aceeasi centrala pentru toata comunitatea). Numarul de panouri se
    # recalculeaza singur din lista de case (vezi NUMBER_OF_PANELS).

    model = ModelComunitateEnergetica(
        orizont=orizont,
        id_case=ID_CASE_COMUNITATE,
        banda=BANDA_TOLERANTA,
        interval_orar=INTERVAL_ORAR_RECOMANDARI,
        procent_ajustare=PROCENT_AJUSTARE,
        rosu_la_deficit=ROSU_LA_DEFICIT,
    )

    print(f"\n=== SIMULARE MULTI-AGENT: {model.n_pasi} pasi (ore) ===")
    print(
        f"Centrala: {pv_system.number_of_panels} panouri, "
        f"{pv_system.peak_power_kW:.2f} kWp "
        f"(cota-parte a {len(ID_CASE_COMUNITATE)} case din centrala comunitatii)"
    )
    print(
        "Regula semafor: ROSU = consumul depaseste productia (deficit) => "
        "casa scade consumul; VERDE = productia depaseste consumul (surplus) "
        "=> casa creste consumul; GALBEN = in banda de "
        f"+-{100 * BANDA_TOLERANTA:.0f}%."
        if ROSU_LA_DEFICIT else
        "Regula semafor: conventia INVERSA (ROSU la surplus, VERDE la deficit)."
    )
    rezultate = model.ruleaza()

    # --- Pasul 4: grafice + rezumat ---------------------------------------
    genereaza_grafic_simulare_orara(rezultate, id_casa_ales)
    genereaza_grafic_lunar_simulare(rezultate, id_casa_ales)
    genereaza_grafic_distributie_semnale(rezultate, id_casa_ales)
    afiseaza_rezumat_simulare(rezultate, id_casa_ales)

    nume_csv = f"rezultate_simulare_multiagent_{id_casa_ales}.csv"
    rezultate.to_csv(nume_csv)
    print(f"\nRezultatele complete (pas cu pas) salvate ca: {nume_csv}")

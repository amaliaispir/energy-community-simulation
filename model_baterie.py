"""
=============================================================================
MODELARE BATERIE (ESS) - Energy Reservoir Model + indicatori SC / SS
=============================================================================

Etapa noua a proiectului, construita peste modelarea PV deja facuta in
modelare_pv.py. Implementeaza modelul bateriei (sistemul de stocare - ESS)
si logica de control al fluxului de energie descrise in articolul
profesorului:

    Simoiu, M.S.; Fagarasan, I.; Ploix, S.; Calofir, V. "Sizing and
    Management of an Energy System for a Metropolitan Station with Storage
    and Related District Energy Community." Energies 2021, 14, 5997.
    https://doi.org/10.3390/en14185997

REFERINTE EXACTE LA ARTICOL (verificate direct in PDF):
  - Sectiunea 4.4 (pag. 9-10), Ecuatiile (8)-(10): modelul SOC (Energy
    Reservoir Model) si conventia de semn p_ESS (Ecuatia 4, pag. 7:
    p_Load = p_G + p_ESS + p_PV; conventie confirmata si la Ecuatia 10:
    "if the battery is charging, then p_ESS < 0, and if discharging,
    p_ESS > 0").
  - Sectiunea 3 (pag. 5-6), Ecuatiile (1)-(2): autoconsum (SC) si
    autosuficienta (SS), forma discreta, INCLUZAND flexibilitatea ESS
    (min(p_PV, p_Load - p_ESS)).

CE FACE ACEST MODUL (o singura casa - Casa CASA_IMPLICITA din
config_proiect.py, adica 2000901):

  Acesta e pasul de TESTARE/VALIDARE cerut de profesor inainte de a avea
  incredere in modelul bateriei la scara de comunitate: "o sa consideram ca
  avem o SINGURA baterie pentru comunitate, de capacitate mare... ca sa
  testezi, poti sa folosesti bateria + simularea ta cu o casa". Deci aici
  folosim EXACT bateria mare a comunitatii (Q_NOM_ESS_KWH, implicit 50 kWh -
  ACEEASI valoare ca Q_NOM_ESS_COMUNITATE_KWH din simulare_comunitate_
  baterie.py), dar aplicata doar pe simularea unei singure case, ca sa
  putem verifica vizual (cele doua grafice de mai jos) ca modelul
  functioneaza corect, inainte de a rula pe toate cele 9 case.

  PASUL A: reface pipeline-ul din modelare_pv.py (date IRISE -> interval
    valid -> meteo BATEM -> model solar -> centrala PV) pentru casa aleasa,
    cu centrala ei PV (PANOURI_O_CASA = 6 panouri Trina Solar Vertex S
    425W, ~2,54 kWp - cota-parte a unei case din centrala comunitara de 50
    de panouri, ACEEASI valoare deja folosita de modelare_pv.py; am
    corectat aici o valoare separata de 7 panouri/~3 kWp folosita initial,
    ca sa nu existe doua marimi diferite pentru "centrala PV a unei case"
    in proiect) - aceasta e "simularea ta cu o casa" mentionata de profesor.

  PASUL B: aliniaza seria de productie PV (din modelul solar BATEM) cu
    seria de consum orar curatat al casei (din pipeline-ul IRISE), pe
    intersectia orelor comune celor doua surse.

  PASUL C: simuleaza BATERIA MARE (Energy Reservoir Model, Ecuatiile 8-10)
    cu logica de control ceruta (surplus solar incarca bateria pana la
    SOC_max, deficitul e acoperit din baterie pana la SOC_min, restul
    ramas necoperit vine din retea) - aplicata pe datele acestei case.

  PASUL D: calculeaza indicatorii de autoconsum (SC) si autosuficienta
    (SS) - Ecuatiile (1)-(2), forma discreta - atat CU baterie (folosind
    p_ESS rezultat din simulare), cat si FARA baterie (p_ESS = 0 peste
    tot, adica autoconsumul "brut", direct PV<->casa).

  PASUL E: genereaza cele doua grafice Plotly cerute de profesor:
      - grafic_balanta_energetica_2000901.html
      - grafic_evolutie_soc_2000901.html

=============================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modelare_pv import (
    sincronizeaza_interval_cu_irise,
    construieste_date_meteo,
    construieste_model_solar,
    orientare_optima,
    construieste_centrala_pv,
    calculeaza_productie_pv,
    pregateste_consum_orar_casa,
)
from config_proiect import CASA_IMPLICITA, PARAMETRI_PANOU, NUME_PANOU, PANOURI_O_CASA


# =========================================================================
# ZONA DE CONFIGURARE
# =========================================================================

# --- Centrala PV a casei folosite pentru testul bateriei ----------------
# Preluata DIRECT din config_proiect.py (PANOURI_O_CASA) - cota-parte a
# unei case din centrala comunitara de 50 de panouri (Etapa 2), ACEEASI
# valoare pe care o foloseste deja modelare_pv.py pentru analiza unei
# singure case. Initial folosisem aici o centrala separata, dimensionata
# direct la "3 kWp" (7 panouri) - am corectat-o, ca sa nu existe doua
# cifre diferite pentru "centrala PV a unei case" in acelasi proiect.
NUMAR_PANOURI_BATERIE = PANOURI_O_CASA  # 6 panouri => ~2,54 kWp

# --- Parametrii bateriei (Ecuatiile 8-10 din articol, pag. 9-10) --------
# ATENTIE: aceasta e ACEEASI baterie "mare" a comunitatii (nu o baterie mica
# redimensionata pentru o casa) - vezi Q_NOM_ESS_COMUNITATE_KWH din
# simulare_comunitate_baterie.py, care foloseste aceeasi valoare. Aici o
# testam doar pe o singura casa, per cerinta explicita a profesorului.
Q_NOM_ESS_KWH = 50.0    # capacitate nominala baterie (varianta 2: 60.0 kWh)
SOC_MIN = 0.20          # 20% - limita minima de protectie
SOC_MAX = 0.90          # 90% - limita maxima de protectie
SOC_INITIAL = 0.20      # SOC la ora 0 (varianta alternativa ceruta: 0.50)
DELTA_SOC_LIM = 0.35    # limita de putere per pas orar, ca fractie din Q_nom
DT_H = 1.0              # pas orar (Delta t = 1h, ca in articol)


# =========================================================================
# PASUL B: aliniem seria de productie PV cu seria de consum orar al casei
# =========================================================================
def alinieaza_serii(pv_datetimes, pv_power_Wh, consum_orar_casa):
    """
    Productia PV vine din modelul solar BATEM (indexata dupa
    solar_model.datetimes), iar consumul orar vine din pipeline-ul IRISE
    (indexat dupa coloana "Ora" din consum_orar_casa, fus orar
    Europe/Paris - vezi converteste_timpul/calculeaza_consum_orar din
    MAIN_corectat.py). Pentru simularea bateriei avem nevoie de o
    diferenta P_PV - P_Load ORA CU ORA, deci cele doua serii trebuie
    aliniate pe exact aceleasi momente de timp - nu e suficient sa fie
    doar suprapuse vizual pe un grafic (cum se intampla la Pasul 5b din
    modelare_pv.py).

    Strategia (in ordine, prima care da un rezultat nevid castiga):
      1. Aliniere directa pe index (daca ambele serii sunt deja in acelasi
         fus orar / ambele naive).
      2. Daca cele doua serii au fus orar diferit (una are tz, cealalta
         nu, sau au tz-uri diferite), le aducem pe amandoua la UTC inainte
         de aliniere.
      3. Daca tot nu iese nicio ora comuna (ex. BATEM intoarce timp local
         "naiv" care nu coincide numeric cu UTC), cadem pe o aliniere
         POZITIONALA (trunchiem ambele serii la lungimea minima comuna),
         cu un avertisment explicit in consola - util ca sa nu blocheze
         testarea, dar de VERIFICAT manual daca se ajunge aici.

    Returneaza un DataFrame cu coloanele: Ora, P_PV_Wh, P_Load_Wh.
    """
    serie_pv = pd.Series(
        np.asarray(pv_power_Wh, dtype=float),
        index=pd.DatetimeIndex(pd.Series(pv_datetimes)),
    )
    serie_load = consum_orar_casa.set_index("Ora")["Consum_Orar_Casa_Wh"].astype(float)

    # Eliminam eventualele ore DUPLICATE din index (de ex. ora "dubla" de la
    # trecerea pe ora de iarna, cand ceasul da inapoi si aceeasi ora locala
    # apare de doua ori) - .reindex() de mai jos cere un index unic, iar
    # pastrarea primei aparitii e suficient de precisa la rezolutie orara.
    serie_pv = serie_pv[~serie_pv.index.duplicated(keep="first")]
    serie_load = serie_load[~serie_load.index.duplicated(keep="first")]

    def _comun(a, b):
        return a.index.intersection(b.index).sort_values()

    orar_comun = _comun(serie_pv, serie_load)

    if len(orar_comun) == 0:
        # Incercare 2: aducem ambele la UTC
        pv2, load2 = serie_pv.copy(), serie_load.copy()
        if pv2.index.tz is None:
            pv2.index = pv2.index.tz_localize("UTC")
        else:
            pv2.index = pv2.index.tz_convert("UTC")
        if load2.index.tz is None:
            load2.index = load2.index.tz_localize("UTC")
        else:
            load2.index = load2.index.tz_convert("UTC")
        orar_comun = _comun(pv2, load2)
        if len(orar_comun) > 0:
            serie_pv, serie_load = pv2, load2

    if len(orar_comun) > 0:
        df = pd.DataFrame({
            "Ora": orar_comun,
            "P_PV_Wh": serie_pv.reindex(orar_comun).values,
            "P_Load_Wh": serie_load.reindex(orar_comun).values,
        }).reset_index(drop=True)
        print(f"Aliniere productie/consum: {len(df)} ore comune "
              f"({df['Ora'].iloc[0]} -> {df['Ora'].iloc[-1]})")
        return df

    # Incercare 3: aliniere pozitionala (ultima solutie, cu avertisment)
    n = min(len(serie_pv), len(serie_load))
    print(
        "\n!!! ATENTIE: nicio ora comuna intre seria de productie PV si cea "
        "de consum (nici direct, nici dupa conversia la UTC). Se foloseste "
        f"o aliniere POZITIONALA (primele {n} ore ale fiecarei serii, in "
        "ordine) - verificati manual fusul orar intors de BATEM "
        "(solar_model.datetimes) fata de coloana Ora din consum_orar_casa "
        "daca rezultatele par nerealiste."
    )
    df = pd.DataFrame({
        "Ora": pd.DatetimeIndex(serie_load.index[:n]),
        "P_PV_Wh": serie_pv.values[:n],
        "P_Load_Wh": serie_load.values[:n],
    })
    return df


# =========================================================================
# PASUL C: simularea bateriei, ora cu ora (Ecuatiile 8-10)
# =========================================================================
def simuleaza_baterie(
    p_pv_kw,
    p_load_kw,
    Q_nom_kWh=Q_NOM_ESS_KWH,
    soc_min=SOC_MIN,
    soc_max=SOC_MAX,
    soc_initial=SOC_INITIAL,
    delta_soc_lim=DELTA_SOC_LIM,
    dt=DT_H,
):
    """
    Simuleaza bateria (Energy Reservoir Model) ora cu ora, pentru seriile
    p_pv_kw / p_load_kw (aceeasi lungime, in kW - numeric egal cu kWh
    pentru un pas orar).

    CONVENTIA DE SEMN (Ecuatia 4 & 10 din articol):
      p_ESS > 0  => bateria se DESCARCA (sursa de putere pentru sistem)
      p_ESS < 0  => bateria se INCARCA (sarcina/consumator pentru sistem)

    LOGICA DE CONTROL, pentru fiecare ora i:
      Delta_P = P_PV,i - P_Load,i
      - Delta_P > 0 (surplus solar):
            p_ESS,i = -min(Delta_P, putere_max_incarcare)
            unde putere_max_incarcare respecta AMBELE limite:
              (a) limita de putere per pas orar: Delta_SOC_lim * Q_nom / dt
              (b) capacitatea ramasa pana la SOC_max: (SOC_max - SOC_i) * Q_nom / dt
            Daca surplusul depaseste ce poate prelua bateria, diferenta e
            considerata "pierduta"/injectata (nu mai e stocata).
      - Delta_P < 0 (deficit):
            p_ESS,i = min(|Delta_P|, putere_max_descarcare)
            unde putere_max_descarcare respecta AMBELE limite:
              (a) limita de putere per pas orar (ca mai sus)
              (b) energia ramasa pana la SOC_min: (SOC_i - SOC_min) * Q_nom / dt
            Ce ramane neacoperit de baterie vine din retea:
              p_Grid,i = P_Load,i - P_PV,i - p_ESS,i  (>= 0)

    DINAMICA SOC (Ecuatia 10, discretizata la Delta t = 1h):
      SOC_{i+1} = SOC_i - p_ESS,i * Delta t / Q_nom_ESS

    Returneaza un DataFrame cu coloanele:
      P_PV_kW, P_Load_kW, P_ESS_kW, P_Grid_kW, SOC (la INCEPUTUL orei i,
      adica starea folosita pentru decizia din ora i), SOC_next (la
      sfarsitul orei i / inceputul orei i+1).
    """
    p_pv = np.asarray(p_pv_kw, dtype=float)
    p_load = np.asarray(p_load_kw, dtype=float)
    m = len(p_pv)
    if len(p_load) != m:
        raise ValueError("P_PV si P_Load trebuie sa aiba aceeasi lungime.")

    p_ess = np.zeros(m)
    p_grid = np.zeros(m)
    soc = np.zeros(m + 1)
    soc[0] = soc_initial

    putere_limita_kw = delta_soc_lim * Q_nom_kWh / dt  # limita FIXA de putere (kW)

    for i in range(m):
        delta_p = p_pv[i] - p_load[i]
        soc_i = soc[i]

        if delta_p > 0:
            # --- surplus solar -> incarcam bateria (p_ESS < 0) ----------
            putere_max_incarcare_soc = max(0.0, (soc_max - soc_i) * Q_nom_kWh / dt)
            putere_max_incarcare = min(putere_limita_kw, putere_max_incarcare_soc)
            p_ess[i] = -min(delta_p, putere_max_incarcare)
        elif delta_p < 0:
            # --- deficit -> descarcam bateria (p_ESS > 0) ----------------
            deficit = -delta_p
            putere_max_descarcare_soc = max(0.0, (soc_i - soc_min) * Q_nom_kWh / dt)
            putere_max_descarcare = min(putere_limita_kw, putere_max_descarcare_soc)
            p_ess[i] = min(deficit, putere_max_descarcare)
        else:
            p_ess[i] = 0.0

        # Balanta de putere (Ecuatia 4): p_Load = p_G + p_ESS + p_PV
        p_grid[i] = max(0.0, p_load[i] - p_pv[i] - p_ess[i])

        # Dinamica SOC (Ecuatia 10)
        soc_urmator = soc_i - p_ess[i] * dt / Q_nom_kWh
        # Clip de siguranta (protectie la erori de rotunjire in virgula
        # mobila - constrangerile de mai sus deja garantau matematic ca
        # soc_urmator ramane in [soc_min, soc_max]).
        soc[i + 1] = min(max(soc_urmator, soc_min), soc_max)

    rezultat = pd.DataFrame({
        "P_PV_kW": p_pv,
        "P_Load_kW": p_load,
        "P_ESS_kW": p_ess,
        "P_Grid_kW": p_grid,
        "SOC": soc[:-1],
        "SOC_next": soc[1:],
    })

    # Verificare (parte din pasul de validare a simularii): SOC trebuie sa
    # ramana mereu in intervalul [SOC_min, SOC_max].
    soc_valori = rezultat["SOC_next"]
    if soc_valori.min() < soc_min - 1e-9 or soc_valori.max() > soc_max + 1e-9:
        print(
            f"!!! ATENTIE: SOC a iesit din limitele [{soc_min}, {soc_max}] "
            f"(min={soc_valori.min():.4f}, max={soc_valori.max():.4f}) - "
            "verificati logica de control."
        )
    else:
        print(
            f"Verificare SOC: ramane in tot timpul in [{soc_min:.2f}, "
            f"{soc_max:.2f}] (min simulat={soc_valori.min():.4f}, "
            f"max simulat={soc_valori.max():.4f}) - OK."
        )

    return rezultat


# =========================================================================
# PASUL D: indicatorii de performanta - Autoconsum (SC) si Autonomie (SS)
# =========================================================================
def calculeaza_SC_SS(p_pv_kw, p_load_kw, p_ess_kw, dt=DT_H):
    """
    Ecuatiile (1)-(2) din articol (forma discreta), INCLUZAND flexibilitatea
    oferita de baterie prin p_ESS (daca p_ess_kw e un vector de zerouri,
    formulele se reduc la autoconsumul/autosuficienta "brute", fara
    baterie):

        SC = sum(min(P_PV,i, P_Load,i - p_ESS,i) * dt) / sum(P_PV,i * dt)
        SS = sum(min(P_PV,i, P_Load,i - p_ESS,i) * dt) / sum(P_Load,i * dt)

    Returneaza (SC, SS) ca fractii in [0, 1] (nu procente).
    """
    p_pv = np.asarray(p_pv_kw, dtype=float)
    p_load = np.asarray(p_load_kw, dtype=float)
    p_ess = np.asarray(p_ess_kw, dtype=float)

    energie_utila = np.minimum(p_pv, p_load - p_ess) * dt
    energie_pv_totala = np.sum(p_pv) * dt
    energie_load_totala = np.sum(p_load) * dt

    SC = float(np.sum(energie_utila) / energie_pv_totala) if energie_pv_totala > 0 else float("nan")
    SS = float(np.sum(energie_utila) / energie_load_totala) if energie_load_totala > 0 else float("nan")
    return SC, SS


# =========================================================================
# PASUL E1: grafic Plotly - balanta energetica (Consum, PV, Baterie, Retea)
# =========================================================================
def genereaza_grafic_balanta_energetica(rezultat_baterie, id_casa, nume_fisier=None, eticheta=None):
    """
    grafic_balanta_energetica_<id_casa>.html - pe aceeasi axa de timp:
      - Consumul casei (P_Load)
      - Productia PV (P_PV)
      - Puterea bateriei (P_ESS) - conventie: >0 descarca, <0 incarca
      - Puterea din retea (P_Grid)

    `id_casa` controleaza numele implicit al fisierului (ex. "comunitate" ->
    grafic_balanta_energetica_comunitate.html); `eticheta` controleaza doar
    textul din titlul graficului (implicit "Casa <id_casa>", dar poate fi
    orice text, ex. "Comunitate (9 case)").
    """
    if nume_fisier is None:
        nume_fisier = f"grafic_balanta_energetica_{id_casa}.html"
    if eticheta is None:
        eticheta = f"Casa {id_casa}"

    df = rezultat_baterie
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Ora"], y=df["P_Load_kW"], name="Consum casa (P_Load)",
        line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=df["Ora"], y=df["P_PV_kW"], name="Productie PV (P_PV)",
        line=dict(color="#e6a800"),
    ))
    fig.add_trace(go.Scatter(
        x=df["Ora"], y=df["P_ESS_kW"], name="Putere baterie (P_ESS, +desc/-inc)",
        line=dict(color="#2ca02c"),
    ))
    fig.add_trace(go.Scatter(
        x=df["Ora"], y=df["P_Grid_kW"], name="Putere din retea (P_Grid)",
        line=dict(color="#d62728", dash="dot"),
    ))
    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))

    fig.update_layout(
        title=f"Balanta energetica cu baterie - {eticheta}",
        hovermode="x unified",
        xaxis_title="Data si ora",
        yaxis_title="Putere (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"\nGrafic balanta energetica salvat ca: {nume_fisier}")
    return fig


# =========================================================================
# PASUL E2: grafic Plotly - evolutia SOC
# =========================================================================
def genereaza_grafic_evolutie_soc(rezultat_baterie, id_casa, soc_min=SOC_MIN, soc_max=SOC_MAX, nume_fisier=None, eticheta=None):
    """
    grafic_evolutie_soc_<id_casa>.html - evolutia in timp a starii de
    incarcare (SOC, in %), cu liniile de referinta SOC_min si SOC_max.

    Vezi genereaza_grafic_balanta_energetica pentru semnificatia lui
    `id_casa` (nume fisier) vs. `eticheta` (text in titlu).
    """
    if nume_fisier is None:
        nume_fisier = f"grafic_evolutie_soc_{id_casa}.html"
    if eticheta is None:
        eticheta = f"Casa {id_casa}"

    df = rezultat_baterie
    soc_procent = df["SOC_next"] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Ora"], y=soc_procent, name="SOC (%)",
        line=dict(color="#9467bd"), fill="tozeroy",
        fillcolor="rgba(148, 103, 189, 0.15)",
    ))
    fig.add_hline(y=soc_max * 100, line=dict(color="#d62728", dash="dash"),
                  annotation_text=f"SOC_max = {soc_max * 100:.0f}%", annotation_position="top left")
    fig.add_hline(y=soc_min * 100, line=dict(color="#d62728", dash="dash"),
                  annotation_text=f"SOC_min = {soc_min * 100:.0f}%", annotation_position="bottom left")

    fig.update_layout(
        title=f"Evolutia SOC (stare de incarcare baterie) - {eticheta}",
        hovermode="x unified",
        xaxis_title="Data si ora",
        yaxis_title="SOC (%)",
        yaxis=dict(range=[0, 100]),
    )

    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"Grafic evolutie SOC salvat ca: {nume_fisier}")
    return fig


# =========================================================================
# PASUL A+B+C+D: pipeline complet, pentru o singura casa
# =========================================================================
def ruleaza_simulare_baterie(
    id_casa=CASA_IMPLICITA,
    Q_nom_kWh=Q_NOM_ESS_KWH,
    number_of_panels=NUMAR_PANOURI_BATERIE,
    soc_min=SOC_MIN,
    soc_max=SOC_MAX,
    soc_initial=SOC_INITIAL,
    delta_soc_lim=DELTA_SOC_LIM,
    dt=DT_H,
):
    """
    Ruleaza tot pipeline-ul: date IRISE -> interval valid -> meteo BATEM ->
    model solar -> centrala PV (PANOURI_O_CASA, ~2,54 kWp) -> consum orar
    curatat -> aliniere -> simulare baterie -> indicatori SC/SS (cu si
    fara baterie).

    Returneaza un dictionar cu tot ce e nevoie mai departe pentru grafice
    si afisarea rezumatului:
      {
        "id_casa", "pv_system", "rezultat_baterie",
        "SC_cu_baterie", "SS_cu_baterie",
        "SC_fara_baterie", "SS_fara_baterie",
        "Q_nom_kWh", "soc_min", "soc_max",
      }
    """
    print(f"\n=== MODELARE BATERIE: pipeline complet pentru casa {id_casa} ===")

    # --- Pasul 1 (din modelare_pv.py): sincronizam intervalul cu IRISE ---
    rezultate_pv = sincronizeaza_interval_cu_irise(id_casa=id_casa)

    # --- Pasul 2: date meteo (BATEM + Grenoble.json) ----------------------
    weather_data = construieste_date_meteo(
        data_inceput=rezultate_pv["data_inceput"],
        data_sfarsit=rezultate_pv["data_sfarsit"],
    )

    # --- Pasul 3: modelul solar + orientarea optima ------------------------
    solar_model = construieste_model_solar(weather_data)
    best_exposure, best_slope = orientare_optima(
        solar_model, initial_slope_deg=150, initial_exposure_deg=0,
    )

    # --- Pasul 4: centrala PV a casei (PANOURI_O_CASA, ~2,54 kWp) ---------
    pv_system = construieste_centrala_pv(
        solar_model, best_exposure, best_slope,
        number_of_panels=number_of_panels, **PARAMETRI_PANOU,
    )
    pv_power_Wh = calculeaza_productie_pv(pv_system)
    print(
        f"Centrala PV de test (baterie): {pv_system.number_of_panels} panouri "
        f"{NUME_PANOU}, putere de varf {pv_system.peak_power_kW:.2f} kWp "
        f"(PANOURI_O_CASA din config_proiect.py, cota-parte a unei case "
        f"din centrala comunitara)"
    )

    # --- Pasul 5a: consumul orar curatat al casei --------------------------
    consum_orar_casa = pregateste_consum_orar_casa(
        rezultate_pv["df_consum"], rezultate_pv["id_casa"],
    )

    # --- Pasul B: aliniere productie <-> consum, pe orele comune ----------
    df_aliniat = alinieaza_serii(solar_model.datetimes, pv_power_Wh, consum_orar_casa)

    p_pv_kw = df_aliniat["P_PV_Wh"].to_numpy() / 1000.0
    p_load_kw = df_aliniat["P_Load_Wh"].to_numpy() / 1000.0

    # --- Pasul C: simularea bateriei ---------------------------------------
    rezultat_baterie = simuleaza_baterie(
        p_pv_kw, p_load_kw,
        Q_nom_kWh=Q_nom_kWh, soc_min=soc_min, soc_max=soc_max,
        soc_initial=soc_initial, delta_soc_lim=delta_soc_lim, dt=dt,
    )
    rezultat_baterie.insert(0, "Ora", df_aliniat["Ora"].to_numpy())

    # --- Pasul D: indicatorii SC / SS, CU si FARA baterie ------------------
    SC_cu_baterie, SS_cu_baterie = calculeaza_SC_SS(
        p_pv_kw, p_load_kw, rezultat_baterie["P_ESS_kW"].to_numpy(), dt=dt,
    )
    SC_fara_baterie, SS_fara_baterie = calculeaza_SC_SS(
        p_pv_kw, p_load_kw, np.zeros_like(p_pv_kw), dt=dt,
    )

    return {
        "id_casa": id_casa,
        "pv_system": pv_system,
        "rezultat_baterie": rezultat_baterie,
        "p_pv_kw": p_pv_kw,
        "p_load_kw": p_load_kw,
        "SC_cu_baterie": SC_cu_baterie,
        "SS_cu_baterie": SS_cu_baterie,
        "SC_fara_baterie": SC_fara_baterie,
        "SS_fara_baterie": SS_fara_baterie,
        "Q_nom_kWh": Q_nom_kWh,
        "soc_min": soc_min,
        "soc_max": soc_max,
    }


# =========================================================================
# Afisarea rezumatului in consola
# =========================================================================
def afiseaza_rezumat_baterie(rezultate):
    df = rezultate["rezultat_baterie"]
    id_casa = rezultate["id_casa"]
    Q_nom = rezultate["Q_nom_kWh"]

    energie_incarcata_kWh = -df.loc[df["P_ESS_kW"] < 0, "P_ESS_kW"].sum() * DT_H
    energie_descarcata_kWh = df.loc[df["P_ESS_kW"] > 0, "P_ESS_kW"].sum() * DT_H
    energie_retea_kWh = df["P_Grid_kW"].sum() * DT_H
    energie_pv_kWh = df["P_PV_kW"].sum() * DT_H
    energie_load_kWh = df["P_Load_kW"].sum() * DT_H

    print(f"\n=== REZUMAT SIMULARE BATERIE - Casa {id_casa} "
          f"(Q_nom = {Q_nom:.1f} kWh) ===")
    print(f"Interval simulat:              {len(df)} ore "
          f"({len(df) / 24:.1f} zile)")
    print(f"Productie PV totala:           {energie_pv_kWh:.1f} kWh")
    print(f"Consum casa total:             {energie_load_kWh:.1f} kWh")
    print(f"Energie incarcata in baterie:  {energie_incarcata_kWh:.1f} kWh")
    print(f"Energie descarcata din baterie:{energie_descarcata_kWh:.1f} kWh")
    print(f"Energie extrasa din retea:     {energie_retea_kWh:.1f} kWh")
    print(f"SOC minim / maxim atins:       "
          f"{df['SOC_next'].min() * 100:.1f}% / {df['SOC_next'].max() * 100:.1f}%")

    print(f"\n--- Indicatori de performanta (Ecuatiile 1-2 din articol) ---")
    print(f"{'':22s}{'FARA baterie':>15s}{'CU baterie':>15s}{'Diferenta':>15s}")
    print(f"{'Self-Consumption (SC)':22s}"
          f"{100 * rezultate['SC_fara_baterie']:>14.1f}%"
          f"{100 * rezultate['SC_cu_baterie']:>14.1f}%"
          f"{100 * (rezultate['SC_cu_baterie'] - rezultate['SC_fara_baterie']):>+14.1f}pp")
    print(f"{'Self-Sufficiency (SS)':22s}"
          f"{100 * rezultate['SS_fara_baterie']:>14.1f}%"
          f"{100 * rezultate['SS_cu_baterie']:>14.1f}%"
          f"{100 * (rezultate['SS_cu_baterie'] - rezultate['SS_fara_baterie']):>+14.1f}pp")


# =========================================================================
# Comparatie EFICIENTA cu o alta capacitate de baterie, pe ACEEASI casa
# (fara sa re-ceara IRISE + BATEM - doar re-ruleaza simularea bateriei,
# care e rapida, refolosind p_pv_kw/p_load_kw deja calculate)
# =========================================================================
def recalculeaza_cu_alta_baterie(rezultate_baza, Q_nom_kWh):
    p_pv_kw = rezultate_baza["p_pv_kw"]
    p_load_kw = rezultate_baza["p_load_kw"]

    rezultat_baterie = simuleaza_baterie(
        p_pv_kw, p_load_kw,
        Q_nom_kWh=Q_nom_kWh, soc_min=rezultate_baza["soc_min"], soc_max=rezultate_baza["soc_max"],
        soc_initial=SOC_INITIAL, delta_soc_lim=DELTA_SOC_LIM, dt=DT_H,
    )
    rezultat_baterie.insert(0, "Ora", rezultate_baza["rezultat_baterie"]["Ora"].to_numpy())

    SC_cu, SS_cu = calculeaza_SC_SS(p_pv_kw, p_load_kw, rezultat_baterie["P_ESS_kW"].to_numpy())
    SC_fara, SS_fara = calculeaza_SC_SS(p_pv_kw, p_load_kw, np.zeros_like(p_pv_kw))
    return {
        "id_casa": rezultate_baza["id_casa"],
        "pv_system": rezultate_baza["pv_system"],
        "rezultat_baterie": rezultat_baterie,
        "p_pv_kw": p_pv_kw,
        "p_load_kw": p_load_kw,
        "SC_cu_baterie": SC_cu, "SS_cu_baterie": SS_cu,
        "SC_fara_baterie": SC_fara, "SS_fara_baterie": SS_fara,
        "Q_nom_kWh": Q_nom_kWh,
        "soc_min": rezultate_baza["soc_min"], "soc_max": rezultate_baza["soc_max"],
    }


# =========================================================================
# ZONA DE COMENZI
# =========================================================================
if __name__ == "__main__":
    # Testul cerut de profesor: bateria MARE a comunitatii (Q_NOM_ESS_KWH =
    # 50 kWh, aceeasi valoare ca la comunitate), aplicata pe simularea unei
    # singure case (2000901) - doar ca sa verificam vizual, pe cele doua
    # grafice de mai jos, ca modelul bateriei functioneaza corect, inainte
    # sa rulam pe toate cele 9 case (simulare_comunitate_baterie.py).
    ID_CASA_DORITA = CASA_IMPLICITA  # 2000901, din config_proiect.py

    rezultate = ruleaza_simulare_baterie(
        id_casa=ID_CASA_DORITA,
        Q_nom_kWh=Q_NOM_ESS_KWH,   # 50.0 kWh - bateria MARE a comunitatii
    )
    afiseaza_rezumat_baterie(rezultate)

    genereaza_grafic_balanta_energetica(
        rezultate["rezultat_baterie"], rezultate["id_casa"],
    )
    genereaza_grafic_evolutie_soc(
        rezultate["rezultat_baterie"], rezultate["id_casa"],
        soc_min=rezultate["soc_min"], soc_max=rezultate["soc_max"],
    )

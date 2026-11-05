"""
=============================================================================
SIMULARE COMUNITATE ENERGETICA (9 case valide) + BATERIE COMUNITARA
=============================================================================

Obiectivul principal al proiectului: extindem testul facut pe o singura
casa (model_baterie.py, casa 2000901) la INTREAGA comunitate - toate cele 9
case valide din baza IRISE - cu o centrala PV comunitara si o baterie
comunitara la scara de cartier/district, exact ca in articolul profesorului
(Simoiu et al., 2021), unde optimizarea si indicatorii (SC, SS) sunt
formulati la nivel de COMUNITATE (community/district level), nu per casa.

CE REFOLOSESTE acest script (nimic nu se re-implementeaza de la zero):
  - din modelare_pv.py:  pipeline-ul IRISE (citire, curatare, marcarea
    caselor valide), pipeline-ul meteo/solar BATEM, constructia centralei PV
  - din model_baterie.py: modelul bateriei (simuleaza_baterie, Ecuatiile
    8-10), indicatorii SC/SS (calculeaza_SC_SS, Ecuatiile 1-2), alinierea
    seriilor de timp (alinieaza_serii) si graficele Plotly (genereaza_
    grafic_balanta_energetica / genereaza_grafic_evolutie_soc) - ACELEASI
    functii, doar aplicate pe seria de consum AGREGATA a comunitatii in loc
    de o singura casa.
  - din config_proiect.py: panoul real (Trina Solar Vertex S 425W).

PASII (cerinta profesorului):
  1. Identificam cele 9 case valide (status_case['Status'] == 'Valid').
  2. Procesam consumul orar curatat al FIECAREI case (completeaza_golurile +
     filtreaza_valorile_exagerate + calculeaza_consum_orar, exact ca la o
     singura casa - de fapt reutilizam direct pregateste_consum_orar_casa)
     si le insumam pe indexul temporal comun -> P_Load,comunitate(t).
  3. Dimensionam centrala PV comunitara la PANOURI_COMUNITATE = 50 panouri
     (~21,2 kWp) cu acelasi panou si aceeasi orientare optima (recalculata
     dinamic pentru amplasamentul comunitatii, care iese ~acelasi -4 deg /
     157 deg gasite si la Etapa individuala, fiind aceeasi locatie -
     Grenoble). IMPORTANT: acest numar de panouri NU e ales din nou aici -
     e preluat direct din config_proiect.py (PANOURI_COMUNITATE), unde a
     fost deja dimensionat "de mana" la Etapa 2 a proiectului
     (pv_comunitate.py), cu justificare explicita: 50 de panouri acopera
     ~89% din consumul anual cumulat al celor 9 case (aproape de
     neutralitate energetica), iar peste ~54 de panouri productia anuala
     ar depasi deja consumul. Am folosit initial 27 kWp (64 panouri) aici,
     dar asta contrazicea decizia deja luata la Etapa 2 - corectat acum ca
     sa ramana O SINGURA sursa a adevarului pentru marimea centralei
     comunitare, in tot proiectul.
  4. Dimensionam bateria comunitara la 50 kWh (sau 60 kWh), cu ACELEASI
     constrangeri de protectie din articol (SOC_min=0.20, SOC_max=0.90,
     SOC_initial=0.20, Delta_SOC_lim=0.35).
  5. Simulam ora cu ora (acelasi model ca la o casa - modelul e agnostic la
     scara: functioneaza identic daca P_PV/P_Load sunt ale unei case sau
     ale intregii comunitati).
  6. Calculam SC si SS comunitare, CU si FARA baterie (Ecuatiile 1-2).
  7. Generam graficele cerute:
       - grafic_balanta_energetica_comunitate.html
       - grafic_evolutie_soc_comunitate.html
=============================================================================
"""

import numpy as np
import pandas as pd

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
from model_baterie import (
    alinieaza_serii,
    simuleaza_baterie,
    calculeaza_SC_SS,
    genereaza_grafic_balanta_energetica,
    genereaza_grafic_evolutie_soc,
    SOC_MIN,
    SOC_MAX,
    SOC_INITIAL,
    DELTA_SOC_LIM,
    DT_H,
)


# =========================================================================
# ZONA DE CONFIGURARE
# =========================================================================

# --- Centrala PV comunitara: PANOURI_COMUNITATE panouri (~21,2 kWp) -----
# Preluat DIRECT din config_proiect.py, unde a fost deja dimensionat la
# Etapa 2 (pv_comunitate.py) - 50 de panouri, ~89% acoperire anuala a
# consumului celor 9 case. NU redimensionam din nou aici, ca sa nu apara
# doua numere diferite pentru "aceeasi" centrala comunitara in proiect.
NUMAR_PANOURI_COMUNITATE = PANOURI_COMUNITATE  # 50 panouri => ~21,2 kWp

# --- Bateria comunitara: 50 kWh (varianta 2 din cerinta: 60 kWh) --------
Q_NOM_ESS_COMUNITATE_KWH = 50.0
# SOC_min, SOC_max, SOC_initial, Delta_SOC_lim - ACELEASI ca la o casa,
# importate direct din model_baterie.py (SOC_MIN, SOC_MAX, SOC_INITIAL,
# DELTA_SOC_LIM), ca sa nu poata diverge intre cele doua etape.


# =========================================================================
# PASUL 1: identificam cele 9 case valide
# =========================================================================
def selecteaza_case_valide_comunitate(status_case):
    """
    Returneaza lista (sortata) a ID-urilor caselor cu Status == 'Valid' -
    aceleasi 9 case documentate in config_proiect.py: 2000901, 2000904,
    2000905, 2000906, 2000907, 2000908, 2000909, 2000911, 2000913.
    """
    case_valide_df = status_case.loc[status_case["Status"] == "Valid"]
    if case_valide_df.empty:
        raise ValueError("Nicio casa nu are Status == 'Valid' - nu putem forma o comunitate.")
    return sorted(int(h) for h in case_valide_df["HouseIDREF"])


# =========================================================================
# PASUL 2: agregarea consumului orar al celor 9 case
# =========================================================================
def agrega_consum_comunitate(df_consum, case_valide):
    """
    Pentru fiecare casa din `case_valide`, calculeaza consumul orar curatat
    (pregateste_consum_orar_casa - aceeasi functie folosita si pentru o
    singura casa, aplica exact completeaza_golurile + filtreaza_valorile_
    exagerate + calculeaza_consum_orar din MAIN_corectat.py), apoi
    INSUMEAZA toate cele 9 serii pe indexul temporal comun (uniunea orelor
    tuturor caselor - o ora lipsa la o casa individuala e tratata ca 0 Wh
    din partea ei la ora respectiva, NU exclude toata ora din comunitate).

    Returneaza (consum_comunitate, detaliu_pe_casa):
      consum_comunitate - Series (index=Ora, valori=Wh), suma comunitatii
      detaliu_pe_casa   - DataFrame, o coloana pe casa (util pentru debug/
                          verificare, nu e necesar mai departe in pipeline)
    """
    serii = []
    for id_casa in case_valide:
        consum_orar_casa = pregateste_consum_orar_casa(df_consum, id_casa)
        s = consum_orar_casa.set_index("Ora")["Consum_Orar_Casa_Wh"].astype(float)
        s = s[~s.index.duplicated(keep="first")]
        s.name = id_casa
        serii.append(s)
        print(f"  Casa {id_casa}: {len(s)} ore procesate, "
              f"consum total {s.sum() / 1000:.0f} kWh pe interval")

    return _insumeaza_serii_pe_ora(serii)


def _insumeaza_serii_pe_ora(serii):
    """
    Logica pura de agregare (separata de pregateste_consum_orar_casa /
    IRISE, ca sa poata fi testata independent): primeste o lista de
    pandas.Series (indexate dupa Ora, cate una per casa) si le insumeaza pe
    UNIUNEA indexului lor (o ora lipsa la o casa = 0 Wh din partea ei la
    acea ora, nu exclude ora din comunitate).

    Returneaza (consum_comunitate, detaliu_pe_casa) - vezi
    agrega_consum_comunitate.
    """
    detaliu_pe_casa = pd.concat(serii, axis=1)  # uniune (outer join) automata pe index
    ore_lipsa = int(detaliu_pe_casa.isna().sum().sum())
    if ore_lipsa:
        print(f"Atentie: {ore_lipsa} valori lipsa (case fara date la anumite ore din uniunea "
              f"orelor) - tratate ca 0 Wh la agregarea comunitatii.")

    consum_comunitate = detaliu_pe_casa.fillna(0.0).sum(axis=1)
    consum_comunitate.name = "Consum_Orar_Comunitate_Wh"
    return consum_comunitate, detaliu_pe_casa


def determina_interval_comunitate(df_consum, case_valide):
    """
    Data de inceput/sfarsit (format DD/MM/YYYY, cerut de BATEM) care
    acopera TOATE cele 9 case - min/max peste uniunea masuratorilor lor,
    nu doar a unei singure case (spre deosebire de extrage_interval_casa
    din modelare_pv.py, care lucreaza pe o singura casa).
    """
    date_comunitate = df_consum.loc[
        df_consum["HouseIDREF"].isin(case_valide), "DataOra"
    ]
    if date_comunitate.empty:
        raise ValueError("Nu exista nicio masuratoare pentru casele valide date.")
    data_inceput = date_comunitate.min().strftime("%d/%m/%Y")
    data_sfarsit = date_comunitate.max().strftime("%d/%m/%Y")
    return data_inceput, data_sfarsit


# =========================================================================
# PASUL 3+4+5+6: pipeline complet - centrala PV comunitara + baterie
# =========================================================================
def simuleaza_comunitate_cu_baterie(
    number_of_panels=NUMAR_PANOURI_COMUNITATE,
    Q_nom_kWh=Q_NOM_ESS_COMUNITATE_KWH,
    soc_min=SOC_MIN,
    soc_max=SOC_MAX,
    soc_initial=SOC_INITIAL,
    delta_soc_lim=DELTA_SOC_LIM,
    dt=DT_H,
):
    """
    Pipeline complet pentru comunitate: identificare case valide -> consum
    agregat -> meteo/solar BATEM pe intervalul comunitatii -> centrala PV
    comunitara -> aliniere -> simulare baterie -> indicatori SC/SS (cu si
    fara baterie).

    Returneaza un dictionar cu tot ce e necesar mai departe (grafice,
    rezumat, comparatii de sensibilitate):
      {
        "case_valide", "pv_system", "solar_model", "best_exposure",
        "best_slope", "rezultat_baterie", "p_pv_kw", "p_load_kw",
        "SC_cu_baterie", "SS_cu_baterie", "SC_fara_baterie", "SS_fara_baterie",
        "Q_nom_kWh", "soc_min", "soc_max",
      }
    """
    print("\n=== SIMULARE COMUNITATE: Pasul 1 - identificam casele valide ===")
    df_consum, status_case = ruleaza_pana_la_marcarea_caselor()
    case_valide = selecteaza_case_valide_comunitate(status_case)
    print(f"Case valide in comunitate ({len(case_valide)}): {case_valide}")

    print("\n=== SIMULARE COMUNITATE: Pasul 2 - agregarea consumului orar ===")
    consum_comunitate, _detaliu_pe_casa = agrega_consum_comunitate(df_consum, case_valide)
    print(f"Consum total comunitate (interval brut): {consum_comunitate.sum() / 1_000_000:.2f} MWh")

    data_inceput, data_sfarsit = determina_interval_comunitate(df_consum, case_valide)
    print(f"Interval comunitate (cerut BATEM): {data_inceput} -> {data_sfarsit}")

    # --- Pasul 3: date meteo + model solar + centrala PV comunitara -------
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
        f"{NUME_PANOU}, putere de varf {pv_system.peak_power_kW:.2f} kWp "
        f"(numar de panouri preluat din config_proiect.py, dimensionat la "
        f"Etapa 2), orientare optima "
        f"{best_exposure:.0f} grade expunere / {best_slope:.0f} grade inclinare"
    )

    # --- Pasul B (aliniere): reutilizam alinieaza_serii din model_baterie -
    consum_orar_comunitate_df = consum_comunitate.reset_index()
    consum_orar_comunitate_df.columns = ["Ora", "Consum_Orar_Casa_Wh"]
    df_aliniat = alinieaza_serii(solar_model.datetimes, pv_power_Wh, consum_orar_comunitate_df)

    p_pv_kw = df_aliniat["P_PV_Wh"].to_numpy() / 1000.0
    p_load_kw = df_aliniat["P_Load_Wh"].to_numpy() / 1000.0

    # --- Pasul 4+5: bateria comunitara + simularea ora cu ora --------------
    rezultat_baterie = simuleaza_baterie(
        p_pv_kw, p_load_kw,
        Q_nom_kWh=Q_nom_kWh, soc_min=soc_min, soc_max=soc_max,
        soc_initial=soc_initial, delta_soc_lim=delta_soc_lim, dt=dt,
    )
    rezultat_baterie.insert(0, "Ora", df_aliniat["Ora"].to_numpy())

    # --- Pasul 6: indicatorii SC / SS comunitari, CU si FARA baterie ------
    SC_cu_baterie, SS_cu_baterie = calculeaza_SC_SS(
        p_pv_kw, p_load_kw, rezultat_baterie["P_ESS_kW"].to_numpy(), dt=dt,
    )
    SC_fara_baterie, SS_fara_baterie = calculeaza_SC_SS(
        p_pv_kw, p_load_kw, np.zeros_like(p_pv_kw), dt=dt,
    )

    return {
        "case_valide": case_valide,
        "pv_system": pv_system,
        "solar_model": solar_model,
        "best_exposure": best_exposure,
        "best_slope": best_slope,
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
# Afisarea rezumatului comunitar in consola
# =========================================================================
def afiseaza_rezumat_comunitate(rezultate):
    df = rezultate["rezultat_baterie"]
    Q_nom = rezultate["Q_nom_kWh"]
    nr_case = len(rezultate["case_valide"])

    energie_incarcata_kWh = -df.loc[df["P_ESS_kW"] < 0, "P_ESS_kW"].sum() * DT_H
    energie_descarcata_kWh = df.loc[df["P_ESS_kW"] > 0, "P_ESS_kW"].sum() * DT_H
    energie_tranzitata_kWh = energie_incarcata_kWh + energie_descarcata_kWh
    energie_retea_kWh = df["P_Grid_kW"].sum() * DT_H
    energie_pv_kWh = df["P_PV_kW"].sum() * DT_H
    energie_load_kWh = df["P_Load_kW"].sum() * DT_H

    zile_interval = len(df) / 24
    energie_pv_anualizata_MWh = (energie_pv_kWh / 1000) * 365 / zile_interval

    print(f"\n=== REZUMAT SIMULARE COMUNITATE ({nr_case} case) - "
          f"Q_nom = {Q_nom:.1f} kWh ===")
    print(f"Case incluse: {rezultate['case_valide']}")
    print(f"Centrala PV: {rezultate['pv_system'].number_of_panels} panouri, "
          f"{rezultate['pv_system'].peak_power_kW:.2f} kWp")
    print(f"Interval simulat: {len(df)} ore ({zile_interval:.1f} zile)")
    print(f"Productie PV totala pe interval: {energie_pv_kWh / 1000:.2f} MWh "
          f"(anualizata: {energie_pv_anualizata_MWh:.2f} MWh/an)")
    print(f"Consum comunitar total pe interval: {energie_load_kWh / 1000:.2f} MWh")
    print(f"Energie tranzitata prin baterie: {energie_tranzitata_kWh / 1000:.2f} MWh "
          f"(incarcata {energie_incarcata_kWh / 1000:.2f} MWh, "
          f"descarcata {energie_descarcata_kWh / 1000:.2f} MWh)")
    print(f"Energie importata din retea: {energie_retea_kWh / 1000:.2f} MWh")
    print(f"SOC minim / maxim atins: {df['SOC_next'].min() * 100:.1f}% / "
          f"{df['SOC_next'].max() * 100:.1f}%")

    print(f"\n--- Indicatori de performanta comunitari (Ecuatiile 1-2) ---")
    print(f"{'':22s}{'FARA baterie':>15s}{'CU baterie':>15s}{'Diferenta':>15s}")
    print(f"{'SC (Self-Consumption)':22s}"
          f"{100 * rezultate['SC_fara_baterie']:>14.1f}%"
          f"{100 * rezultate['SC_cu_baterie']:>14.1f}%"
          f"{100 * (rezultate['SC_cu_baterie'] - rezultate['SC_fara_baterie']):>+14.1f}pp")
    print(f"{'SS (Self-Sufficiency)':22s}"
          f"{100 * rezultate['SS_fara_baterie']:>14.1f}%"
          f"{100 * rezultate['SS_cu_baterie']:>14.1f}%"
          f"{100 * (rezultate['SS_cu_baterie'] - rezultate['SS_fara_baterie']):>+14.1f}pp")


# =========================================================================
# Comparatie eficienta de sensibilitate (fara sa re-cerem meteo/consum)
# =========================================================================
def recalculeaza_varianta(rezultate_baza, number_of_panels=None, Q_nom_kWh=None):
    """
    Recalculeaza SC/SS pentru un alt numar de panouri PV si/sau alta
    capacitate de baterie, FARA sa reia pipeline-ul greu (IRISE + BATEM
    meteo/solar), pentru ca:
      - productia PV, la ACEEASI orientare si locatie, e liniara in numarul
        de panouri (acelasi model solar) - deci putem doar rescala seria
        p_pv_kw deja calculata, in loc sa reconstruim centrala PV;
      - consumul comunitatii (p_load_kw) nu depinde deloc de dimensionarea
        PV/baterie.
    Doar simularea bateriei (rapida - o singura bucla peste ore) se ruleaza
    din nou, cu noii parametri.
    """
    p_load_kw = rezultate_baza["p_load_kw"]
    p_pv_kw = rezultate_baza["p_pv_kw"]

    if number_of_panels is not None:
        panouri_baza = rezultate_baza["pv_system"].number_of_panels
        p_pv_kw = p_pv_kw * (number_of_panels / panouri_baza)

    Q_nom = Q_nom_kWh if Q_nom_kWh is not None else rezultate_baza["Q_nom_kWh"]

    rezultat_baterie = simuleaza_baterie(
        p_pv_kw, p_load_kw,
        Q_nom_kWh=Q_nom, soc_min=rezultate_baza["soc_min"], soc_max=rezultate_baza["soc_max"],
        soc_initial=SOC_INITIAL, delta_soc_lim=DELTA_SOC_LIM, dt=DT_H,
    )
    SC_cu, SS_cu = calculeaza_SC_SS(p_pv_kw, p_load_kw, rezultat_baterie["P_ESS_kW"].to_numpy())
    SC_fara, SS_fara = calculeaza_SC_SS(p_pv_kw, p_load_kw, np.zeros_like(p_pv_kw))
    return SC_cu, SS_cu, SC_fara, SS_fara


# =========================================================================
# ZONA DE COMENZI
# =========================================================================
if __name__ == "__main__":
    rezultate = simuleaza_comunitate_cu_baterie(
        number_of_panels=NUMAR_PANOURI_COMUNITATE,   # 50 panouri (Etapa 2), ~21,2 kWp
        Q_nom_kWh=Q_NOM_ESS_COMUNITATE_KWH,            # 50 kWh
    )
    afiseaza_rezumat_comunitate(rezultate)

    genereaza_grafic_balanta_energetica(
        rezultate["rezultat_baterie"], id_casa="comunitate",
        eticheta=f"Comunitate ({len(rezultate['case_valide'])} case)",
    )
    genereaza_grafic_evolutie_soc(
        rezultate["rezultat_baterie"], id_casa="comunitate",
        soc_min=rezultate["soc_min"], soc_max=rezultate["soc_max"],
        eticheta=f"Comunitate ({len(rezultate['case_valide'])} case)",
    )

    # --- Comparatie suplimentara de sensibilitate, calculata EFICIENT (fara
    #     sa reia pipeline-ul IRISE + BATEM, doar rescaland productia PV si
    #     reruland bateria). Folosim 54 panouri ca a doua varianta PV - nu
    #     un numar arbitrar, ci exact pragul identificat la Etapa 2
    #     (pv_comunitate.py) peste care productia anuala incepe sa
    #     depaseasca consumul comunitatii. -----------------------------------
    print("\n\n=== COMPARATIE SUPLIMENTARA: sensibilitate PV / capacitate baterie ===")
    variante = [
        ("50 panouri / 50 kWh (baza)", None, None),
        ("54 panouri / 50 kWh", 54, None),
        ("50 panouri / 60 kWh", None, 60.0),
        ("54 panouri / 60 kWh", 54, 60.0),
    ]
    print(f"{'Varianta':28s}{'SC fara':>10s}{'SC cu':>10s}{'SS fara':>10s}{'SS cu':>10s}")
    for nume, panouri, q_nom in variante:
        SC_cu, SS_cu, SC_fara, SS_fara = recalculeaza_varianta(
            rezultate, number_of_panels=panouri, Q_nom_kWh=q_nom,
        )
        print(f"{nume:28s}{100*SC_fara:>9.1f}%{100*SC_cu:>9.1f}%{100*SS_fara:>9.1f}%{100*SS_cu:>9.1f}%")

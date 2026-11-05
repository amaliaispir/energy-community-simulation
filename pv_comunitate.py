"""
=============================================================================
DIMENSIONAREA CENTRALEI PV PENTRU COMUNITATEA ENERGETICA (cele 9 case valide)
=============================================================================

Etapa urmatoare fata de modelare_pv.py: acolo suprapuneam productia unei
centrale PV peste consumul UNEI SINGURE case. Aici dimensionam o centrala
pentru o COMUNITATE - toate cele 9 case valide gasite de pipeline-ul din
MAIN_corectat.py (2000901, 2000904, 2000905, 2000906, 2000907, 2000908,
2000909, 2000911, 2000913) - adica vedem consumul lor CUMULAT (suma orara
a celor 9 case) suprapus peste productia unei centrale a carei capacitate
(numar de panouri) o alegem NOI, cu ochiul, uitandu-ne la grafic.

DE CE "CUMULAT" SI NU "O CASA REPREZENTATIVA":
  O comunitate energetica nu produce/consuma pentru o casa "medie" - suma
  cererilor tuturor caselor conteaza, pentru ca centrala fizica trebuie sa
  fie destul de mare cat sa acopere consumul simultan al tuturor caselor
  la un loc (sau macar o parte rezonabila din el).

DE CE UN INTERVAL DE TIMP COMUN:
  Cele 9 case au fost monitorizate in perioade usor diferite (nu toate au
  pornit/oprit exact in aceeasi zi). Ca sa le adunam ora cu ora, avem
  nevoie de un interval in care AVEM date pentru (ideal) toate cele 9, nu
  doar pentru o parte din ele intr-o luna si pentru restul in alta luna.
  PASUL 2 de mai jos verifica intai daca toate 9 se suprapun simultan; daca
  nu s-ar suprapune toate (nu e cazul pe datele astea, dar codul trateaza
  si situatia asta), cauta automat intervalul continuu in care se suprapun
  CELE MAI MULTE case posibil, si spune explicit care case au fost excluse.

DE CE ALEGEM CAPACITATEA "DE MANA", NU PRINTR-UN ALGORITM DE OPTIMIZARE:
  Cerinta explicita a proiectului: uitandu-ne la grafic (productie vs.
  consum cumulat), alegem NOI un numar de panouri care pare potrivit - nici
  prea mic (sa nu acopere aproape deloc consumul), nici absurd de mare
  (surplus urias, cost nejustificat). Scriptul NU alege automat un numar
  "optim" - doar deseneaza graficele si calculeaza rezumatele numerice
  (inclusiv separat pe iarna si pe vara) care ne ajuta sa alegem noi, si
  afiseaza un tabel comparativ intre cateva capacitati candidate, ca punct
  de plecare inainte de alegerea finala.

ATENTIE - IARNA vs. VARA:
  Productia solara la Grenoble variaza enorm intre iarna si vara (zile mai
  scurte, soare mai jos pe cer, mai multi nori iarna). O capacitate aleasa
  uitandu-ne DOAR la vara ar iesi absurd de mare iarna (supra-productie
  fara sens) sau, invers, o capacitate aleasa doar iarna ar fi mult prea
  mica vara. De-aia graficele si rezumatele de mai jos arata EXPLICIT
  iarna (Decembrie-Ianuarie-Februarie) si vara (Iunie-Iulie-August)
  separat, ca sa alegem o capacitate rezonabila in ambele anotimpuri, nu
  doar in unul singur.

STRUCTURA SCRIPTULUI (extinde modelare_pv.py de la 1 casa la 9 case):

  PASUL 1  Rulam pipeline-ul de baza din MAIN_corectat.py COMPLET (pana la
           consumul orar), dar pentru TOATE casele valide deodata - nu
           reluam pipeline-ul separat pentru fiecare casa.
  PASUL 2  Gasim intervalul de timp comun celor 9 case (sau, daca nu s-ar
           suprapune toate, celor mai multe case posibil).
  PASUL 3  Cumulam (adunam ora cu ora) consumul caselor acoperite de
           intervalul comun -> "consumul cumulat al comunitatii".
  PASUL 4  Modelam centrala PV (BATEM), pe acelasi interval de timp gasit
           la Pasul 2, la orientarea optima.
  PASUL 5  (a) tabel comparativ intre cateva capacitati candidate (numar de
           panouri), cu acoperire an/iarna/vara - ajutor pentru alegere;
           (b) capacitatea ALEASA DE NOI (variabila NUMBER_OF_PANELS, de
           schimbat manual dupa ce ne uitam la grafice); (c) grafic orar,
           pe tot intervalul, cu benzi de fundal care marcheaza vizual
           iarna/vara; (d) grafic lunar (productie vs. consum cumulat, pe
           luni calendaristice) - mult mai usor de citit cu ochiul liber
           pentru dimensionare decat graficul orar; (e) rezumat numeric.
=============================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from MAIN_corectat import (
    incarca_date,
    converteste_timpul,
    marcheaza_valorile_suspecte,
    marcheaza_case,
    completeaza_golurile,
    filtreaza_valorile_exagerate,
    calculeaza_consum_orar,
)
from modelare_pv import (
    construieste_date_meteo,
    construieste_model_solar,
    orientare_optima,
    construieste_centrala_pv,
    calculeaza_productie_pv,
)
# Panoul folosit si numarul de panouri ales "de mana" pentru comunitate stau
# in config_proiect.py, ca sa fie ACELEASI in toate etapele - vezi
# PARAMETRI_PANOU si PANOURI_COMUNITATE acolo.
from config_proiect import PARAMETRI_PANOU, PANOURI_COMUNITATE, CASE_IN_COMUNITATE

# Lunile pe care le tratam ca "iarna" / "vara" pentru rezumate si benzile
# de pe grafic (Grenoble, emisfera nordica).
LUNI_IARNA = {12, 1, 2}
LUNI_VARA = {6, 7, 8}


# =========================================================================
# PASUL 1: consumul orar al TUTUROR caselor valide, dintr-o singura trecere
# =========================================================================
def pregateste_consum_orar_comunitate(id_case=None):
    """
    Ruleaza pipeline-ul complet din MAIN_corectat.py (Pasii 1, 2a, 2b, 2c,
    2d, 3, 4), dar SE OPRESTE inainte de partea de grafice individuale pe
    casa/aparat (Pasul 5 din MAIN_corectat.py) - aici ne trebuie doar
    consumul orar pe casa, pentru toate casele comunitatii deodata (nu
    facem un pas separat per casa, ca sa nu recalculam de 9 ori acelasi
    lucru si sa nu regeneram graficele individuale, care exista deja).

    PARAMETRU:
      id_case - None (implicit) = toate casele cu Status == "Valid" (in
                practica, cele 9 case ale comunitatii); sau o lista
                explicita de ID-uri (ex. [2000901, 2000904, 2000905]) DACA
                vrem o sub-comunitate mai mica - dar numai case care sunt
                deja in lista celor valide, altfel ridicam o eroare clara
                cu lista completa a caselor valide disponibile.

    Returneaza (consum_orar_casa, case_folosite, status_case):
      consum_orar_casa - DataFrame (HouseIDREF, Ora, Consum_Orar_Casa_Wh),
                          pentru casele din case_folosite
      case_folosite    - lista de HouseIDREF, sortata
      status_case      - tabelul complet de status (util pentru diagnoza)
    """
    df_consum, df_aparate, df_contor_general = incarca_date()
    df_consum = converteste_timpul(df_consum)
    df_consum = marcheaza_valorile_suspecte(df_consum)
    status_case, raport_lunar = marcheaza_case(df_consum)

    case_valide = sorted(
        int(h) for h in status_case.loc[status_case["Status"] == "Valid", "HouseIDREF"]
    )
    if not case_valide:
        raise ValueError("Nicio casa nu e valida - nu putem forma o comunitate.")

    if id_case is None:
        case_folosite = case_valide
    else:
        case_folosite = sorted(int(h) for h in id_case)
        invalide = [h for h in case_folosite if h not in case_valide]
        if invalide:
            raise ValueError(
                f"Casele {invalide} nu sunt in lista caselor valide. "
                f"Case valide disponibile: {case_valide}"
            )

    df_consum = df_consum[df_consum["HouseIDREF"].isin(case_folosite)].copy()

    # --- Pasul 2d: completam golurile ramase (media vecinilor / ora din zi)
    df_consum = completeaza_golurile(df_consum)

    # --- Pasul 3: taiem valorile individuale exagerate care au scapat de 2b
    df_consum = filtreaza_valorile_exagerate(df_consum)

    # --- Pasul 4: consum pe ora, pentru toate casele deodata --------------
    _, consum_orar_casa = calculeaza_consum_orar(df_consum)

    print(f"\n=== COMUNITATE: {len(case_folosite)} case folosite: {case_folosite} ===")
    return consum_orar_casa, case_folosite, status_case


# =========================================================================
# PASUL 2: intervalul de timp comun al comunitatii
# =========================================================================
def calculeaza_intervalele_caselor(consum_orar_casa):
    """Pentru fiecare casa, prima si ultima ora cu date (dupa Pasul 1)."""
    return (
        consum_orar_casa.groupby("HouseIDREF")["Ora"]
        .agg(Ora_Min="min", Ora_Max="max")
        .reset_index()
    )


def gaseste_interval_comun(intervale_case):
    """
    Primeste intervale_case (HouseIDREF, Ora_Min, Ora_Max) si gaseste
    intervalul de timp comun al comunitatii:

      1. Verificam INTAI daca TOATE casele se suprapun simultan - adica
         exista un moment de la care toate au inceput deja si pana la
         care niciuna nu s-a mai oprit: [max(Ora_Min), min(Ora_Max)]. Daca
         acest interval e valid (inceputul <= sfarsitul), il folosim direct
         si comunitatea ramane cu toate casele.

      2. DACA NU (cazul in care unele case nu se suprapun deloc cu
         altele), cautam intervalul continuu in care se suprapun CAT MAI
         MULTE case: pentru fiecare pereche de puncte consecutive dintre
         toate capetele de interval (Ora_Min/Ora_Max ale tuturor caselor),
         numaram cate case acopera in intregime acel sub-interval, si
         pastram sub-intervalul cu numarul maxim. Intre doua asemenea
         puncte consecutive, setul de case care acopera nu se schimba (nu
         incepe/se termina nicio casa la mijloc), deci verificarea e
         corecta si completa, nu doar o aproximare. Cu 9 case (n mic),
         complexitatea O(n^2) a acestei cautari e nesemnificativa.

    Returneaza (data_inceput, data_sfarsit, case_acoperite, toate_se_suprapun):
      data_inceput, data_sfarsit - Timestamp-uri (cu fus orar), primele si
                                    ultimele ore ale intervalului ales
      case_acoperite              - lista de HouseIDREF acoperite de acel
                                    interval, sortata
      toate_se_suprapun           - True daca a fost nevoie doar de
                                    varianta 1 (toate casele), False daca
                                    a fost nevoie de cautarea de la 2
    """
    inceput_toate = intervale_case["Ora_Min"].max()
    sfarsit_toate = intervale_case["Ora_Max"].min()

    if inceput_toate <= sfarsit_toate:
        case_acoperite = sorted(int(h) for h in intervale_case["HouseIDREF"])
        return inceput_toate, sfarsit_toate, case_acoperite, True

    print(
        "\n!! Casele NU se suprapun toate pe acelasi interval - cautam "
        "intervalul continuu in care se suprapun CAT MAI MULTE case."
    )
    puncte = sorted(set(intervale_case["Ora_Min"]) | set(intervale_case["Ora_Max"]))
    mai_bun_nr, mai_bun_start, mai_bun_end = 0, None, None
    for t_start, t_end in zip(puncte[:-1], puncte[1:]):
        acopera = intervale_case[
            (intervale_case["Ora_Min"] <= t_start) & (intervale_case["Ora_Max"] >= t_end)
        ]
        if len(acopera) > mai_bun_nr:
            mai_bun_nr, mai_bun_start, mai_bun_end = len(acopera), t_start, t_end

    case_acoperite = sorted(int(h) for h in intervale_case.loc[
        (intervale_case["Ora_Min"] <= mai_bun_start) & (intervale_case["Ora_Max"] >= mai_bun_end),
        "HouseIDREF",
    ])
    return mai_bun_start, mai_bun_end, case_acoperite, False


def _formateaza_data_batem(timestamp):
    """La fel ca in modelare_pv.py: 'DD/MM/YYYY', formatul cerut de BATEM."""
    return timestamp.strftime("%d/%m/%Y")


def sincronizeaza_interval_comunitate(consum_orar_casa):
    """
    Leaga cap la coada Pasul 2: calculeaza intervalele fiecarei case,
    gaseste intervalul comun, afiseaza un raport clar, si intoarce tot ce
    e nevoie mai departe (atat ca text DD/MM/YYYY pentru BATEM, cat si ca
    Timestamp-uri, pentru filtrarea datelor de consum).
    """
    intervale_case = calculeaza_intervalele_caselor(consum_orar_casa)
    inceput, sfarsit, case_acoperite, toate = gaseste_interval_comun(intervale_case)

    print("\n=== PASUL 2: intervalul de timp comun al comunitatii ===")
    print(
        intervale_case.assign(
            Ora_Min=intervale_case["Ora_Min"].dt.strftime("%d/%m/%Y"),
            Ora_Max=intervale_case["Ora_Max"].dt.strftime("%d/%m/%Y"),
        ).to_string(index=False)
    )
    if toate:
        print(f"\nToate cele {len(case_acoperite)} case se suprapun pe acelasi interval.")
    else:
        excluse = sorted(set(int(h) for h in intervale_case["HouseIDREF"]) - set(case_acoperite))
        print(
            f"\nIntervalul cu cele mai multe case suprapuse acopera "
            f"{len(case_acoperite)} din {len(intervale_case)} case: {case_acoperite}"
        )
        print(f"Case excluse din comunitatea analizata (nu acopera acest interval): {excluse}")

    data_inceput = _formateaza_data_batem(inceput)
    data_sfarsit = _formateaza_data_batem(sfarsit)
    zile = (sfarsit - inceput).total_seconds() / 86400
    print(f"Interval comun folosit: {data_inceput} -> {data_sfarsit} ({zile:.0f} zile)")

    return {
        "data_inceput": data_inceput,
        "data_sfarsit": data_sfarsit,
        "inceput_ts": inceput,
        "sfarsit_ts": sfarsit,
        "case_acoperite": case_acoperite,
    }


# =========================================================================
# PASUL 3: consumul cumulat al comunitatii (suma orara a caselor acoperite)
# =========================================================================
def cumuleaza_consum_comunitate(consum_orar_casa, case_acoperite, inceput_ts, sfarsit_ts):
    """
    Restrange consum_orar_casa la casele din `case_acoperite` si la
    intervalul [inceput_ts, sfarsit_ts] (Timestamp-uri, deja la rezolutie
    orara), apoi aduna, ora cu ora, consumul TUTUROR acelor case - asta e
    "consumul cumulat al comunitatii" cerut de tema.

    Pastram si o coloana Nr_Case cu cate case au avut efectiv o
    masuratoare in acea ora (diagnostic - dupa completeaza_golurile ar
    trebui sa fie aproape mereu egal cu len(case_acoperite); daca nu e,
    inseamna ca a mai ramas un gol undeva si suma orei respective e
    subestimata).
    """
    df = consum_orar_casa[
        consum_orar_casa["HouseIDREF"].isin(case_acoperite)
        & consum_orar_casa["Ora"].between(inceput_ts, sfarsit_ts)
    ]

    consum_comunitate = (
        df.groupby("Ora", as_index=False)
        .agg(
            Consum_Comunitate_Wh=("Consum_Orar_Casa_Wh", "sum"),
            Nr_Case=("HouseIDREF", "nunique"),
        )
        .sort_values("Ora")
        .reset_index(drop=True)
    )

    ore_incomplete = int((consum_comunitate["Nr_Case"] < len(case_acoperite)).sum())
    print(f"\n=== PASUL 3: consum cumulat al comunitatii ({len(case_acoperite)} case) ===")
    print(f"Ore in intervalul comun: {len(consum_comunitate)}")
    if ore_incomplete:
        print(
            f"ATENTIE: {ore_incomplete} ore au date de la mai putin de "
            f"{len(case_acoperite)} case (posibil gol ramas) - suma lor e subestimata."
        )
    else:
        print(
            f"Toate orele au date de la toate cele {len(case_acoperite)} case - "
            "suma e completa peste tot."
        )
    return consum_comunitate


# =========================================================================
# PASUL 5a: tabel comparativ intre cateva capacitati candidate
# =========================================================================
def _acoperire(productie_Wh, consum_Wh):
    """Productie totala, consum total (kWh) si gradul de acoperire (%)."""
    productie_kWh = float(np.nansum(productie_Wh)) / 1000
    consum_kWh = float(np.nansum(consum_Wh)) / 1000
    acoperire = 100 * productie_kWh / consum_kWh if consum_kWh else float("nan")
    return productie_kWh, consum_kWh, acoperire


def compara_capacitati(
    solar_model, best_exposure, best_slope, consum_comunitate, lista_numar_panouri,
    **kwargs_panou,
):
    """
    Pentru fiecare numar de panouri din `lista_numar_panouri`, construieste
    o centrala PV (aceeasi orientare optima, acelasi tip de panou) si
    calculeaza acoperirea (productie / consum) pe an intreg, iarna si
    vara - INAINTE sa alegem definitiv NUMBER_OF_PANELS mai jos.

    E doar un ajutor numeric pentru alegerea manuala, NU un optimizator
    automat - decizia finala tot noi o luam, uitandu-ne si la grafice (mai
    jos), nu doar la acest tabel.
    """
    luni_cons = consum_comunitate["Ora"].dt.month.to_numpy()
    cons = consum_comunitate["Consum_Comunitate_Wh"].to_numpy()
    cons_iarna = cons[np.isin(luni_cons, list(LUNI_IARNA))]
    cons_vara = cons[np.isin(luni_cons, list(LUNI_VARA))]

    luni_prod = pd.Series(solar_model.datetimes).dt.month.to_numpy()

    print("\n=== PASUL 5a: comparatie capacitati candidate (inainte de alegerea finala) ===")
    print(f"{'Panouri':>8} {'kWp':>7} {'Acop. an':>10} {'Acop. iarna':>13} {'Acop. vara':>12}")
    for n in lista_numar_panouri:
        pv_system = construieste_centrala_pv(
            solar_model, best_exposure, best_slope, number_of_panels=n, **kwargs_panou,
        )
        prod = np.asarray(calculeaza_productie_pv(pv_system), dtype=float)
        _, _, ac_an = _acoperire(prod, cons)
        _, _, ac_iarna = _acoperire(prod[np.isin(luni_prod, list(LUNI_IARNA))], cons_iarna)
        _, _, ac_vara = _acoperire(prod[np.isin(luni_prod, list(LUNI_VARA))], cons_vara)
        print(
            f"{n:>8} {pv_system.peak_power_kW:>7.2f} {ac_an:>9.0f}% "
            f"{ac_iarna:>12.0f}% {ac_vara:>11.0f}%"
        )
    print(
        "\n(Acoperirea = productie PV / consum cumulat al comunitatii, in %. "
        "100% ar insemna productie exact egala cu consumul pe acel interval - "
        "nu inseamna ca fiecare ora individuala e acoperita, doar totalul.)"
    )


# =========================================================================
# PASUL 5c/d: grafice - orar (cu iarna/vara marcate) si lunar
# =========================================================================
def _blocuri_de_zile_pentru_luni(datetimes, luni_dorite):
    """
    Din seria de date/ore `datetimes`, gaseste blocurile continue de zile
    care apartin lunilor din `luni_dorite` (ex. {12, 1, 2} pentru iarna) -
    returneaza o lista de (inceput, sfarsit), folosita ca sa desenam benzi
    colorate de fundal (vrect) pe grafic, ca sa se vada dintr-o privire
    unde e iarna si unde e vara.
    """
    zile = pd.Series(sorted(set(pd.Series(datetimes).dt.floor("D"))))
    e_anotimp = zile.dt.month.isin(luni_dorite)

    blocuri = []
    start = None
    for i in range(len(zile)):
        if e_anotimp.iloc[i] and start is None:
            start = zile.iloc[i]
        if not e_anotimp.iloc[i] and start is not None:
            blocuri.append((start, zile.iloc[i - 1] + pd.Timedelta(days=1)))
            start = None
    if start is not None:
        blocuri.append((start, zile.iloc[-1] + pd.Timedelta(days=1)))
    return blocuri


def genereaza_grafic_comunitate(
    solar_model, pv_power, consum_comunitate, case_acoperite,
    nume_fisier="grafic_comunitate_productie_vs_consum.html",
):
    """
    Suprapune, pe aceeasi axa de timp, productia orara a centralei PV si
    consumul orar CUMULAT al comunitatii (in kWh, nu Wh - valorile insumate
    pe 9 case sunt mult mai mari decat pentru o singura casa). Adauga benzi
    de fundal albastre (iarna) si galbene (vara), ca sezonalitatea sa se
    vada dintr-o privire, fara sa fie nevoie sa numeri lunile pe axa X.
    """
    fig = go.Figure()

    for start, end in _blocuri_de_zile_pentru_luni(solar_model.datetimes, LUNI_IARNA):
        fig.add_vrect(x0=start, x1=end, fillcolor="#3b7fc4", opacity=0.08, line_width=0, layer="below")
    for start, end in _blocuri_de_zile_pentru_luni(solar_model.datetimes, LUNI_VARA):
        fig.add_vrect(x0=start, x1=end, fillcolor="#e6a800", opacity=0.08, line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=solar_model.datetimes, y=np.asarray(pv_power, dtype=float) / 1000,
        name=f"Productie PV ({len(pv_power)} ore)",
        line=dict(color="#e6a800"),
    ))
    fig.add_trace(go.Scatter(
        x=consum_comunitate["Ora"], y=consum_comunitate["Consum_Comunitate_Wh"] / 1000,
        name=f"Consum cumulat, {len(case_acoperite)} case",
        line=dict(color="#1f77b4"),
    ))

    fig.update_layout(
        title=(
            f"Comunitate ({len(case_acoperite)} case {case_acoperite}): productie PV vs. "
            "consum cumulat  |  benzi albastre = iarna (Dec-Ian-Feb), "
            "benzi galbene = vara (Iun-Iul-Aug)"
        ),
        hovermode="x unified",
        xaxis_title="Data si ora",
        yaxis_title="Energie pe ora (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"\nGrafic comunitate (orar) salvat ca: {nume_fisier}")
    return fig


def _agregheaza_lunar(datetimes, valori_Wh):
    """Suma lunara (kWh), din serii orare (Wh) - folosit pentru graficul lunar."""
    s = pd.Series(np.asarray(valori_Wh, dtype=float), index=pd.DatetimeIndex(pd.Series(datetimes)))
    return s.resample("MS").sum() / 1000


def genereaza_grafic_lunar(
    solar_model, pv_power, consum_comunitate, case_acoperite,
    nume_fisier="grafic_comunitate_lunar.html",
):
    """
    Totaluri LUNARE (kWh/luna), productie vs. consum cumulat - graficul cel
    mai util pentru alegerea "de mana" a capacitatii, pentru ca elimina
    zgomotul orar si arata direct, luna cu luna, unde productia e mult sub
    consum (iarna) si unde e peste (vara).
    """
    productie_lunara = _agregheaza_lunar(solar_model.datetimes, pv_power)
    consum_lunar = _agregheaza_lunar(
        consum_comunitate["Ora"], consum_comunitate["Consum_Comunitate_Wh"],
    )

    luni = sorted(set(productie_lunara.index) | set(consum_lunar.index))
    etichete = [
        f"{l.strftime('%b %Y')} (iarna)" if l.month in LUNI_IARNA
        else f"{l.strftime('%b %Y')} (vara)" if l.month in LUNI_VARA
        else l.strftime("%b %Y")
        for l in luni
    ]
    prod = [float(productie_lunara.get(l, 0.0)) for l in luni]
    cons = [float(consum_lunar.get(l, 0.0)) for l in luni]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=etichete, y=prod, name="Productie PV (kWh/luna)", marker_color="#e6a800"))
    fig.add_trace(go.Bar(
        x=etichete, y=cons, name=f"Consum cumulat, {len(case_acoperite)} case (kWh/luna)",
        marker_color="#1f77b4",
    ))

    fig.update_layout(
        barmode="group",
        title=(
            "Totaluri lunare: productie PV vs. consum cumulat (comunitate) - "
            "cel mai simplu grafic pentru dimensionare"
        ),
        xaxis_title="Luna",
        yaxis_title="Energie (kWh/luna)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"Grafic lunar salvat ca: {nume_fisier}")
    return fig


# =========================================================================
# PASUL 5e: rezumat numeric (an intreg, iarna, vara)
# =========================================================================
def afiseaza_rezumat_comunitate(solar_model, pv_power, consum_comunitate, case_acoperite, pv_system=None):
    """
    Productie totala, consum total si grad de acoperire, calculate separat
    pentru anul intreg, pentru iarna (Dec-Ian-Feb) si pentru vara
    (Iun-Iul-Aug) - exact comparatia ceruta de tema, ca sa alegem o
    capacitate acceptabila si vara, si iarna, nu doar una dintre ele.
    """
    luni_prod = pd.Series(solar_model.datetimes).dt.month.to_numpy()
    luni_cons = consum_comunitate["Ora"].dt.month.to_numpy()

    prod = np.asarray(pv_power, dtype=float)
    cons = consum_comunitate["Consum_Comunitate_Wh"].to_numpy()

    total_prod, total_cons, ac_totala = _acoperire(prod, cons)
    iarna_prod, iarna_cons, ac_iarna = _acoperire(
        prod[np.isin(luni_prod, list(LUNI_IARNA))], cons[np.isin(luni_cons, list(LUNI_IARNA))],
    )
    vara_prod, vara_cons, ac_vara = _acoperire(
        prod[np.isin(luni_prod, list(LUNI_VARA))], cons[np.isin(luni_cons, list(LUNI_VARA))],
    )

    print(f"\n=== PASUL 5: rezumat comunitate ({len(case_acoperite)} case: {case_acoperite}) ===")
    if pv_system is not None:
        print(f"Centrala PV: {pv_system.number_of_panels} panouri, {pv_system.peak_power_kW:.2f} kWp")
    print(f"{'Interval':<15} {'Productie (kWh)':>16} {'Consum (kWh)':>14} {'Acoperire':>10}")
    print(f"{'An intreg':<15} {total_prod:>16.0f} {total_cons:>14.0f} {ac_totala:>9.0f}%")
    print(f"{'Iarna (D-I-F)':<15} {iarna_prod:>16.0f} {iarna_cons:>14.0f} {ac_iarna:>9.0f}%")
    print(f"{'Vara (I-I-A)':<15} {vara_prod:>16.0f} {vara_cons:>14.0f} {ac_vara:>9.0f}%")

    if ac_iarna < 20:
        print(
            "\n  => Iarna, centrala acopera foarte putin din consum (normal la Grenoble - "
            "zile scurte, soare jos, mai multi nori - dar sub ~15-20% poate insemna ca "
            "centrala e prea mica chiar si pentru un minim rezonabil de acoperire iarna)."
        )
    if ac_vara > 200:
        print(
            "\n  => Vara, productia depaseste consumul de peste 2x - surplus mare; de luat "
            "in calcul injectare in retea / stocare, sau o centrala mai mica."
        )

    return {
        "acoperire_an": ac_totala, "acoperire_iarna": ac_iarna, "acoperire_vara": ac_vara,
        "productie_an_kWh": total_prod, "consum_an_kWh": total_cons,
    }


# =========================================================================
# ZONA DE COMENZI
# =========================================================================
if __name__ == "__main__":
    # --- Pasul 1: consumul orar al tuturor caselor valide (comunitatea) --
    consum_orar_casa, case_folosite, status_case = pregateste_consum_orar_comunitate()

    # --- Pasul 2: intervalul de timp comun al comunitatii -----------------
    sync = sincronizeaza_interval_comunitate(consum_orar_casa)

    # --- Pasul 3: consumul cumulat al comunitatii, pe intervalul comun ---
    consum_comunitate = cumuleaza_consum_comunitate(
        consum_orar_casa, sync["case_acoperite"], sync["inceput_ts"], sync["sfarsit_ts"],
    )

    # --- Pasul 4: date meteo + model solar, pe acelasi interval -----------
    weather_data = construieste_date_meteo(sync["data_inceput"], sync["data_sfarsit"])
    solar_model = construieste_model_solar(weather_data)
    best_exposure, best_slope = orientare_optima(solar_model, initial_slope_deg=150, initial_exposure_deg=0)

    # Parametrii panoului (acelasi Trina Solar Vertex S 425W folosit in toate
    # etapele) vin acum din config_proiect.py, nu mai sunt scrisi de mana
    # aici - la scara unei comunitati se schimba doar NUMARUL de panouri, nu
    # tipul lor.

    # --- Pasul 5a: comparatie rapida intre cateva capacitati candidate ---
    # Ajustati lista asta dupa ordinul de marime al consumului cumulat pe
    # care il vedeti afisat la Pasul 3/5 (10 case pot avea nevoie de zeci
    # pana la >100 de panouri, spre deosebire de 4-7 pentru o singura casa).
    CANDIDATI_PANOURI = [20, 30, 40, 50, 60, 80, 100, 130]
    compara_capacitati(
        solar_model, best_exposure, best_slope, consum_comunitate, CANDIDATI_PANOURI,
        **PARAMETRI_PANOU,
    )

    # --- Pasul 5b: capacitatea ALEASA DE NOI, "de mana" --------------------
    # Uitati-va la tabelul de mai sus SI la graficele generate mai jos
    # (lunar e cel mai usor de citit, orar arata si detaliul zi de zi),
    # apoi schimbati manual numarul pana sunteti multumiti de cum arata
    # acoperirea SI iarna, SI vara (nu doar una dintre ele).
    #
    # Valoarea aleasa (40 de panouri pentru 9 case) sta acum in
    # config_proiect.py, ca PANOURI_COMUNITATE - SCHIMBATI-O ACOLO, nu aici.
    # Motivul: simularea multi-agent de la Etapa 3 isi calculeaza centrala ca
    # o cota-parte din aceasta valoare (vezi panouri_pentru()), deci daca
    # schimbati capacitatea comunitatii, se aliniaza automat si Etapa 3.
    NUMBER_OF_PANELS = PANOURI_COMUNITATE

    if len(sync["case_acoperite"]) != CASE_IN_COMUNITATE:
        print(
            f"\nATENTIE: dimensionarea din config_proiect.py presupune "
            f"{CASE_IN_COMUNITATE} case ({PANOURI_COMUNITATE} panouri), dar "
            f"intervalul comun a retinut {len(sync['case_acoperite'])} case. "
            f"Actualizati CASE_IN_COMUNITATE / PANOURI_COMUNITATE, altfel "
            f"cota-parte folosita la Etapa 3 iese calculata gresit."
        )

    pv_system = construieste_centrala_pv(
        solar_model, best_exposure, best_slope,
        number_of_panels=NUMBER_OF_PANELS, **PARAMETRI_PANOU,
    )
    pv_power = calculeaza_productie_pv(pv_system)

    # --- Pasul 5c/d: graficele --------------------------------------------
    genereaza_grafic_comunitate(solar_model, pv_power, consum_comunitate, sync["case_acoperite"])
    genereaza_grafic_lunar(solar_model, pv_power, consum_comunitate, sync["case_acoperite"])

    # --- Pasul 5e: rezumatul numeric ---------------------------------------
    afiseaza_rezumat_comunitate(
        solar_model, pv_power, consum_comunitate, sync["case_acoperite"], pv_system=pv_system,
    )

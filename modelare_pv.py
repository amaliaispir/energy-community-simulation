"""
=============================================================================
MODELARE FOTOVOLTAICA (BATEM) - suprapunere productie PV peste consumul casei
=============================================================================

Aceasta este etapa noua a proiectului: dupa ce MAIN_corectat.py stie sa
proceseze consumul din baza de date IRISE, aici incepem sa modelam o
centrala fotovoltaica (biblioteca BATEM) folosind date meteo reale
(Grenoble.json), ca sa putem compara productia solara cu consumul unei
case anume.

PASUL 0 (facut o singura data, manual/de mana, NU in acest script):
  - folderul "data/" a fost creat in radacina proiectului
  - "Grenoble.json" a fost mutat in "data/Grenoble.json"
  - "setup.ini" a fost creat in radacina proiectului, cu continutul:
        [folders]
        data = data/
  Asta e formatul de care are nevoie BATEM ca sa gaseasca datele meteo.

PASUL 1 (acest script):
  Nu putem modela productia solara "in gol" - trebuie sa stim exact pentru
  ce perioada de timp avem date de consum VALIDE, ca sa cerem lui BATEM
  meteo doar pentru acel interval. Asa ca:
    1. Rulam pipeline-ul de baza din MAIN_corectat.py, dar NE OPRIM la
       Pasul 2c (marcarea caselor ca Valid/Invalid) - nu avem nevoie de
       Pasul 2d (completare goluri) sau mai departe, doar de statusul
       caselor si de coloana DataOra.
    2. Alegem prima casa cu Status == "Valid".
    3. Scoatem data minima si maxima din DataOra, pentru casa aleasa.
    4. Le formatam ca text "DD/MM/YYYY", format cerut de BATEM.

PASUL 2 (acest script):
  Folosim SWDbuilder din BATEM (batem.core.weather) ca sa obtinem seria
  meteo orara pentru Grenoble, EXACT pe intervalul gasit la Pasul 1
  (data_inceput -> data_sfarsit). BATEM citeste datele din
  data/Grenoble.json (vezi Pasul 0) - daca fisierul local e deja complet,
  nu descarca nimic de pe internet.
  Graficul temperatura / nori / precipitatii NU mai foloseste functia
  built-in weather_data.plot(...): pe masini cu Tkinter instalat, aceea
  deschide o fereastra grafica ce BLOCHEAZA scriptul pana o inchizi manual
  - inacceptabil intr-un pipeline care trebuie sa continue singur pana la
  Pasul 3. In loc de asta, construim noi insine graficul, direct in
  Plotly, si il salvam ca fisier HTML (grafic_meteo_plotly.html) - nu se
  deschide nicio fereastra, rularea continua imediat.

PASUL 3 (acest script):
  Cu SolarModel din BATEM (batem.core.solar) calculam:
    - radiatia solara pe o suprafata orizontala (referinta)
    - orientarea optima a panoului (expunere + inclinare) pentru maximul
      de radiatie anuala
    - radiatia solara obtinuta pe orientarea optima
  Apoi construim un grafic propriu (plotly.graph_objects, axa Y
  secundara) care suprapune radiatia optima peste temperatura si
  acoperirea de nori, ca sa raspundem vizual la intrebarea profesorului:
  "ce observati intre radiatie, temperatura si nori?"

PASUL 4 (acest script):
  Construim o centrala fotovoltaica virtuala (PVplant din batem.core.solar),
  montata la orientarea optima gasita la Pasul 3, cu panoul REAL Trina
  Solar Vertex S 425W (1.13m x 1.76m, eficienta 21.3% - vezi NUME_PANOU si
  PARAMETRI_PANOU in config_proiect.py). Dam DIRECT numarul de panouri
  (PANOURI_O_CASA = 4, cota-parte a unei case din centrala comunitatii
  dimensionata la Etapa 2 - vezi panouri_pentru() in config_proiect.py) -
  nu o putere de varf tinta. Calculam productia orara de energie
  (pv_system.powers_W()). Puterea de varf a centralei (pv_system.peak_power_kW,
  ~1.69 kWp cu 4 panouri de acest tip) e un REZULTAT calculat de BATEM din
  numarul de panouri, eficienta si dimensiunile panoului - nu invers.
  Reprezentam apoi si PROFILUL DE PRODUCTIE pe tot anul, SINGUR (fara
  consumul suprapus peste el): un grafic cu doua panouri - profilul orar,
  ora cu ora, si totalurile lunare, unde sezonalitatea vara/iarna se vede
  dintr-o privire.

PASUL 5 (acest script):
  Calculam consumul orar al casei alese la Pasul 1 (cu calculeaza_consum_orar din
  MAIN_corectat.py, aplicata pe datele deja procesate la Pasul 1, DAR
  completate cu Pasul 2d si filtrate cu Pasul 3 din MAIN_corectat.py -
  altfel golurile si valorile aberante ramase produc varfuri false pe
  grafic) si il suprapunem, intr-un grafic Plotly, peste productia orara
  a centralei de la Pasul 4. Afisam si un rezumat numeric: productie
  anuala, consum anual, grad de acoperire.

In pasii urmatori vom folosi aceasta perioada (data_inceput, data_sfarsit)
ca sa cerem lui BATEM modelul meteo si productia PV pentru exact acelasi
interval pentru care avem consum masurat, ca cele doua profiluri (consum
si productie) sa poata fi suprapuse pe acelasi orizont de timp.
=============================================================================
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd

from MAIN_corectat import (
    incarca_date,
    converteste_timpul,
    marcheaza_case,
    marcheaza_valorile_suspecte,
    completeaza_golurile,
    filtreaza_valorile_exagerate,
    calculeaza_consum_orar,
)
# Cifrele comune tuturor etapelor (casa analizata, panoul, numarul de
# panouri) stau intr-un singur loc, ca sa nu mai difere de la un script la
# altul - vezi config_proiect.py.
from config_proiect import (
    CASA_IMPLICITA,
    NUME_PANOU,
    PARAMETRI_PANOU,
    PANOURI_O_CASA,
)


# =========================================================================
# PASUL 1a: pipeline-ul de baza, oprit la marcarea caselor (Pasul 2c)
# =========================================================================
def ruleaza_pana_la_marcarea_caselor():
    """
    Reia exact primii pasi din main() al MAIN_corectat.py (Pasul 1, 2a, 2b,
    2c), FARA sa mearga mai departe (nu completam goluri, nu filtram
    valori exagerate, nu calculam consum orar) - pentru scopul de-aici
    (aflarea perioadei valide de date) ne trebuie doar df_consum cu coloana
    DataOra si tabelul de status al caselor.

    Returneaza:
      df_consum   - DataFrame cu masuratorile (coloana DataOra e deja
                    convertita din epoch), pentru TOATE casele (validare
                    facuta doar la nivel de status, nu s-a filtrat inca)
      status_case - DataFrame cu o linie per casa: HouseIDREF, Status
                    ("Valid"/"Invalid"), Motiv etc. (vezi marcheaza_case
                    din MAIN_corectat.py)
    """
    # --- Pasul 1: citim datele din baza IRISE -----------------------------
    df_consum, df_aparate, df_contor_general = incarca_date()

    # --- Pasul 2a: timpul din epoch in data/ora normala -------------------
    df_consum = converteste_timpul(df_consum)

    # --- Pasul 2b: marcam valorile suspecte (fara coloana State) ----------
    df_consum = marcheaza_valorile_suspecte(df_consum)

    # --- Pasul 2c: care case au un an complet de date ----------------------
    status_case, raport_lunar = marcheaza_case(df_consum)

    return df_consum, status_case


# =========================================================================
# PASUL 1b: alegem casa de analizat (fie ceruta explicit, fie prima valida)
# =========================================================================
def selecteaza_casa(status_case, id_casa=None):
    """
    Primeste tabelul de status (rezultat din marcheaza_case) si alege o
    casa de analizat:

      - id_casa=None (implicit)       -> prima casa cu Status == "Valid"
                                          (comportamentul de dinainte)
      - id_casa=<un numar>, ex 2000904 -> ACEA casa, DAR numai daca e in
                                          lista caselor valide - altfel
                                          ridicam o eroare clara (nu
                                          continuam pe ascuns cu o casa cu
                                          date incomplete/lipsa).

    Ridica o eroare clara si daca nu exista NICIO casa valida in toata
    baza (indiferent de id_casa), ca sa nu continuam din greseala mai
    departe in BATEM.
    """
    case_valide = status_case.loc[status_case["Status"] == "Valid"]
    if case_valide.empty:
        raise ValueError(
            "Nicio casa nu are Status == 'Valid'. Nu putem alege o casa "
            "pentru modelarea PV."
        )

    lista_valide = sorted(int(h) for h in case_valide["HouseIDREF"])

    if id_casa is None:
        return int(case_valide.iloc[0]["HouseIDREF"])

    id_casa = int(id_casa)
    if id_casa not in lista_valide:
        if id_casa in set(int(h) for h in status_case["HouseIDREF"]):
            motiv = status_case.loc[
                status_case["HouseIDREF"] == id_casa, "Motiv"
            ].iloc[0]
            raise ValueError(
                f"Casa {id_casa} exista in baza, dar NU e valida (Status "
                f"'Invalid', motiv: {motiv}). Case valide disponibile: "
                f"{lista_valide}"
            )
        raise ValueError(
            f"Casa {id_casa} nu exista in baza de date IRISE. Case valide "
            f"disponibile: {lista_valide}"
        )
    return id_casa


# =========================================================================
# PASUL 1c: extragem si formatam intervalul de timp al casei alese
# =========================================================================
def extrage_interval_casa(df_consum, id_casa):
    """
    Pentru casa `id_casa`, cauta in df_consum (coloana DataOra) prima si
    ultima data pentru care exista o masuratoare, si le intoarce ca text
    in formatul "DD/MM/YYYY" (asa cum are nevoie BATEM).

    Returneaza un tuplu (data_inceput_str, data_sfarsit_str).
    """
    date_casa = df_consum.loc[df_consum["HouseIDREF"] == id_casa, "DataOra"]
    if date_casa.empty:
        raise ValueError(
            f"Nu exista nicio masuratoare pentru casa {id_casa} in df_consum."
        )

    data_inceput = date_casa.min()
    data_sfarsit = date_casa.max()

    return _formateaza_data(data_inceput), _formateaza_data(data_sfarsit)


def _formateaza_data(data_ora):
    """Formateaza un timestamp (cu fus orar) ca text 'DD/MM/YYYY', pentru BATEM."""
    return data_ora.strftime("%d/%m/%Y")


# =========================================================================
# PIPELINE PASUL 1: leaga tot ce e mai sus, cap la coada
# =========================================================================
def sincronizeaza_interval_cu_irise(id_casa=None):
    """
    Ruleaza tot Pasul 1 al etapei de modelare PV:
      1. reia pipeline-ul de baza pana la marcarea caselor (Pasul 2c)
      2. alege casa de analizat - fie id_casa (daca e dat si e valid), fie
         prima casa valida (daca id_casa=None)
      3. extrage si formateaza data de inceput / sfarsit pentru casa aleasa

    PARAMETRU:
      id_casa - None (implicit) = prima casa cu Status == "Valid";
                sau un ID concret (ex. 2000904) = ACEA casa, daca e valida
                (altfel ridica ValueError cu lista caselor valide).

    Returneaza un dictionar cu tot ce va fi nevoie in pasii urmatori
    (modelul meteo si cel fotovoltaic din BATEM):
      {
        "id_casa": ...,
        "data_inceput": "DD/MM/YYYY",
        "data_sfarsit": "DD/MM/YYYY",
        "df_consum": DataFrame (util mai departe, ca sa nu se re-citeasca DB),
        "status_case": DataFrame,
      }
    """
    df_consum, status_case = ruleaza_pana_la_marcarea_caselor()
    id_casa_ales = selecteaza_casa(status_case, id_casa=id_casa)
    data_inceput, data_sfarsit = extrage_interval_casa(df_consum, id_casa_ales)

    mod_alegere = (
        "prima casa cu Status == 'Valid'" if id_casa is None
        else f"casa ceruta explicit (id_casa={id_casa})"
    )

    print("\n=== MODELARE PV - PASUL 1: sincronizarea intervalului cu IRISE ===")
    print(f"Casa aleasa ({mod_alegere}): {id_casa_ales}")
    print(f"Data de inceput (DD/MM/YYYY):              {data_inceput}")
    print(f"Data de sfarsit (DD/MM/YYYY):               {data_sfarsit}")

    return {
        "id_casa": id_casa_ales,
        "data_inceput": data_inceput,
        "data_sfarsit": data_sfarsit,
        "df_consum": df_consum,
        "status_case": status_case,
    }


# =========================================================================
# PASUL 2a: construim seria meteo, cu BATEM, pe intervalul casei alese
# =========================================================================
def construieste_date_meteo(
    data_inceput,
    data_sfarsit,
    location="Grenoble",
    latitude_north_deg=45.19154994547585,
    longitude_east_deg=5.722065312331381,
):
    """
    Construieste obiectul SiteWeatherData al BATEM pentru Grenoble, pe
    intervalul [data_inceput, data_sfarsit] (format "DD/MM/YYYY").

    BATEM cauta datele in data/<location>.json (adica data/Grenoble.json,
    vezi Pasul 0 + setup.ini). Daca fisierul local acopera deja perioada
    ceruta, NU se face niciun apel de retea - se citeste doar local.

    Returneaza obiectul weather_data (SiteWeatherData), din care se pot
    extrage seriile individuale cu weather_data.get('nume_variabila').
    """
    # Import facut aici (nu la nivel de modul) ca sa dam un mesaj de eroare
    # clar daca cineva ruleaza scriptul fara sa aiba instalat BATEM.
    from batem.core.weather import SWDbuilder

    builder = SWDbuilder(
        location=location,
        latitude_north_deg=latitude_north_deg,
        longitude_east_deg=longitude_east_deg,
    )
    weather_data = builder(
        from_stringdate=data_inceput,
        to_stringdate=data_sfarsit,
    )
    return weather_data


# =========================================================================
# PASUL 2b: grafic Plotly propriu: temperatura / umiditate / nori / vant / precipitatii
# =========================================================================
def genereaza_grafic_meteo(weather_data, nume_fisier="grafic_meteo_plotly.html"):
    """
    Deseneaza temperatura, umiditatea relativa, acoperirea de nori, viteza
    vantului si precipitatiile pe acelasi interval de timp ca si datele de
    consum - dar cu Plotly, construit de mana, NU cu weather_data.plot()
    din BATEM.

    DE CE NU weather_data.plot():
      Pe o masina cu Tkinter instalat (cazul obisnuit intr-un mediu local,
      ex. VS Code pe Windows), acea functie deschide o fereastra grafica
      interactiva si BLOCHEAZA scriptul pana o inchizi manual pe ecran.
      Intr-un pipeline care trebuie sa mearga singur pana la capat (Pasul 3
      si mai departe), asta e inacceptabil. Functia de-aici nu deschide
      nicio fereastra - scrie direct un fisier HTML si rularea continua.

    CELE CINCI VARIABILE AU SCARI FOARTE DIFERITE
      (temperatura: -10..35°C, umiditate: 0..100%, nori: 0..100%, vant: de
      obicei 0..60 km/h, precipitatii: de obicei 0..10 mm/h, cu varfuri
      rare) - de aceea le punem pe CINCI randuri separate, cu aceeasi axa
      de timp (subiect comun), in loc sa le suprapunem pe o singura axa,
      unde precipitatiile ar disparea turtite la baza graficului.

    Umiditatea si viteza vantului vin direct din datele meteo brute
    (Grenoble.json, campurile "relative_humidity_2m" si "wind_speed_10m"),
    pe care BATEM le expune sub numele "humidity" (%) si "wind_speed_km_h"
    (km/h) - vezi DEFAULT_FROM_TO_NAMING din batem.core.weather.

    Salveaza graficul ca fisier HTML de sine statator (se deschide direct
    in browser, la dublu-click, fara sa mai fie nevoie de Python).
    """
    datetimes = weather_data.datetimes
    temperatura = weather_data.get("temperature")
    umiditate = weather_data.get("humidity")
    nori = weather_data.get("cloudiness")
    vant = weather_data.get("wind_speed_km_h")
    precipitatii = weather_data.get("precipitation")

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        subplot_titles=(
            "Temperatura (°C)", "Umiditate relativa (%)", "Acoperire nori (%)",
            "Viteza vantului (km/h)", "Precipitatii (mm)",
        ),
    )

    fig.add_trace(
        go.Scatter(x=datetimes, y=temperatura, name="Temperatura (°C)",
                    line=dict(color="#d62728")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=datetimes, y=umiditate, name="Umiditate relativa (%)",
                    line=dict(color="#2ca02c")),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=datetimes, y=nori, name="Acoperire nori (%)",
                    line=dict(color="#7f7f7f")),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=datetimes, y=vant, name="Viteza vantului (km/h)",
                    line=dict(color="#9467bd")),
        row=4, col=1,
    )
    fig.add_trace(
        go.Bar(x=datetimes, y=precipitatii, name="Precipitatii (mm)",
               marker=dict(color="#1f77b4")),
        row=5, col=1,
    )

    fig.update_layout(
        title="Date meteo Grenoble: temperatura, umiditate, nori, vant, precipitatii",
        hovermode="x unified",
        showlegend=False,
        height=1200,
    )
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="km/h", row=4, col=1)
    fig.update_yaxes(title_text="mm", row=5, col=1)
    fig.update_xaxes(title_text="Data si ora", row=5, col=1)

    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"\nGrafic meteo (Plotly) salvat ca: {nume_fisier} "
          f"(fara fereastra grafica, fara blocaj)")
    return fig


# =========================================================================
# PASUL 3a: modelul solar si radiatia pe suprafata orizontala
# =========================================================================
def construieste_model_solar(weather_data):
    """Initializeaza SolarModel-ul BATEM pornind de la seria meteo obtinuta la Pasul 2."""
    from batem.core.solar import SolarModel

    return SolarModel(weather_data)


def radiatie_orizontala(solar_model):
    """
    Radiatia (W) pe o suprafata de 1 m^2, orientata orizontal spre cer
    (exposure_deg=0, slope_deg=180 - conventia BATEM pentru "orizontala
    in sus"). E o referinta, nu orientarea reala a panoului.
    """
    return solar_model.irradiances_W(exposure_deg=0, slope_deg=180, mask=None)


# =========================================================================
# PASUL 3b: orientarea optima a panoului fotovoltaic
# =========================================================================
def orientare_optima(solar_model, initial_slope_deg=150, initial_exposure_deg=0):
    """
    Cauta (pornind de la valorile initiale date) unghiurile de expunere
    (fata de sud) si de inclinare care maximizeaza radiatia solara anuala
    primita de panou.

    Returneaza (best_exposure, best_slope), amandoua in grade.
    """
    best_exposure, best_slope = solar_model.best_direction(
        initial_slope_deg=initial_slope_deg,
        initial_exposure_deg=initial_exposure_deg,
        mask=None,
    )
    print("\n=== PASUL 3: orientarea optima a panoului fotovoltaic ===")
    print(f"Expunere optima (fata de sud, grade):  {best_exposure}")
    print(f"Inclinare optima (fata de orizontala, grade): {best_slope}")
    return best_exposure, best_slope


def radiatie_optima(solar_model, best_exposure, best_slope):
    """Radiatia (W) pe suprafata de 1 m^2, la orientarea optima gasita mai sus."""
    return solar_model.irradiances_W(best_exposure, best_slope, mask=None)


# =========================================================================
# PASUL 3c: graficul comparativ radiatie / temperatura / nori
# =========================================================================
def genereaza_grafic_comparativ(
    solar_model,
    weather_data,
    irradiance_optimal,
    nume_fisier="grafic_comparatie_radiatie.html",
):
    """
    Grafic interactiv (plotly.graph_objects) cu axa de timp comuna
    (solar_model.datetimes) si doua axe Y:
      - axa Y primara:   radiatia solara optima (W/m^2)
      - axa Y secundara: temperatura (°C) SI acoperirea de nori (%),
                          suprapuse, ca sa se vada usor legatura dintre ele

    Salveaza graficul ca fisier HTML de sine statator (functioneaza si
    fara server, se deschide direct in browser).
    """
    temperatura = weather_data.get("temperature")
    nori = weather_data.get("cloudiness")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=solar_model.datetimes, y=irradiance_optimal,
            name="Radiatie solara optima (W/m²)",
            line=dict(color="#e6a800"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=solar_model.datetimes, y=temperatura,
            name="Temperatura (°C)",
            line=dict(color="#d62728"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=solar_model.datetimes, y=nori,
            name="Acoperire nori (%)",
            line=dict(color="#7f7f7f"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Radiatie solara optima vs. temperatura si acoperirea de nori (Grenoble)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Data si ora")
    fig.update_yaxes(title_text="Radiatie solara (W/m²)", secondary_y=False)
    fig.update_yaxes(title_text="Temperatura (°C) / Acoperire nori (%)", secondary_y=True)

    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"\nGrafic comparativ salvat ca: {nume_fisier}")
    return fig


# =========================================================================
# PASUL 3d: concluzia ceruta de profesor, calculata din datele reale
# =========================================================================
def analizeaza_relatia_radiatie_temperatura_nori(solar_model, weather_data, irradiance_optimal):
    """
    Nu doar afirmam relatia radiatie / temperatura / nori "din auzite" -
    o VERIFICAM pe datele reale ale intervalului analizat, cu doua teste
    simple, si printam concluzia bazata pe cifre.

    1. Corelatia radiatie <-> nori: ar trebui sa fie negativa (mai multi
       nori => mai putina radiatie ajunsa la sol).
    2. Diferenta clara intre "zile senine" (nori <= 20%) si "zile innorate"
       (nori >= 70%): media radiatiei ar trebui sa fie mult mai mica in a
       doua categorie.
    3. Corelatia radiatie <-> temperatura, calculata pe tot intervalul -
       de obicei pozitiva, dar in mare parte un efect SEZONIER (vara sunt
       si mai multe ore de soare, si temperaturi mai mari), nu neaparat un
       decalaj fin, orar, de inertie termica.
    """
    irr = np.asarray(irradiance_optimal, dtype=float)
    temp = np.asarray(weather_data.get("temperature"), dtype=float)
    nori = np.asarray(weather_data.get("cloudiness"), dtype=float)

    valide = ~(np.isnan(irr) | np.isnan(temp) | np.isnan(nori))
    corr_radiatie_nori = float(np.corrcoef(irr[valide], nori[valide])[0, 1])
    corr_radiatie_temp = float(np.corrcoef(irr[valide], temp[valide])[0, 1])

    zile_senine = nori <= 20.0
    zile_innorate = nori >= 70.0
    medie_senin = float(np.nanmean(irr[zile_senine])) if zile_senine.any() else float("nan")
    medie_innorat = float(np.nanmean(irr[zile_innorate])) if zile_innorate.any() else float("nan")
    raport = medie_senin / medie_innorat if medie_innorat else float("nan")

    print("\n=== PASUL 3: relatia radiatie / temperatura / nori (pe date reale) ===")
    print(f"Corelatie radiatie <-> acoperire nori: {corr_radiatie_nori:+.2f} "
          f"(negativa = mai multi nori, mai putina radiatie - cum era de asteptat)")
    print(f"Corelatie radiatie <-> temperatura:    {corr_radiatie_temp:+.2f} "
          f"(pozitiva, dar mai ales pe seama sezonului: vara vin la pachet "
          f"radiatie mare SI temperaturi mari)")
    print(f"Radiatie medie in momente senine (nori <= 20%):  {medie_senin:.0f} W/m²")
    print(f"Radiatie medie in momente innorate (nori >= 70%): {medie_innorat:.0f} W/m²")
    print(f"Raport senin / innorat: de {raport:.1f}x mai multa radiatie cand cerul e senin")

    print(
        "\nCONCLUZIE: pe datele analizate, norii au clar efectul cel mai "
        "puternic si mai direct asupra radiatiei - corelatia este negativa "
        f"({corr_radiatie_nori:+.2f}) iar in momentele senine radiatia optima "
        f"este de circa {raport:.1f}x mai mare decat in cele innorate: "
        f"({medie_senin:.0f} vs. {medie_innorat:.0f} W/m²). Legatura cu "
        "temperatura exista si ea (corelatie pozitiva, "
        f"{corr_radiatie_temp:+.2f}), dar e mai degraba un efect SEZONIER "
        "(iarna: putina radiatie si frig; vara: multa radiatie si cald) "
        "decat un decalaj fin, de la o ora la alta - un semn de inertie "
        "termica reala ar insemna ca varful de temperatura vine la cateva "
        "ore DUPA varful de radiatie in aceeasi zi, nu doar 'in aceeasi "
        "perioada a anului'; la rezolutia orara a acestor date, acel "
        "decalaj fin nu iese suficient de clar ca sa fie afirmat cu "
        "incredere, spre deosebire de efectul norilor, care e evident."
    )


# =========================================================================
# PASUL 4a: construim centrala fotovoltaica (PVplant), la orientarea optima
# =========================================================================
def construieste_centrala_pv(
    solar_model,
    best_exposure,
    best_slope,
    number_of_panels=PANOURI_O_CASA,
    pv_efficiency=PARAMETRI_PANOU["pv_efficiency"],
    panel_width_m=PARAMETRI_PANOU["panel_width_m"],
    panel_height_m=PARAMETRI_PANOU["panel_height_m"],
    distance_between_arrays_m=PARAMETRI_PANOU["distance_between_arrays_m"],
):
    """
    Construieste o centrala fotovoltaica virtuala (PVplant), montata pe
    acoperis/plan (MOUNT_TYPES.PLAN), la orientarea optima gasita la
    Pasul 3 (best_exposure, best_slope) - NU orientarea "din carte", ca sa
    profitam maximal de conditiile solare reale din Grenoble.

    PARAMETRII PANOULUI - specificatiile REALE ale unui panou de pe piata,
    Trina Solar Vertex S 425W (nu cifre rotunde inventate):
      number_of_panels             - CATE panouri punem pe acoperis (dat
                                     direct de noi, nu calculat din puterea
                                     de varf dorita - vezi mai jos)
      pv_efficiency                - randamentul panoului (21.3%)
      panel_width_m, panel_height_m - dimensiunile unui panou (1.13 x 1.76 m)
      distance_between_arrays_m    - distanta intre siruri de panouri (3 m),
                                     ca sa nu se umbreasca intre ele

    VERIFICARE RAPIDA a numelui panoului (425W):
      0.213 (randament) x (1.13 x 1.76) m² (suprafata) x 1000 W/m²
      (iradianta standard de test, STC) = ~423.6 W ≈ 425W - se potriveste
      cu fisa tehnica reala a panoului Trina Solar Vertex S 425W.

    DE CE number_of_panels SI NU peak_power_kW:
      PVplant din BATEM accepta ORICARE dintre cei doi parametri (vezi
      batem.core.solar, clasa PVplant.__init__): fie ii dai peak_power_kW
      si BATEM calculeaza singur un numar intreg de panouri (rotunjind),
      fie ii dai direct number_of_panels si atunci puterea de varf
      (pv_system.peak_power_kW) e cea care rezulta, nu invers. Aici alegem
      a doua varianta - dam noi numarul de panouri (ex. cate incap pe
      acoperisul unei case, sau cate vrem sa punem pentru o comunitate
      energetica), iar puterea de varf e un REZULTAT al acestei alegeri,
      nu o tinta impusa de la inceput. Formula BATEM (vezi PVplant):
        peak_power_kW = number_of_panels * panel_width_m * panel_height_m
                         * pv_efficiency
      Cu valorile de mai sus, 4 panouri => ~1.69 kWp (afisat mai jos, la
      Pasul 4b).

    DE UNDE VINE VALOAREA IMPLICITA A LUI number_of_panels (4):
      Nu e un numar ales la intamplare: e COTA-PARTE a unei case din
      centrala dimensionata "de mana" pentru comunitate la Etapa 2
      (pv_comunitate.py): 40 de panouri pentru 9 case => round(40/9) = 4
      panouri pentru o casa. Vezi panouri_pentru() in config_proiect.py.

    Returneaza obiectul pv_system (PVplant).
    """
    from batem.core.solar import PVplant, MOUNT_TYPES

    pv_system = PVplant(
        solar_model=solar_model,
        exposure_deg=best_exposure,
        slope_deg=best_slope,
        mount_type=MOUNT_TYPES.PLAN,
        number_of_panels=number_of_panels,
        panel_width_m=panel_width_m,
        panel_height_m=panel_height_m,
        pv_efficiency=pv_efficiency,
        distance_between_arrays_m=distance_between_arrays_m,
    )
    return pv_system


# =========================================================================
# PASUL 4b: productia orara a centralei + rezumat
# =========================================================================
def calculeaza_productie_pv(pv_system):
    """
    Puterea orara produsa de intreaga centrala (toate panourile la un loc),
    in W. Deoarece seria meteo/solara e la rezolutie ORARA, valoarea in W
    a unei ore e numeric egala cu energia produsa in ora aceea, in Wh -
    nu mai trebuie inmultit cu nimic ca sa trecem de la putere la energie.
    """
    return pv_system.powers_W()


def afiseaza_rezumat_pv(
    pv_system,
    pv_power,
    nume_panou=NUME_PANOU,
):
    """
    Printeaza panoul folosit, numarul de panouri (dat de noi, la Pasul 4a),
    puterea de varf REZULTATA din acel numar de panouri, si productia
    totala de energie pe intervalul analizat (in MWh), plus o estimare
    anualizata (scalata la 365 de zile), utila pentru ca intervalul nostru
    (22/01/1998 -> 04/02/1999) e putin mai lung decat un an calendaristic.
    """
    productie_totala_Wh = sum(pv_power)
    productie_totala_MWh = productie_totala_Wh / 1_000_000

    zile_interval = (pv_system.datetimes[-1] - pv_system.datetimes[0]).total_seconds() / 86400
    productie_anualizata_MWh = productie_totala_MWh * 365 / zile_interval

    print("\n=== PASUL 4: centrala fotovoltaica (PVplant) ===")
    print(f"Panou utilizat: {nume_panou} "
          f"({pv_system.panel_width_m:.2f}m x {pv_system.panel_height_m:.2f}m, "
          f"Eficienta: {100 * pv_system.pv_efficiency:.1f}%)")
    print(f"Numar de panouri (dat direct de noi): {pv_system.number_of_panels}")
    print(f"Suprafata totala PV:                 {pv_system.surface_pv_m2:.1f} m²")
    print(f"Putere de varf rezultata (din numarul de panouri): "
          f"{pv_system.peak_power_kW:.2f} kWp")
    print(f"Productie totala pe interval ({zile_interval:.0f} zile): "
          f"{productie_totala_MWh:.2f} MWh")
    print(f"Productie anualizata (scalata la 365 zile):          "
          f"{productie_anualizata_MWh:.2f} MWh/an")


# =========================================================================
# PASUL 4c: PROFILUL DE PRODUCTIE pe un an, SINGUR (fara consum peste el)
# =========================================================================
def genereaza_grafic_profil_productie(
    solar_model,
    pv_power,
    pv_system=None,
    nume_fisier="grafic_profil_productie_pv.html",
):
    """
    Reprezinta grafic PROFILUL DE PRODUCTIE al centralei pe tot intervalul
    analizat (~un an), SINGUR - fara consumul casei suprapus peste el.

    DE CE UN GRAFIC SEPARAT, cand exista deja graficul productie vs. consum
    de la Pasul 5b:
      Acolo, productia e desenata ca sa fie COMPARATA cu consumul, iar axa Y
      e impartita intre doua serii cu ordine de marime diferite - forma
      profilului de productie in sine (cat de mult creste vara, cat de mult
      scade iarna) se citeste greu. Cerinta temei era insa explicit sa
      reprezentam profilul de productie in timp, pe un an de zile, ca atare.
      De-aia aici productia e singura pe grafic.

    Graficul are DOUA PANOURI, cu aceeasi informatie la doua rezolutii:
      1. PROFILUL ORAR - toate orele anului, una langa alta. Se vede
         "anvelopa" productiei: fiecare zi e un varf (zi/noapte), iar
         inaltimea varfurilor creste spre vara si scade spre iarna. E
         graficul detaliat, dar dens.
      2. TOTALURILE LUNARE (kWh/luna) - aceeasi productie, adunata pe luni.
         Elimina zgomotul zilnic si arata curat sezonalitatea: cat produce
         centrala intr-o luna de vara fata de una de iarna. Bare colorate
         diferit pentru lunile de iarna (Dec-Ian-Feb) si de vara
         (Iun-Iul-Aug), ca sa nu fie nevoie sa numeri lunile pe axa X.

    Printeaza si raportul vara / iarna al productiei lunare medii - cifra
    care justifica, mai tarziu, de ce dimensionarea centralei trebuie sa
    tina cont de amandoua anotimpurile, nu doar de vara.
    """
    luni_iarna, luni_vara = {12, 1, 2}, {6, 7, 8}

    serie = pd.Series(
        np.asarray(pv_power, dtype=float),
        index=pd.DatetimeIndex(pd.Series(solar_model.datetimes)),
    )
    lunar_kWh = serie.resample("MS").sum() / 1000

    etichete, culori = [], []
    for luna in lunar_kWh.index:
        if luna.month in luni_iarna:
            etichete.append(f"{luna.strftime('%b %Y')} (iarna)")
            culori.append("#3b7fc4")
        elif luna.month in luni_vara:
            etichete.append(f"{luna.strftime('%b %Y')} (vara)")
            culori.append("#e6a800")
        else:
            etichete.append(luna.strftime("%b %Y"))
            culori.append("#bdbdbd")

    titlu_centrala = ""
    if pv_system is not None:
        titlu_centrala = (
            f" - {pv_system.number_of_panels} panouri, "
            f"{pv_system.peak_power_kW:.2f} kWp"
        )

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.12,
        subplot_titles=(
            "Profilul orar al productiei (fiecare ora a anului)",
            "Totaluri lunare (kWh/luna) - albastru = iarna, galben = vara",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=serie.index, y=serie.values, name="Productie orara (Wh)",
            line=dict(color="#e6a800", width=1),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=etichete, y=lunar_kWh.values, name="Productie lunara (kWh)",
            marker_color=culori,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"Profilul de productie al centralei PV pe un an{titlu_centrala}",
        hovermode="x unified",
        showlegend=False,
        height=850,
    )
    fig.update_yaxes(title_text="Energie pe ora (Wh)", row=1, col=1)
    fig.update_yaxes(title_text="Energie pe luna (kWh)", row=2, col=1)
    fig.update_xaxes(title_text="Data si ora", row=1, col=1)
    fig.update_xaxes(title_text="Luna", row=2, col=1)

    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"\nGrafic profil de productie salvat ca: {nume_fisier}")

    medie_iarna = lunar_kWh[[l.month in luni_iarna for l in lunar_kWh.index]].mean()
    medie_vara = lunar_kWh[[l.month in luni_vara for l in lunar_kWh.index]].mean()
    if medie_iarna and not np.isnan(medie_iarna) and not np.isnan(medie_vara):
        print(
            f"  Productie lunara medie: iarna {medie_iarna:.0f} kWh/luna, "
            f"vara {medie_vara:.0f} kWh/luna "
            f"(de {medie_vara / medie_iarna:.1f}x mai mult vara decat iarna)"
        )
    return fig


# =========================================================================
# PASUL 5a: consumul orar al casei alese, din datele deja procesate
# =========================================================================
def pregateste_consum_orar_casa(df_consum, id_casa):
    """
    Calculeaza consumul orar al casei `id_casa`, folosind
    calculeaza_consum_orar din MAIN_corectat.py, aplicata pe df_consum
    asa cum a fost deja procesat la Pasul 1 (Pasii 1, 2a, 2b din
    MAIN_corectat.py - date convertite in timp local si cu valorile
    suspecte marcate ca NaN) - PLUS, chiar aici, Pasul 2d si Pasul 3 din
    MAIN_corectat.py, care lipseau inainte.

    CORECTIE FATA DE VERSIUNEA ANTERIOARA - de ce erau necesare 2d si 3:
      Fara Pasul 2d (completeaza_golurile), golurile marcate NaN la 2b
      raman goluri; calculeaza_consum_orar arunca doar orele cu PREA
      PUTINE masuratori (sub 4 din 6) - o ora cu 4-5 masuratori dintr-un
      gol PARTIAL trece testul, dar suma ei tot iese subestimata, si mai
      grav, factorul de rescalare (x6/nr_masuratori) din
      calculeaza_consum_orar poate AMPLIFICA o singura masuratoare
      accidental mare pana la o valoare aberanta - exact spike-ul care a
      deformat axa Y a graficului anterior. Fara Pasul 3
      (filtreaza_valorile_exagerate), orice valoare individuala ramasa
      absurd de mare (peste limita fizica a aparatului, sau peste
      consumul lui normal de functionare) trece nefiltrata in suma orara.
      Aplicam acum amandoua, in ACEEASI ordine ca in main() din
      MAIN_corectat.py (2d inainte de 3), pe datele DOAR ale casei alese
      (nu pe toate cele 14 case, ca sa nu recalculam degeaba).

    Returneaza consum_orar_casa: DataFrame cu coloanele HouseIDREF, Ora,
    Consum_Orar_Casa_Wh (vezi calculeaza_consum_orar in MAIN_corectat.py).
    """
    df_casa = df_consum.loc[df_consum["HouseIDREF"] == id_casa].copy()
    if df_casa.empty:
        raise ValueError(f"Nu exista masuratori pentru casa {id_casa}.")

    # --- Pasul 2d: completam golurile ramase (media vecinilor / ora din zi)
    df_casa = completeaza_golurile(df_casa)

    # --- Pasul 3: taiem valorile individuale exagerate care au scapat de 2b
    df_casa = filtreaza_valorile_exagerate(df_casa)

    _, consum_orar_casa = calculeaza_consum_orar(df_casa)
    return consum_orar_casa


# =========================================================================
# PASUL 5b: graficul productie PV vs. consum casa
# =========================================================================
def genereaza_grafic_productie_vs_consum(
    solar_model,
    pv_power,
    consum_orar_casa,
    id_casa,
    nume_fisier=None,
):
    """
    Suprapune, pe aceeasi axa de timp, productia orara a centralei PV
    (Wh, de la Pasul 4) si consumul orar al casei alese (Wh, de la
    Pasul 5a). Cele doua serii vin din surse diferite (model solar, resp.
    masuratori reale de consum) si pot avea capete usor diferite - Plotly
    le deseneaza fiecare pe axa ei temporala, fara sa fie nevoie sa
    coincida exact ca lungime.

    NUMELE FISIERULUI include ID-ul casei (ex: grafic_productie_vs_consum_
    2000904.html) - daca nu se da nume_fisier explicit, se genereaza automat
    din id_casa. Asa nu se mai suprascrie graficul unei case cand analizam
    alta casa (inainte, numele era fix, "grafic_productie_vs_consum.html",
    identic indiferent de casa aleasa).

    Salveaza graficul ca fisier HTML de sine statator.
    """
    if nume_fisier is None:
        nume_fisier = f"grafic_productie_vs_consum_{id_casa}.html"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=solar_model.datetimes, y=pv_power,
        name=f"Productie PV ({len(pv_power)} ore, W≈Wh)",
        line=dict(color="#e6a800"),
    ))
    fig.add_trace(go.Scatter(
        x=consum_orar_casa["Ora"], y=consum_orar_casa["Consum_Orar_Casa_Wh"],
        name=f"Consum casa {id_casa} (Wh)",
        line=dict(color="#1f77b4"),
    ))

    fig.update_layout(
        title=f"Productia centralei PV vs. consumul orar al casei {id_casa}",
        hovermode="x unified",
        xaxis_title="Data si ora",
        yaxis_title="Energie pe ora (Wh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.write_html(nume_fisier, include_plotlyjs="cdn")
    print(f"\nGrafic productie vs. consum salvat ca: {nume_fisier}")
    return fig


# =========================================================================
# PASUL 5c: rezumatul numeric - productie, consum, grad de acoperire
# =========================================================================
def afiseaza_rezumat_productie_vs_consum(pv_power, consum_orar_casa, id_casa, pv_system=None):
    """
    Printeaza productia totala PV, consumul total al casei si gradul de
    acoperire (productie / consum * 100%), pe intervalele de timp
    disponibile pentru fiecare serie (nu sunt neaparat identice ca lungime -
    vezi observatia din genereaza_grafic_productie_vs_consum).
    """
    productie_totala_kWh = sum(pv_power) / 1000
    consum_total_kWh = consum_orar_casa["Consum_Orar_Casa_Wh"].sum() / 1000
    grad_acoperire = 100 * productie_totala_kWh / consum_total_kWh

    print(f"\n=== PASUL 5: productie PV vs. consum casa {id_casa} ===")
    print(f"Productie totala PV:   {productie_totala_kWh / 1000:.2f} MWh "
          f"({productie_totala_kWh:.0f} kWh)")
    print(f"Consum total casa:     {consum_total_kWh:.0f} kWh")
    print(f"Grad de acoperire (productie / consum): {grad_acoperire:.0f}%")

    putere_reala_kWp = pv_system.peak_power_kW if pv_system is not None else None
    if grad_acoperire > 100:
        detaliu_putere = (
            f" (centrala de {putere_reala_kWp:.2f} kWp)" if putere_reala_kWp else ""
        )
        print(
            f"  => Productia PV{detaliu_putere} depaseste consumul casei - "
            "surplusul ar putea fi injectat in retea sau stocat intr-o "
            "baterie; o centrala mai mica ar acoperi consumul fara surplus."
        )
    else:
        detaliu_putere = (
            f" (centrala de {putere_reala_kWp:.2f} kWp)" if putere_reala_kWp else ""
        )
        print(
            f"  => Productia PV{detaliu_putere} NU acopera tot consumul casei - "
            "diferenta trebuie luata din retea (sau dintr-o baterie, daca ar exista)."
        )


# =========================================================================
# ZONA DE COMENZI
# =========================================================================
if __name__ == "__main__":
    # Casa analizata vine acum din config_proiect.py (CASA_IMPLICITA), NU
    # scrisa de mana aici - ca sa fie garantat ACEEASI casa si in
    # simulare_multiagent.py, altfel cifrele celor doua etape nu se pot
    # compara intre ele. Daca vrei alta casa, schimbi CASA_IMPLICITA in
    # config_proiect.py si se aliniaza toate scripturile deodata.
    # Daca pui acolo un ID care nu exista sau care nu e valid (Status !=
    # "Valid"), scriptul se opreste cu o eroare care iti arata lista completa
    # a caselor valide disponibile - nu trebuie sa o stii dinainte.
    ID_CASA_DORITA = CASA_IMPLICITA

    # --- Pasul 1: sincronizam intervalul de timp cu baza IRISE -----------
    rezultate_pv = sincronizeaza_interval_cu_irise(id_casa=ID_CASA_DORITA)

    # --- Pasul 2: date meteo (BATEM + Grenoble.json), pe acelasi interval -
    weather_data = construieste_date_meteo(
        data_inceput=rezultate_pv["data_inceput"],
        data_sfarsit=rezultate_pv["data_sfarsit"],
    )
    genereaza_grafic_meteo(weather_data)

    # --- Pasul 3: modelul de radiatie solara -------------------------------
    solar_model = construieste_model_solar(weather_data)

    irradiance_horizontal = radiatie_orizontala(solar_model)

    best_exposure, best_slope = orientare_optima(
        solar_model, initial_slope_deg=150, initial_exposure_deg=0,
    )
    irradiance_optimal = radiatie_optima(solar_model, best_exposure, best_slope)

    genereaza_grafic_comparativ(solar_model, weather_data, irradiance_optimal)
    analizeaza_relatia_radiatie_temperatura_nori(
        solar_model, weather_data, irradiance_optimal,
    )

    # --- Pasul 4: centrala fotovoltaica (PVplant), la orientarea optima ---
    # Panoul (Trina Solar Vertex S 425W) si numarul de panouri vin din
    # config_proiect.py: PANOURI_O_CASA = 4, adica cota-parte a unei case din
    # centrala de 40 de panouri dimensionata pentru cele 9 case la Etapa 2
    # (pv_comunitate.py). Asa, cifrele de aici si cele din simularea
    # multi-agent descriu EXACT aceeasi centrala.
    NUMBER_OF_PANELS = PANOURI_O_CASA
    pv_system = construieste_centrala_pv(
        solar_model, best_exposure, best_slope,
        number_of_panels=NUMBER_OF_PANELS, **PARAMETRI_PANOU,
    )
    pv_power = calculeaza_productie_pv(pv_system)
    afiseaza_rezumat_pv(pv_system, pv_power, nume_panou=NUME_PANOU)

    # --- Pasul 4c: profilul de productie pe un an, singur ------------------
    genereaza_grafic_profil_productie(solar_model, pv_power, pv_system=pv_system)

    # --- Pasul 5: productie PV vs. consumul casei alese la Pasul 1 --------
    consum_orar_casa = pregateste_consum_orar_casa(
        rezultate_pv["df_consum"], rezultate_pv["id_casa"],
    )
    genereaza_grafic_productie_vs_consum(
        solar_model, pv_power, consum_orar_casa, rezultate_pv["id_casa"],
    )
    afiseaza_rezumat_productie_vs_consum(
        pv_power, consum_orar_casa, rezultate_pv["id_casa"], pv_system=pv_system,
    )

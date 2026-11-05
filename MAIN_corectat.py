"""
=============================================================================
PROIECT: Analiza consumului de energie in locuinte (baza de date IRISE)
=============================================================================

CE CONTINE BAZA DE DATE
  - Tabelul House       : lista caselor monitorizate
  - Tabelul Appliance   : lista aparatelor din fiecare casa
  - Tabelul Consumption : cate o masuratoare la fiecare 10 minute, pentru
                          fiecare aparat, timp de aproximativ un an
                          (1998-1999, in Franta)

  Coloana "Value" din Consumption = ENERGIA consumata in acele 10 minute,
  masurata in Wh (wati-ora). NU e putere.
  Diferenta conteaza: energiile se ADUNA, puterile s-ar media. Verificat pe
  aparatele care au puterea scrisa in nume - de exemplu masina de spalat de
  3,25 kW are ca maxim 537 Wh/10min, iar 537 x 6 = 3222 W, adica exact cat
  scrie pe eticheta. Deci Value e energie pe interval de 10 minute.

  Coloana "State" NU se foloseste in acest fisier (cerinta explicita: nu ne
  bazam pe ea). Vezi mai jos, la PASUL 2b, ce inseamna asta in practica si
  ce limitare demonstrata aduce.


ORDINEA PASILOR (si de ce e importanta ordinea)

  PASUL 1  Citim datele din SQL si scoatem un singur "aparat" virtual:
           "Site consumption" (consumul intregii case, inregistrat ca si
           cum ar fi un aparat; daca l-am lasa inauntru, am numara consumul
           de doua ori). "Total site light consumption" NU se scoate - se
           trateaza ca un aparat normal.

  PASUL 2a Transformam timpul din format "epoch" in data si ora normala

  PASUL 2b MARCAM valorile suspecte, FARA sa folosim coloana State, si le
           punem pe "gol" (NaN). Folosim doua semnale statistice:
             (i)  pana generala: momente in care aproape toate aparatele
                  casei citesc simultan 0
             (ii) sir suspect la un singur aparat: o secventa de zerouri
                  mult mai lunga decat tot restul istoricului aparatului
           ATENTIE - limitare demonstrata: aceste doua semnale NU pot
           distinge sigur "senzorul a murit" de "aparatul chiar nu a fost
           folosit". Exemplu concret gasit in date: la casa 2000902,
           plita a stat "moarta" 181 zile la rand; la casa 2000910, masina
           de spalat a stat legitim nefolosita 330 de zile la rand - mai
           mult decat cazul stricat. Fara coloana State, cele doua situatii
           produc exact acelasi tipar in coloana Value, deci metoda de aici
           e cea mai buna aproximare posibila, nu o solutie sigura.

  PASUL 2c Verificam care case au un an complet de date si care nu.
           Casele carora le lipsesc luni intregi sunt scoase din analiza.

  PASUL 2d ABIA ACUM umplem golurile ramase, cu media valorilor vecine.
           E sigur sa facem asta aici: casele cu goluri mari au fost deja
           eliminate la 2c, deci ce a mai ramas sunt goluri mici (minute,
           nu luni). Daca am fi umplut golurile inainte de 2c, am fi
           inventat 6 luni de date false pentru casa 2000902.

  PASUL 3  Cautam valori exagerat de mari care au scapat de filtrul 2b

  PASUL 4  Adunam masuratorile de la 10 minute in consum pe ora

  PASUL 5  Facem graficele interactive cu Plotly

=============================================================================
"""

import re
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px

DB_PATH = "irise38.sqlite3"

# La fiecare 10 minute avem o masuratoare, deci 6 masuratori intr-o ora
MASURATORI_PE_ORA = 6


# =========================================================================
# PASUL 1: Citim datele si scoatem totalul casei (nu si totalul luminilor)
# =========================================================================
def incarca_date(db_path=DB_PATH):
    """
    Citeste Appliance si Consumption si scoate din analiza un singur
    "aparat" care de fapt nu e aparat, ci un total deja calculat:

      - "Site consumption" = consumul INTREGII case

    Daca l-am lasa in lista de aparate, consumul lui s-ar aduna peste
    consumul aparatelor individuale si totalul casei ar iesi dublu.

    "Total site light consumption" (totalul becurilor) NU se scoate - se
    trateaza ca un aparat normal, la fel ca restul. Verificat in date: la
    majoritatea caselor, lampile cu senzor propriu (Non halogen lamp 1, 2,
    ...) reprezinta doar 1-43% din acest total, deci "Total site light
    consumption" prinde si iluminat fara senzor propriu - nu e o simpla
    dublura a lampilor individuale, ca "Site consumption" pentru toata
    casa. Poate exista un mic dublaj (pana la ~43% intr-o casa) la lampile
    care au si senzor individual SI intra in total; il acceptam.

    Consumul intregii case il pastram totusi separat (df_contor_general),
    ca sa avem cu ce compara la final: suma aparatelor nu are voie sa
    depaseasca ce a masurat contorul general al casei.

    ATENTIE:
      Coloana Appliance.ID nu e unica in toata baza. Fiecare casa isi
      numeroteaza aparatele de la 0. Deci exista un "aparat 3" in casa
      2000900 SI un "aparat 3" in casa 2000901, care sunt lucruri diferite.
      Ca sa identifici corect un aparat ai nevoie de perechea
      (ID, HouseIDREF). De asta toate JOIN-urile de mai jos se fac pe
      AMBELE coloane, nu doar pe ID.

      Coloana State NU e citita aici - nu o folosim nicaieri in pipeline.

      Tabelul House NU e citit in intregime - nici macar coloanele
      StartingEpochTime/EndingEpochTime. Intervalul fiecarei case il
      calculam direct din Consumption (sursa de adevar despre ce s-a
      masurat efectiv), nu din ce declara House ca ar trebui sa fie.
    """
    conn = sqlite3.connect(db_path)

    df_aparate_toate = pd.read_sql_query("SELECT * FROM Appliance", conn)
    # Nu ne trebuie continutul tabelului House (nici macar coloanele
    # StartingEpochTime/EndingEpochTime) - intervalul fiecarei case il
    # calculam direct din Consumption, care e sursa de adevar. Citim doar
    # un numar, ca sa afisam un rezumat, fara sa incarcam tot tabelul.
    nr_case_in_tabel = pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM House", conn
    )["n"].iloc[0]

    este_total = df_aparate_toate["Name"].str.contains(
        "Site consumption", case=False, na=False, regex=False,
    ) & ~df_aparate_toate["Name"].str.contains(
        "Total site light consumption", case=False, na=False, regex=False,
    )
    df_aparate = df_aparate_toate[~este_total].copy()

    # Filtram direct in SQL, ca sa nu incarcam in memorie randuri pe care
    # oricum le-am arunca imediat dupa aceea.
    df_consum = pd.read_sql_query(
        """
        SELECT c.ApplianceIDREF, c.HouseIDREF, c.EpochTime, c.Value, a.Name
        FROM Consumption c
        JOIN Appliance a
          ON a.ID = c.ApplianceIDREF AND a.HouseIDREF = c.HouseIDREF
        WHERE a.Name NOT LIKE '%Site consumption%'
           OR a.Name LIKE '%Total site light consumption%'
        """,
        conn,
    )

    df_contor_general = pd.read_sql_query(
        """
        SELECT c.HouseIDREF, c.EpochTime, c.Value
        FROM Consumption c
        JOIN Appliance a
          ON a.ID = c.ApplianceIDREF AND a.HouseIDREF = c.HouseIDREF
        WHERE a.Name LIKE '%Site consumption%'
          AND a.Name NOT LIKE '%Total site light consumption%'
        """,
        conn,
    )
    conn.close()

    print("=== PASUL 1: Citirea datelor ===")
    print(f"Case in tabelul House:              {nr_case_in_tabel}")
    print(f"Case care au efectiv masuratori:    "
          f"{df_consum['HouseIDREF'].nunique()}")
    print(f"Aparate: {len(df_aparate_toate)} -> {len(df_aparate)} "
          f"(am scos {este_total.sum()} 'aparat' care era de fapt totalul "
          f"casei; 'Total site light consumption' a fost pastrat)")
    print(f"Masuratori ramase:                  {len(df_consum)}")

    return df_consum, df_aparate, df_contor_general


# =========================================================================
# PASUL 2a: Timpul, din "epoch" in data si ora normala
# =========================================================================
def converteste_timpul(df, coloana="EpochTime", fus="Europe/Paris"):
    """

    "Epoch time" inseamna numarul de secunde scurse de la 1 ianuarie 1970.
    De exemplu 886000000 inseamna 28 februarie 1998. E un format comod
    pentru calculator, dar ilizibil pentru om.

    Convertim si in ora locala a Frantei (datele sunt franceze, din 1998-99),
    adica UTC+1 iarna si UTC+2 vara.

    DE CE CONTEAZA ORA LOCALA:
      Daca am lasa ora in UTC, varful de seara ar aparea cu o ora
      mai devreme iarna si cu doua vara. Mediate pe tot anul, cele doua
      varfuri s-ar suprapune prost si graficul ar iesi aplatizat.
    """
    df = df.copy()
    df["DataOra"] = (
        pd.to_datetime(df[coloana], unit="s", utc=True).dt.tz_convert(fus)
    )
    return df.sort_values(["HouseIDREF", "DataOra"]).reset_index(drop=True)


# =========================================================================
# PASUL 2b: Marcam valorile suspecte, FARA coloana State
# =========================================================================
def _limitele_sirurilor(valori_bool):
    """
    Ajutator de nivel jos, folosit in TREI locuri din fisierul asta care au
    nevoie sa gaseasca siruri de valori True consecutive intr-un vector
    (pana generala, sir suspect de zerouri, luni consecutive lipsa la
    marcheaza_case). Inainte era scris de 3 ori separat; acum e intr-un
    singur loc.

    Primeste un vector 0/1 sau True/False (in ordine cronologica) si
    intoarce doua array-uri paralele: indicii de INCEPUT si de SFARSIT
    (exclusiv) ai fiecarui sir de True.

    Exemplu: [0,1,1,0,1] -> start=[1,4], sfarsit=[3,5]
             (adica un sir de lungime 2, la pozitiile 1-2, si unul de
             lungime 1, la pozitia 4).
    """
    v = np.asarray(valori_bool).astype(int)
    limite = np.flatnonzero(np.diff(np.concatenate(([0], v, [0]))))
    return limite[::2], limite[1::2]  # start, sfarsit


def _lungimile_sirurilor(valori_bool):
    """La fel ca _limitele_sirurilor, dar intoarce direct lungimile."""
    start, sfarsit = _limitele_sirurilor(valori_bool)
    return sfarsit - start


def _cel_mai_lung_sir_boolean(valori_bool, prag_lungime):
    """
    Ajutator comun: primeste un vector de True/False (in ordine cronologica)
    si returneaza un vector la fel de lung, cu True doar in intervalele unde
    a existat un sir de True consecutive de lungime >= prag_lungime.

    Folosit de detectorul de "pana generala" de mai jos.
    """
    start, sfarsit = _limitele_sirurilor(valori_bool)
    rezultat = np.zeros(len(np.asarray(valori_bool)), dtype=bool)
    for s, e in zip(start, sfarsit):
        if e - s >= prag_lungime:
            rezultat[s:e] = True
    return rezultat


def masca_pana_generala(df, prag_procent_simultan=95.0, prag_lungime_zile=10):
    """
    Semnalul 1 din 2: PANA GENERALA a aparaturii de masura.

    Ideea: e statistic imposibil ca aproape TOATE aparatele unei case
    (frigider, congelator, lumini, masina de spalat...) sa citeasca exact 0
    in acelasi moment, din intamplare - fiecare are propriul tipar de
    functionare. Daca se intampla PENTRU MULT TIMP LA RAND, nu e "toata
    casa era goala o noapte", e aparatura de inregistrare care a picat.

    DE CE E NEVOIE SI DE DURATA (nu doar de procent):
      Testat initial doar cu procentul (fara conditia de durata): la ora
      3 noaptea, in multe case normale, majoritatea aparatelor sunt oprite
      (lumini, TV, masina de spalat) si doar frigiderul/congelatorul mai
      trag curent din cand in cand - asa ca un prag de procent, SINGUR,
      prindea 14% din toate masuratorile din toata baza, in aproape orice
      casa. Evident gresit: nopti normale, nu pane.
      Adaugand conditia de durata (implicit >= 10 zile la rand), raman
      marcate doar perioadele lungi, gen 2000900 (~25.6 zile la rand cu
      aproape toate aparatele pe 0), nu si liniste de o noapte.

    Returneaza o masca (True/False, aceeasi forma ca df) - NU modifica df.
    Se combina cu masca de la siruri suspecte inainte de aplicare, ca cele
    doua semnale sa nu se calce unul pe celalalt.
    """
    zero = (df["Value"] == 0)
    total_aparate = df.groupby(["HouseIDREF", "DataOra"])["Value"].transform("size")
    nr_zero = zero.groupby([df["HouseIDREF"], df["DataOra"]]).transform("sum")
    procent_zero_simultan = 100 * nr_zero / total_aparate
    suspect_moment = procent_zero_simultan >= prag_procent_simultan

    # Un moment (casa, DataOra) se repeta o data per aparat. Facem sirurile
    # o singura data pe lista unica de momente, cronologic, per casa - apoi
    # rezultatul se aplica automat la toate randurile-aparat de la momentul
    # respectiv (pentru ca suspect_moment e deja identic pe toate randurile
    # aceluiasi moment).
    unic = (
        pd.DataFrame({"HouseIDREF": df["HouseIDREF"], "DataOra": df["DataOra"],
                      "suspect": suspect_moment})
        .drop_duplicates(["HouseIDREF", "DataOra"])
        .sort_values(["HouseIDREF", "DataOra"])
    )
    prag_masuratori = prag_lungime_zile * 24 * MASURATORI_PE_ORA
    unic["pana"] = unic.groupby("HouseIDREF")["suspect"].transform(
        lambda s: _cel_mai_lung_sir_boolean(s.to_numpy(), prag_masuratori)
    )

    harta = unic.set_index(["HouseIDREF", "DataOra"])["pana"]
    masca = pd.Series(
        harta.reindex(
            pd.MultiIndex.from_arrays([df["HouseIDREF"], df["DataOra"]])
        ).to_numpy(),
        index=df.index,
    ).fillna(False)

    print("\n=== PASUL 2b (i): pana generala (fara State) ===")
    print(f"Momente cu >= {prag_procent_simultan:.0f}% din aparate simultan "
          f"pe 0, sustinut >= {prag_lungime_zile} zile la rand: marcate "
          f"{masca.sum()} masuratori ({100 * masca.mean():.2f}% din total)")
    return masca


def masca_siruri_suspecte(df, prag_lungime_zile=3, prag_raport=8.0):
    """
    Semnalul 2 din 2: SIR SUSPECT DE ZEROURI la un singur aparat.

    Ideea: pentru fiecare aparat, ne uitam la TOATE secventele lui de
    zerouri consecutive de-a lungul anului. Daca cea mai lunga secventa e
    mult mai lunga (raport >= prag_raport) decat a doua cea mai lunga SI
    dureaza cel putin prag_lungime_zile zile, o marcam ca suspecta.

    ATENTIE - LIMITARE DEMONSTRATA, ACCEPTATA CONSTIENT:
      Aceasta metoda NU poate distinge sigur intre "senzorul a murit" si
      "aparatul chiar n-a fost folosit multa vreme". Verificat pe date:
        - casa 2000902, plita electrica: cel mai lung sir = 26.200
          masuratori (~181 zile). Se cunoaste independent (din analiza
          anterioara bazata pe State) ca aici senzorul a fost mort.
        - casa 2000910, masina de spalat: cel mai lung sir = 47.502
          masuratori (~330 zile), dintr-un aparat confirmat SANATOS - pur
          si simplu nu a fost folosita aproape deloc.
      Al doilea caz are un sir mai lung decat primul. Statistic sunt
      identice. Aceasta functie va marca ambele cazuri drept "suspecte",
      deci poate elimina din eroare aparate/case bune (fals pozitiv) sau
      poate rata aparate stricate cu tipar mai putin extrem (fals negativ).
      E cea mai buna aproximare posibila FARA coloana State, nu o solutie
      sigura - decizie asumata, nu eroare de implementare.

    Returneaza o masca (True/False, aceeasi forma ca df) - NU modifica df.
    Se calculeaza pe datele ORIGINALE, nu pe rezultatul masca_pana_generala,
    ca sa nu se fragmenteze un sir lung real in bucati mici de cate una din
    cele doua metode, mascand astfel problema una pe cealalta.
    """
    ordine = df.sort_values(["HouseIDREF", "ApplianceIDREF", "DataOra"]).index
    prag_masuratori = prag_lungime_zile * 24 * MASURATORI_PE_ORA

    def _marcheaza_grup(valori):
        v = valori.to_numpy()
        e_zero = v == 0
        start, sfarsit = _limitele_sirurilor(e_zero)
        lungimi = sfarsit - start
        de_marcat = np.zeros(len(v), dtype=bool)
        if len(lungimi) >= 1:
            ordin = np.argsort(lungimi)[::-1]
            top1 = lungimi[ordin[0]]
            top2 = lungimi[ordin[1]] if len(lungimi) > 1 else 0
            raport = top1 / (top2 + 1)
            if top1 >= prag_masuratori and raport >= prag_raport:
                de_marcat[start[ordin[0]]:sfarsit[ordin[0]]] = True
        return de_marcat

    marcaje_ordonate = (
        df.loc[ordine].groupby(["HouseIDREF", "ApplianceIDREF"])["Value"]
        .transform(_marcheaza_grup)
    )
    masca = marcaje_ordonate.astype(bool).reindex(df.index)

    print("\n=== PASUL 2b (ii): siruri suspecte de zerouri (fara State) ===")
    print(f"Masuratori marcate ca lipsa: {masca.sum()} "
          f"({100 * masca.mean():.2f}% din total)")
    print("ATENTIE: metoda asta poate confunda un aparat stricat cu unul "
          "pur si simplu nefolosit mult timp (limitare demonstrata, vezi "
          "docstring-ul functiei).")
    return masca


def marcheaza_valorile_suspecte(df):
    """
    Combina cele doua semnale de mai sus (pana generala + sir suspect),
    calculate AMANDOUA pe datele originale, si le aplica intr-o singura
    trecere. Asa niciuna dintre metode nu ii "stinge" celeilalte dovada,
    ca s-ar fi intamplat daca le lantuiam una peste rezultatul celeilalte.
    """
    m1 = masca_pana_generala(df)
    m2 = masca_siruri_suspecte(df)
    masca_finala = m1 | m2

    df = df.copy()
    df.loc[masca_finala, "Value"] = np.nan

    print(f"\nTotal marcat (2b, combinat): {masca_finala.sum()} masuratori "
          f"({100 * masca_finala.mean():.2f}% din total; suprapunere intre "
          f"cele doua semnale: {(m1 & m2).sum()})")
    return df


# =========================================================================
# PASUL 2c: Care case au un an complet de date?
# =========================================================================
def marcheaza_case(
    df,
    prag_lipsa_luna=50.0,
    prag_lipsa_pe_aparat=15.0,
    min_luni_consecutive=2,
):
    """
    CERINTA 4: o functie care marcheaza casele ca valide sau invalide.

    -------------------------------------------------------------------
    Vrem case care au fost masurate continuu un an intreg. Daca la o casa
    lipsesc luni intregi (oamenii au plecat, s-a stricat aparatura), datele
    ei ar strica mediile, deci o scoatem din analiza.

    Functia calculeaza "cat de completa e casa", apoi da un verdict
    Valid / Invalid si spune si DE CE, in coloana "Motiv".

    -------------------------------------------------------------------
    ISTORIC - de ce doar 2 verificari, nu 5
    -------------------------------------------------------------------
    Varianta initiala avea 5 verificari (durata totala, pauza maxima intre
    doua masuratori, % lipsa pe casa insumat, % lipsa pe aparat, luni
    lipsa). Verificate pe datele reale, primele 3 nu discriminau NIMIC:
      - durata: toate cele 14 case aveau 364.7-398.9 zile, niciodata sub
        pragul de 350 - nu respingea nicio casa, niciodata.
      - pauza maxima: identica, 1.2 ore, la toate cele 14 case - nu varia
        deloc.
      - % lipsa pe casa: 13 din 14 case aveau 0.0%, iar cea mai proasta
        (2000900) avea 6.8% - tot sub pragul de 10%, deci nu respingea
        nicio casa nici asta.
    Au fost scoase. Au ramas doar cele doua care chiar au dat verdicte
    diferite intre case pe datele astea.

    -------------------------------------------------------------------
    VERIFICAREA 1: PROCENT DE MASURATORI LIPSA PE FIECARE APARAT
    -------------------------------------------------------------------
    Pentru fiecare aparat din casa, separat, ce procent din masuratorile
    lui lipsesc? Luam cel mai afectat aparat din casa. Trebuie sub 15%.

    DE CE PE APARAT, NU PE SUMA CASEI:
    Daca am aduna consumul tuturor aparatelor la un moment dat si am
    verifica doar suma, un aparat complet mort ar putea sa nu se observe -
    daca la ora 10:00 frigiderul are masuratoare buna dar boilerul
    lipseste, suma tot iese un numar, si momentul pare in regula.
    Efectul real pe datele noastre - casa 2000902: boilerul si plita sunt
    COMPLET moarte luni la rand, dar frigiderul, televizorul si becurile
    merg normal - o verificare pe suma ar rata complet problema. De asta
    ne uitam la fiecare aparat separat, nu la suma.

    -------------------------------------------------------------------
    VERIFICAREA 2: LUNI LIPSA
    -------------------------------------------------------------------
    Verificam luna cu luna, pentru fiecare aparat separat, cat la suta din
    masuratorile asteptate exista de fapt (Acoperire_%). O luna sub 50%
    e "sub prag" pentru aparatul respectiv.

    Nu respingem o casa doar pentru ca a avut o luna slaba la un singur
    aparat - ala e un senzor care s-a stricat o data, nu "casa lipseste".
    Respingem doar doua situatii, care chiar strica analiza:

      (i) GOL DE MAI MULTE LUNI
          Acelasi aparat are 2 sau mai multe luni CONSECUTIVE sub prag.
          Prinde casa 2000902 (5 luni consecutive).

      (ii) PANA GENERALA
          O luna intreaga e sub prag la TOATE aparatele deodata. Nu e
          vacanta, e o cadere a aparaturii de inregistrare - dar lipseste
          la fel de mult consum.
          Prinde casa 2000900.

    Contraexemplu, ca sa se vada diferenta: casa 2000907 are masina de
    spalat la 35% intr-o singura luna, restul aparatelor 100%. Nu o
    respingem - pierderea e mica si localizata.

    -------------------------------------------------------------------
    DOUA DETALII TEHNICE
    -------------------------------------------------------------------
    a) Prima si ultima luna ale unui aparat sunt de obicei incomplete
       (masuratorile au inceput pe 20 ianuarie, nu pe 1). Le sarim, dar
       numai daca chiar sunt incomplete ca inregistrare (sub 90% din
       randurile calendaristice). Nu le sarim automat "pentru ca sunt
       primele" - altfel o casa care moare in ultimele luni ar scapa.

    b) Daca o luna lipseste COMPLET din tabel, ea nu apare deloc cand
       grupam datele, deci nu ar putea trage in jos niciun minim. Ca sa
       evitam asta, completam explicit lista de luni de la prima la ultima
       si punem 0 acolo unde nu exista nimic.
    """
    # -----------------------------------------------------------------
    # (a) Verificarea 1: procent de masuratori lipsa PE FIECARE APARAT
    #     Luam aparatul cel mai afectat din fiecare casa.
    # -----------------------------------------------------------------
    rezultat = (
        df.assign(Lipsa=df["Value"].isna())
        .groupby(["HouseIDREF", "ApplianceIDREF"])["Lipsa"].mean()
        .groupby(level=0).max()
        .mul(100).round(1)
        .rename("Procent_Lipsa_Aparat_Max")
        .to_frame()
    )

    # -----------------------------------------------------------------
    # (b) Verificarea 2: luna cu luna, pentru fiecare aparat
    # -----------------------------------------------------------------
    pe_aparat = df[["HouseIDREF", "ApplianceIDREF", "DataOra", "Value"]].copy()
    # tz_localize(None) doar ca sa nu apara un warning de la pandas;
    # granitele lunilor raman cele din ora locala, care ne intereseaza.
    pe_aparat["Luna"] = pe_aparat["DataOra"].dt.tz_localize(None).dt.to_period("M")

    lunar = pe_aparat.groupby(["HouseIDREF", "ApplianceIDREF", "Luna"]).agg(
        masuratori_bune=("Value", "count"),   # count sare peste NaN
        randuri_existente=("Value", "size"),  # size le numara pe toate
    )

    # Detaliul (b) din docstring: completam lunile lipsa cu zero
    toate_lunile = pd.MultiIndex.from_tuples(
        [
            (casa, aparat, luna)
            for (casa, aparat), g in lunar.groupby(level=[0, 1])
            for luna in pd.period_range(
                g.index.get_level_values("Luna").min(),
                g.index.get_level_values("Luna").max(),
                freq="M",
            )
        ],
        names=["HouseIDREF", "ApplianceIDREF", "Luna"],
    )
    lunar = lunar.reindex(toate_lunile).fillna(0).astype(int)

    # Cate masuratori ar trebui sa fie intr-o luna completa:
    # 6 pe ora x 24 ore x numarul de zile al lunii respective
    zile_in_luna = lunar.index.get_level_values("Luna").days_in_month
    lunar["Masuratori_Asteptate"] = MASURATORI_PE_ORA * 24 * zile_in_luna
    lunar["Acoperire_%"] = (
        100 * lunar["masuratori_bune"] / lunar["Masuratori_Asteptate"]
    ).round(1)

    # Detaliul (a) din docstring: sarim lunile de la capete DOAR daca sunt
    # incomplete ca inregistrare
    numar_luna = pd.Series(
        lunar.index.get_level_values("Luna").asi8, index=lunar.index
    )
    e_prima_sau_ultima = (
        numar_luna == numar_luna.groupby(level=[0, 1]).transform("min")
    ) | (
        numar_luna == numar_luna.groupby(level=[0, 1]).transform("max")
    )
    lunar["O_Evaluam"] = (~e_prima_sau_ultima) | (
        lunar["randuri_existente"] >= 0.9 * lunar["Masuratori_Asteptate"]
    )

    de_evaluat = lunar[lunar["O_Evaluam"]].copy()
    rezultat = rezultat.join(
        de_evaluat.groupby(level=0)["Acoperire_%"].min()
        .rename("Cea_Mai_Slaba_Luna_%")
    )

    # -----------------------------------------------------------------
    # (e) Cele doua situatii care duc la respingere
    # -----------------------------------------------------------------
    de_evaluat["Sub_Prag"] = de_evaluat["Acoperire_%"] < prag_lipsa_luna

    def _cel_mai_lung_sir(valori):
        """
        Cel mai lung sir de luni consecutive marcate ca fiind sub prag.
        Foloseste acelasi ajutator _lungimile_sirurilor ca restul fisierului
        (vezi PASUL 2b), in loc sa reimplementeze logica cu o bucla proprie.
        """
        lungimi = _lungimile_sirurilor(valori)
        return int(lungimi.max()) if len(lungimi) else 0

    # (i) gol de mai multe luni, la acelasi aparat
    rezultat = rezultat.join(
        de_evaluat.groupby(level=[0, 1])["Sub_Prag"].apply(_cel_mai_lung_sir)
        .groupby(level=0).max()
        .rename("Luni_Consecutive_Lipsa")
    )

    # (ii) pana generala: o luna sub prag la TOATE aparatele deodata
    rezultat = rezultat.join(
        de_evaluat.groupby(level=[0, 2])["Sub_Prag"].all()
        .groupby(level=0).sum()
        .rename("Luni_Pana_Generala")
    )

    # -----------------------------------------------------------------
    # (f) Retinem luna si aparatul cel mai afectat, ca verdictul sa poata
    #     fi explicat cu un exemplu concret, nu doar cu un numar
    # -----------------------------------------------------------------
    nume_aparate = (
        df.drop_duplicates(["HouseIDREF", "ApplianceIDREF"])
        .set_index(["HouseIDREF", "ApplianceIDREF"])["Name"]
    )
    cel_mai_slab = de_evaluat.groupby(level=0)["Acoperire_%"].idxmin()
    rezultat["Luna_Critica"] = pd.Series(
        {casa: str(idx[2]) for casa, idx in cel_mai_slab.items()}
    )
    rezultat["Aparat_Critic"] = pd.Series(
        {casa: nume_aparate.get((idx[0], idx[1]), f"aparat #{idx[1]}")
         for casa, idx in cel_mai_slab.items()}
    )

    # -----------------------------------------------------------------
    # (g) Verdictul final: casa e valida daca trece AMBELE verificari
    # -----------------------------------------------------------------
    rezultat["Status"] = np.where(
        (rezultat["Procent_Lipsa_Aparat_Max"] <= prag_lipsa_pe_aparat)
        & (rezultat["Luni_Consecutive_Lipsa"] < min_luni_consecutive)
        & (rezultat["Luni_Pana_Generala"] == 0),
        "Valid",
        "Invalid",
    )

    # Scriem in clar de ce a picat fiecare casa
    motive = pd.Series("", index=rezultat.index)
    motive += np.where(
        rezultat["Procent_Lipsa_Aparat_Max"] > prag_lipsa_pe_aparat,
        "un aparat masurat prost tot anul; ", "")
    motive += np.where(
        rezultat["Luni_Consecutive_Lipsa"] >= min_luni_consecutive,
        "ii lipsesc mai multe luni la rand; ", "")
    motive += np.where(
        rezultat["Luni_Pana_Generala"] > 0,
        "pana generala o luna intreaga; ", "")
    rezultat["Motiv"] = motive.str.rstrip("; ")

    return rezultat.reset_index(), lunar.reset_index()


# =========================================================================
# PASUL 2d: Umplem golurile cu media valorilor vecine
# =========================================================================
def completeaza_golurile(df, fereastra="1h"):
    """
    CERINTA 6, partea de "inlocuire cu media".

    Fiecare valoare lipsa (marcata la pasul 2b) e inlocuita cu o medie.
    Folosim doua metode, in ordinea asta, pentru ca golurile din date sunt
    de doua feluri complet diferite.

    -------------------------------------------------------------------
    CE FEL DE GOLURI AVEM (masurat pe datele noastre)
    -------------------------------------------------------------------
    In casele valide raman 45.337 masuratori lipsa, grupate in 221 de
    "pete" de gol. Dar petele nu sunt deloc egale:
      - 39 de pete sunt scurte  (sub o ora)  -> 77 masuratori in total
      - 69 de pete tin peste o zi            -> 40.300 masuratori
    Deci majoritatea golurilor nu sunt "o masuratoare pierduta", ci
    perioade lungi in care senzorul a fost mort. Nu poti umple o zi
    intreaga cu media ultimelor 10 minute, pentru ca ultimele 10 minute
    lipsesc si ele.

    -------------------------------------------------------------------
    METODA 1: media vecinilor imediati (pentru golurile scurte)
    -------------------------------------------------------------------
    Sa zicem ca la ora 14:20 lipseste masuratoarea de la frigider. Ne
    uitam la toate masuratorile frigiderului din intervalul 13:50 - 14:50
    (o ora, centrata pe momentul lipsa), facem media lor si o punem in loc.

    Concret: daca frigiderul a consumat 12, 14, [lipsa], 13, 15 Wh in
    masuratori consecutive, golul devine media lor, adica 13.5 Wh.

    -------------------------------------------------------------------
    METODA 2: media aparatului la aceeasi ora din zi (pentru golurile lungi)
    -------------------------------------------------------------------
    Daca lipseste o zi intreaga, vecinii nu ne ajuta. Atunci ne folosim de
    faptul ca aparatele au un tipar zilnic destul de constant.

    Pentru fiecare aparat calculam, din TOATE zilele in care a fost masurat
    corect, cat consuma in medie la ora 0, cat la ora 1, si asa mai departe
    pana la ora 23. Apoi, pentru o valoare lipsa de marti la 14:20, punem
    media aparatului la ora 14.

    Exemplu: daca boilerul consuma in medie 180 Wh la ora 6 dimineata in
    toate zilele masurate, atunci golurile lui de la ora 6 se completeaza
    cu 180 Wh.

    """
    df = df.copy().sort_values(["HouseIDREF", "ApplianceIDREF", "DataOra"])
    goluri_initial = int(df["Value"].isna().sum())

    # --- METODA 1: media vecinilor din ora din jur -----------------------
    medie_vecini = (
        df.set_index("DataOra")
        .groupby(["HouseIDREF", "ApplianceIDREF"])["Value"]
        .transform(lambda s: s.rolling(fereastra, center=True, min_periods=1).mean())
        .to_numpy()
    )
    df["Value"] = df["Value"].fillna(pd.Series(medie_vecini, index=df.index))
    dupa_metoda1 = int(df["Value"].isna().sum())

    # --- METODA 2: media aparatului la aceeasi ora din zi ----------------
    df["_ora_din_zi"] = df["DataOra"].dt.hour
    medie_pe_ora_din_zi = (
        df.groupby(["HouseIDREF", "ApplianceIDREF", "_ora_din_zi"])["Value"]
        .transform("mean")
    )
    df["Value"] = df["Value"].fillna(medie_pe_ora_din_zi)
    df = df.drop(columns=["_ora_din_zi"])
    goluri_ramase = int(df["Value"].isna().sum())

    print("\n=== PASUL 2d: completarea golurilor cu media ===")
    print(f"Goluri de completat:                        {goluri_initial}")
    print(f"  completate cu media vecinilor (ora din jur):   "
          f"{goluri_initial - dupa_metoda1}")
    print(f"  completate cu media aparatului la aceeasi ora: "
          f"{dupa_metoda1 - goluri_ramase}")
    print(f"  ramase goale (nu exista date deloc):           {goluri_ramase}")
    print(f"Total completat: "
          f"{100 * (goluri_initial - goluri_ramase) / max(goluri_initial, 1):.1f}% "
          f"din goluri, adica "
          f"{100 * (goluri_initial - goluri_ramase) / len(df):.2f}% "
          f"din toate masuratorile")
    return df


# =========================================================================
# PASUL 3: Valori exagerat de mari care au scapat de filtrul de la 2b
# =========================================================================
def _putere_din_nume(nume):
    """
    Multe aparate au puterea scrisa chiar in nume, de exemplu
    'Halogen lamp 1 (0,5kW)'. Extragem 0,5 kW = 500 W si calculam cat ar
    consuma becul ala daca ar sta aprins 10 minute la putere maxima:
        500 W x (10/60) ore = 83.3 Wh
    Aia e limita fizica: mai mult de atat nu are cum sa consume.

    Daca numele nu contine nicio putere, returnam infinit - adica nu putem
    impune nicio limita pentru aparatul ala.
    """
    gasit = re.search(r"([\d,\.]+)\s*kW", str(nume))
    if not gasit:
        return np.inf
    return float(gasit.group(1).replace(",", ".")) * 1000 / MASURATORI_PE_ORA


def filtreaza_valorile_exagerate(df, factor=3.0, toleranta_nominal=2.0,
                                 fereastra="50min"):
    """
    CERINTA 6, partea de "filtrare a spike-urilor".

    Cautam valori absurd de mari care nu au fost prinse la pasul 2b
    (acela prinde mai ales zerouri/goluri, nu neaparat varfurile mari).

    CUM DECIDEM CA O VALOARE E ABSURDA (oricare din cele doua reguli):

      (A) LIMITA FIZICA
          Daca aparatul are puterea scrisa in nume, stim cat poate consuma
          la maximum in 10 minute. Orice depaseste dublul acelei limite e
          imposibil.
          Verificat pe date: masina de spalat de 3,25 kW are ca maxim real
          537 Wh/10min, adica 537 x 6 = 3222 W - exact cat scrie pe ea.

      (B) LIMITA RELATIVA
          Pentru aparatele fara putere in nume, ne uitam la cum consuma ele
          de obicei cand sunt PORNITE, si cerem ca maximul sa nu fie de mai
          mult de 3 ori peste asta.

          De ce "cand sunt pornite" si nu "in medie": un electrocasnic sta
          oprit majoritatea timpului. Daca am lua media peste tot, ea ar fi
          aproape zero (dominata de orele in care aparatul sta), si atunci
          ORICE pornire normala ar parea o anomalie.

          Ca sa aflam "cat consuma cand e pornit", luam doar valorile
          diferite de zero si ne uitam la valoarea sub care se afla 99.9%
          din masuratorile de functionare.
          Practic, cat consuma la efort maxim in regim normal.

    CU CE INLOCUIM valorile gasite: cu media vecinilor pe 50 de minute,
    la fel ca la pasul 2d.

    CE REZULTAT NE ASTEPTAM: putine sau zero. Printam dedesubt si dovada:
    cat de aproape e maximul fiecarui aparat de regimul lui normal de
    functionare.
    """
    df = df.copy().sort_values(["HouseIDREF", "ApplianceIDREF", "DataOra"])
    pe_aparat = [df["HouseIDREF"], df["ApplianceIDREF"]]

    valori_nenule = df["Value"].where(df["Value"] > 0)
    consum_normal_pornit = valori_nenule.groupby(pe_aparat).transform(
        lambda s: s.quantile(0.999)
    )

    limita_fizica = df["Name"].map(_putere_din_nume) * toleranta_nominal
    limita_relativa = consum_normal_pornit * factor

    df["Exagerat"] = (df["Value"] > limita_fizica) | (df["Value"] > limita_relativa)
    nr = int(df["Exagerat"].sum())

    if nr:
        fara_exagerate = df.assign(V=df["Value"].mask(df["Exagerat"]))
        medie_vecini = (
            fara_exagerate.set_index("DataOra")
            .groupby(["HouseIDREF", "ApplianceIDREF"])["V"]
            .transform(lambda s: s.rolling(fereastra, center=True, min_periods=1).mean())
            .to_numpy()
        )
        df["Value"] = np.where(df["Exagerat"], medie_vecini, df["Value"])

    print("\n=== PASUL 3: valori exagerate ramase ===")
    print(f"Valori exagerate gasite:  {nr} din {len(df)} "
          f"({100 * nr / len(df):.4f}%)")
    print(f"Energie totala pastrata:  {df['Value'].sum() / 1000:.0f} kWh")

    # Dovada ca "0 gasite" e o concluzie, nu o functie care nu ruleaza:
    # cat de departe e maximul fiecarui aparat de regimul lui normal.
    maxim_pe_aparat = df["Value"].groupby(pe_aparat).max()
    normal_pe_aparat = valori_nenule.groupby(pe_aparat).quantile(0.999)
    raport = (maxim_pe_aparat / normal_pe_aparat).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    print(f"Verificare pe cele {len(raport)} aparate - de cate ori e maximul "
          f"peste consumul normal de functionare:")
    print(f"  tipic (mediana): {raport.median():.2f}x, cel mai rau caz: "
          f"{raport.max():.2f}x, prag de alarma: {factor:.1f}x")
    if nr == 0:
        print(f"  => Niciun aparat nu se apropie de prag. Valorile absurde "
              f"fusesera deja prinse la pasul 2b.")

    return df.drop(columns=["Exagerat"])


# =========================================================================
# PASUL 4: De la 10 minute la consum pe ora
# =========================================================================
def calculeaza_consum_orar(df, min_masuratori=4, rescaleaza=True):
    """
    CERINTA 5: o functie care calculeaza consumul pe ora, si pentru fiecare
    aparat, si pentru toata casa (datele initiale sunt din 10 in 10 minute).

    SE FACE IN DOUA ETAPE, amandoua cu adunare:

      Etapa 1 - in timp, pentru acelasi aparat:
        Energia consumata intr-o ora = suma energiilor din cele 6 intervale
        de 10 minute ale acelei ore.
            E_ora = E_(00-10) + E_(10-20) + ... + E_(50-60)

      Etapa 2 - intre aparate, in aceeasi ora:
        Consumul casei intr-o ora = suma consumurilor tuturor aparatelor in
        acea ora. Aici adunarea e evidenta.

    ORE INCOMPLETE:
      Dupa pasul 2d ar trebui ca aproape toate orele sa aiba 6 masuratori.
      Daca totusi o ora are mai putin de 4, o scoatem (prea putine date ca
      sa spunem ceva despre ea). Pentru orele cu 4 sau 5 masuratori,
      extindem proportional la ora intreaga: suma x 6 / cate avem.
    """
    df = df.copy()

    # Rotunjim la ora fixa. Facem rotunjirea in UTC si abia apoi revenim la
    # ora locala: la trecerea de la ora de vara la cea de iarna, ora 02:00
    # exista de doua ori in aceeasi zi, iar pandas ar da eroare daca ar
    # trebui sa aleaga. Decalajul Frantei fiind un numar intreg de ore,
    # granitele orelor sunt oricum aceleasi in ambele fusuri.
    fus = df["DataOra"].dt.tz
    df["Ora"] = df["DataOra"].dt.tz_convert("UTC").dt.floor("h").dt.tz_convert(fus)

    consum_orar_aparat = df.groupby(
        ["HouseIDREF", "ApplianceIDREF", "Ora"], as_index=False
    ).agg(
        Consum_Orar_Wh=("Value", "sum"),
        Nr_Masuratori=("Value", "count"),
    )

    inainte = len(consum_orar_aparat)
    consum_orar_aparat = consum_orar_aparat[
        consum_orar_aparat["Nr_Masuratori"] >= min_masuratori
    ].copy()

    if rescaleaza:
        consum_orar_aparat["Consum_Orar_Wh"] *= (
            MASURATORI_PE_ORA / consum_orar_aparat["Nr_Masuratori"]
        )

    consum_orar_casa = (
        consum_orar_aparat.groupby(["HouseIDREF", "Ora"], as_index=False)
        ["Consum_Orar_Wh"].sum()
        .rename(columns={"Consum_Orar_Wh": "Consum_Orar_Casa_Wh"})
    )

    print("\n=== PASUL 4: consum pe ora (in Wh) ===")
    print(f"Masuratori la 10 minute:                {len(df)}")
    print(f"Ore obtinute, pe aparat:                {inainte}")
    print(f"  scoase (sub {min_masuratori} masuratori in ora):        "
          f"{inainte - len(consum_orar_aparat)}")
    print(f"Ore obtinute, pe casa:                  {len(consum_orar_casa)}")
    print(f"Cate masuratori a inghitit o ora:       "
          f"{len(df) / max(len(consum_orar_aparat), 1):.2f}  (normal ~6)")
    return consum_orar_aparat, consum_orar_casa


def compara_cu_contorul_general(consum_orar_casa, df_contor_general):
    """
    VERIFICARE

    Fiecare casa are, pe langa aparatele monitorizate individual, un contor
    general care masoara TOT ce consuma casa ("Site consumption"). Pe acela
    l-am scos din analiza la pasul 1, ca sa nu numaram de doua ori - dar il
    putem folosi acum ca sa verificam ca nu am gresit nimic.

    DOUA LUCRURI VERIFICAM:

      1. Suma aparatelor nu are voie sa depaseasca contorul general.
         Daca ar depasi, ar insemna ca am numarat ceva de doua ori sau ca
         am gresit unitatile de masura.

      2. Cat la suta din casa e de fapt monitorizata pe aparate.
         Niciodata nu e 100%: nu toate circuitele au senzor. Lumina din
         hol, calculatorul, incarcatoarele - toate intra in contorul
         general dar nu au aparat dedicat. Un procent de 30-70% e normal
         pentru genul asta de studiu.


    COLOANELE TABELULUI:
      Suma_Aparate_kWh       - cat au consumat, la un loc, aparatele
                               monitorizate individual, intr-un an
      Contor_General_kWh     - cat a consumat toata casa intr-un an,
                               dupa contorul general
      Contor_Masurat_%       - cat la suta din an a functionat contorul
                               general (restul l-am extrapolat)
      Din_Casa_Monitorizat_% - Suma_Aparate / Contor_General.
                               Cat la suta din consumul casei e explicat de
                               aparatele pe care le analizam.
    """
    # Aplicam acelasi detector de "sir suspect" ca la aparate (fara State),
    # tratand contorul general ca un singur "aparat" per casa.
    contor = df_contor_general.copy()
    contor["ApplianceIDREF"] = 0
    contor = converteste_timpul(contor)
    contor.loc[masca_siruri_suspecte(contor), "Value"] = np.nan
    grupat = contor.groupby("HouseIDREF")["Value"]

    fractiune_masurata = grupat.count() / grupat.size()
    total_contor_kwh = (grupat.sum() / fractiune_masurata) / 1000
    total_aparate_kwh = (
        consum_orar_casa.groupby("HouseIDREF")["Consum_Orar_Casa_Wh"].sum() / 1000
    )

    tabel = pd.DataFrame({
        "Suma_Aparate_kWh": total_aparate_kwh.round(0),
        "Contor_General_kWh": total_contor_kwh.round(0),
        "Contor_Masurat_%": (fractiune_masurata * 100).round(1),
    }).dropna()
    tabel["Din_Casa_Monitorizat_%"] = (
        100 * tabel["Suma_Aparate_kWh"] / tabel["Contor_General_kWh"]
    ).round(1)

    print("\n=== VERIFICARE: suma aparatelor fata de contorul general "
          "al casei (consum pe un an) ===")
    print(tabel.to_string())
    print("\nCitire: 'Din_Casa_Monitorizat_%' arata cat din consumul total "
          "al casei este explicat")
    print("de aparatele analizate. Restul sunt consumatori fara senzor "
          "propriu (lumini, prize, etc).")

    depasiri = tabel[tabel["Din_Casa_Monitorizat_%"] > 100]
    if len(depasiri):
        print(f"!! PROBLEMA: la {len(depasiri)} case suma aparatelor "
              f"depaseste contorul general. Ceva e gresit in calcule.")
    else:
        print("OK: la nicio casa suma aparatelor nu depaseste contorul "
              "general. Calculele sunt consistente.")
    return tabel


# =========================================================================
# PASUL 5: Graficele interactive
# =========================================================================
def genereaza_grafice(id_casa, consum_orar_aparat, consum_orar_casa,
                      df_aparate, aparat=None, salveaza_html=True):
    """
    CERINTELE 7 si 8: grafice interactive Plotly

    GRAFICELE CERUTE EXPLICIT, si cum se numesc fisierele:

      "consumul in timp al unui consumator"
          -> aparat_consum_pe_zile.html      (o valoare pe zi, se vede tot anul)
          -> aparat_consum_pe_ore.html       (o valoare pe ora, detaliu fin)

      "consumul orar al unui consumator"
          -> aparat_medie_pe_ora_din_zi.html (media la ora 0, 1, 2 ... 23)

      "consumul casei in timp"
          -> casa_consum_pe_zile.html
          -> casa_consum_pe_ore.html

      "consumul orar al casei"
          -> casa_medie_pe_ora_din_zi.html

      -> clasament_aparate.html   : care e cel mai mare consumator
      -> sezonalitate_pe_luni.html: cum se schimba consumul de la iarna la vara

    """
    casa = consum_orar_casa[
        consum_orar_casa["HouseIDREF"] == id_casa
    ].sort_values("Ora")
    aparate = consum_orar_aparat[
        consum_orar_aparat["HouseIDREF"] == id_casa
    ].copy()
    if casa.empty or aparate.empty:
        raise ValueError(f"Nu exista date orare pentru casa {id_casa}.")

    nume = df_aparate.loc[df_aparate["HouseIDREF"] == id_casa, ["ID", "Name"]]
    aparate = aparate.merge(nume, left_on="ApplianceIDREF", right_on="ID",
                            how="left")
    aparate["Name"] = aparate["Name"].fillna(
        "Aparat #" + aparate["ApplianceIDREF"].astype(str)
    )

    grafice = {}

    # -----------------------------------------------------------------
    # 1. Consumul casei, o valoare pe zi
    #    La rezolutie de o ora ar fi 8760 de puncte pe an, prea multe ca sa
    #    se vada diferenta iarna/vara. Adunate pe zi, tendinta sezoniera
    #    se vede clar.
    # -----------------------------------------------------------------
    pe_zile = (
        casa.set_index("Ora")["Consum_Orar_Casa_Wh"]
        .resample("D").sum().div(1000)
        .rename("kWh").reset_index()
    )
    grafice["casa_consum_pe_zile"] = px.line(
        pe_zile, x="Ora", y="kWh",
        title=f"Casa {id_casa}: cat consuma in fiecare zi, de-a lungul anului",
        labels={"Ora": "Data", "kWh": "Consum pe zi (kWh)"},
    )

    # -----------------------------------------------------------------
    # 2. Consumul casei, o valoare pe ora (toata seria)
    #    render_mode="webgl" ca zoom-ul sa ramana fluid pe 8760 de puncte.
    # -----------------------------------------------------------------
    grafice["casa_consum_pe_ore"] = px.line(
        casa, x="Ora", y="Consum_Orar_Casa_Wh", render_mode="webgl",
        title=f"Casa {id_casa}: consumul ora cu ora",
        labels={"Ora": "Data si ora", "Consum_Orar_Casa_Wh": "Consum intr-o ora (Wh)"},
    )

    # -----------------------------------------------------------------
    # 3. Media casei pe fiecare ora din zi (0 - 23)
    #    Raspunde la "cand se consuma in general?" din cerinta 8.
    #    Luam toate zilele anului si facem media pentru ora 0, apoi pentru
    #    ora 1, si asa mai departe. Rezultatul arata programul zilnic
    #    tipic al locuintei.
    # -----------------------------------------------------------------
    casa = casa.assign(Ora_Din_Zi=casa["Ora"].dt.hour)
    medie_pe_ora = (
        casa.groupby("Ora_Din_Zi", as_index=False)["Consum_Orar_Casa_Wh"].mean()
        .rename(columns={"Consum_Orar_Casa_Wh": "Wh"})
    )
    grafice["casa_medie_pe_ora_din_zi"] = px.line(
        medie_pe_ora, x="Ora_Din_Zi", y="Wh", markers=True,
        title=f"Casa {id_casa}: cat consuma in medie la fiecare ora din zi "
              f"(media pe tot anul, ora Frantei)",
        labels={"Ora_Din_Zi": "Ora din zi (0 = miezul noptii)",
                "Wh": "Consum mediu intr-o ora (Wh)"},
    ).update_xaxes(dtick=1)

    # -----------------------------------------------------------------
    # 4. Clasamentul aparatelor
    # -----------------------------------------------------------------
    clasament = (
        aparate.groupby("Name", as_index=False)["Consum_Orar_Wh"].sum()
        .assign(kWh=lambda d: (d["Consum_Orar_Wh"] / 1000).round(1))
        .sort_values("kWh", ascending=False)
    )
    grafice["clasament_aparate"] = px.bar(
        clasament, x="Name", y="kWh", text="kWh",
        title=f"Casa {id_casa}: cat a consumat fiecare aparat intr-un an",
        labels={"Name": "Aparat", "kWh": "Consum pe an (kWh)"},
    ).update_xaxes(categoryorder="total descending")

    # Alegem aparatul analizat
    if aparat is None:
        nume_ales = clasament.iloc[0]["Name"]
    else:
        potriviri = clasament.loc[
            clasament["Name"].str.contains(aparat, case=False, na=False), "Name"
        ]
        if potriviri.empty:
            raise ValueError(
                f"Niciun aparat cu '{aparat}' in nume la casa {id_casa}. "
                f"Aparate disponibile: {list(clasament['Name'])}"
            )
        nume_ales = potriviri.iloc[0]
    date_aparat = aparate[aparate["Name"] == nume_ales].sort_values("Ora")

    # -----------------------------------------------------------------
    # 5. Media aparatului pe fiecare ora din zi
    # -----------------------------------------------------------------
    medie_aparat = (
        date_aparat.assign(Ora_Din_Zi=date_aparat["Ora"].dt.hour)
        .groupby("Ora_Din_Zi", as_index=False)["Consum_Orar_Wh"].mean()
    )
    grafice["aparat_medie_pe_ora_din_zi"] = px.line(
        medie_aparat, x="Ora_Din_Zi", y="Consum_Orar_Wh", markers=True,
        title=f"{nume_ales}: cat consuma in medie la fiecare ora din zi "
              f"(casa {id_casa})",
        labels={"Ora_Din_Zi": "Ora din zi (0 = miezul noptii)",
                "Consum_Orar_Wh": "Consum mediu intr-o ora (Wh)"},
    ).update_xaxes(dtick=1)

    # -----------------------------------------------------------------
    # 6. Consumul aparatului, o valoare pe zi
    # -----------------------------------------------------------------
    aparat_pe_zile = (
        date_aparat.set_index("Ora")["Consum_Orar_Wh"]
        .resample("D").sum().div(1000)
        .rename("kWh").reset_index()
    )
    grafice["aparat_consum_pe_zile"] = px.line(
        aparat_pe_zile, x="Ora", y="kWh",
        title=f"{nume_ales}: cat consuma in fiecare zi, de-a lungul anului "
              f"(casa {id_casa})",
        labels={"Ora": "Data", "kWh": "Consum pe zi (kWh)"},
    )

    # -----------------------------------------------------------------
    # 7. Consumul aparatului, o valoare pe ora
    # -----------------------------------------------------------------
    grafice["aparat_consum_pe_ore"] = px.line(
        date_aparat, x="Ora", y="Consum_Orar_Wh", render_mode="webgl",
        title=f"{nume_ales}: consumul ora cu ora (casa {id_casa})",
        labels={"Ora": "Data si ora", "Consum_Orar_Wh": "Consum intr-o ora (Wh)"},
    )

    # -----------------------------------------------------------------
    # 8. Sezonalitatea: consum mediu pe zi, luna de luna
    #
    #    ATENTIE: NU folosim totalul lunar.
    #    Prima si ultima luna sunt masurate doar partial (la casa 2000901,
    #    ianuarie 1998 are doar 9 zile masurate din 31). Daca am pune
    #    totalul, bara lui ianuarie ar iesi de 3 ori mai mica decat a lui
    #    februarie si s-ar citi gresit ca "iarna se consuma putin" - exact
    #    pe dos fata de realitate.
    #
    #    Solutia: impartim consumul lunii la numarul de zile efectiv
    #    masurate. Asa toate lunile devin comparabile intre ele.
    #    Verificat: ianuarie trece de la 71.6 kWh (parea mic) la 7.7 kWh/zi,
    #    adica exact la nivelul celorlalte luni de iarna.
    # -----------------------------------------------------------------
    aparate = aparate.assign(
        Luna=aparate["Ora"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    )
    zile_masurate = (
        casa.assign(
            Luna=casa["Ora"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
        )
        .groupby("Luna")["Ora"].nunique()
        .div(24)   # cate ore avem in luna, impartit la 24 = cate zile
    )
    pe_luni = (
        aparate.groupby(["Luna", "Name"], as_index=False)["Consum_Orar_Wh"].sum()
        .assign(
            kWh_pe_zi=lambda d: (d["Consum_Orar_Wh"] / 1000)
            / d["Luna"].map(zile_masurate)
        )
    )
    grafice["sezonalitate_pe_luni"] = px.bar(
        pe_luni, x="Luna", y="kWh_pe_zi", color="Name",
        title=f"Casa {id_casa}: consumul mediu pe zi, luna de luna "
              f"(asa se vede diferenta iarna / vara)",
        labels={"Luna": "Luna", "kWh_pe_zi": "Consum mediu pe zi (kWh)",
                "Name": "Aparat"},
    )

    for g in grafice.values():
        g.update_layout(hovermode="x unified")

    if salveaza_html:
        for nume_fisier, g in grafice.items():
            g.write_html(f"grafic_{id_casa}_{nume_fisier}.html",
                         include_plotlyjs="cdn")
        print(f"\n{len(grafice)} grafice salvate:")
        for nume_fisier in grafice:
            print(f"  grafic_{id_casa}_{nume_fisier}.html")

    total_casa = clasament["kWh"].sum()
    kwh_ales = float(clasament.loc[clasament["Name"] == nume_ales, "kWh"].iloc[0])
    eticheta = ("Cel mai mare consumator" if aparat is None
                else "Aparatul analizat")

    print(f"\n=== Rezumat casa {id_casa} ===")
    print(f"Consum total pe un an:   {total_casa:.0f} kWh")
    print(f"{eticheta}: {nume_ales} ({kwh_ales:.0f} kWh/an, "
          f"{100 * kwh_ales / total_casa:.0f}% din total)")
    if aparat is not None and nume_ales != clasament.iloc[0]["Name"]:
        print(f"  (cel mai mare ramane {clasament.iloc[0]['Name']}, "
              f"{clasament.iloc[0]['kWh']:.0f} kWh/an)")
    return grafice


# =========================================================================
# PIPELINE: toti pasii, in ordine
# =========================================================================
def main(case="toate", aparat=None):
    """
    Ruleaza tot lantul de prelucrare, de la citirea bazei de date pana la
    grafice.

    PARAMETRI (amandoi optionali):

      case   = pentru ce case sa faca grafice
               (implicit 2000911, doar ca sa existe o valoare gata pusa)
               None            -> prima casa valida
               2000904         -> doar casa asta
               [2000904, 2000909] -> casele astea
               "toate"         -> toate casele valide

      aparat = ce aparat sa analizeze in graficele de aparat
               None       -> cel mai mare consumator al casei (implicit)
               "washing"  -> aparatul care are "washing" in nume
               (e suficienta o bucata din nume, nu conteaza literele mari)

    EXEMPLE DE FOLOSIRE:

      main()                                 # implicit
      main(case=2000904)                     # alta casa
      main(case=2000904, aparat="fridge")    # alta casa, alt aparat
      main(case="toate")                     # toate casele valide

    RETURNEAZA un dictionar cu toate datele prelucrate, ca sa poti face
    grafice in plus FARA sa mai astepti o data toata prelucrarea:

      rezultate = main()
      arata_optiunile(rezultate)                        # ce case/aparate am
      deseneaza(rezultate, 2000904)                     # inca o casa
      deseneaza(rezultate, 2000904, aparat="fridge")    # si alt aparat
    """
    # --- PASUL 1: citim datele -------------------------------------------
    df_consum, df_aparate, df_contor_general = incarca_date()

    # --- PASUL 2a: timpul in format normal -------------------------------
    df_consum = converteste_timpul(df_consum)

    # --- PASUL 2b: marcam valorile suspecte, fara coloana State ----------
    df_consum = marcheaza_valorile_suspecte(df_consum)

    # --- PASUL 2c: care case au un an complet ----------------------------
    status_case, raport_lunar = marcheaza_case(df_consum)
    print("\n=== PASUL 2c: care case au un an complet de date ===")
    print(status_case[[
        "HouseIDREF", "Procent_Lipsa_Aparat_Max",
        "Cea_Mai_Slaba_Luna_%", "Luni_Consecutive_Lipsa",
        "Luni_Pana_Generala", "Status",
    ]].to_string(index=False))
    print(f"\nCase valide: {(status_case['Status'] == 'Valid').sum()} "
          f"din {len(status_case)}")
    for _, r in status_case[status_case["Status"] == "Invalid"].iterrows():
        print(f"\n  RESPINSA casa {r['HouseIDREF']}")
        print(f"    Motiv: {r['Motiv']}")
        print(f"    Cel mai afectat: aparatul '{r['Aparat_Critic']}', "
              f"in luna {r['Luna_Critica']}, cu doar "
              f"{r['Cea_Mai_Slaba_Luna_%']}% din masuratori prezente")
        print(f"    Luni consecutive lipsa: {r['Luni_Consecutive_Lipsa']} | "
              f"Luni cu pana generala: {r['Luni_Pana_Generala']}")

    case_valide = status_case.loc[status_case["Status"] == "Valid", "HouseIDREF"]
    df_consum = df_consum[df_consum["HouseIDREF"].isin(case_valide)].copy()

    # --- PASUL 2d: acum putem completa golurile in siguranta -------------
    df_consum = completeaza_golurile(df_consum)

    # --- PASUL 3: valori exagerate ---------------------------------------
    df_consum = filtreaza_valorile_exagerate(df_consum)

    # --- PASUL 4: consum pe ora ------------------------------------------
    consum_orar_aparat, consum_orar_casa = calculeaza_consum_orar(df_consum)
    compara_cu_contorul_general(consum_orar_casa, df_contor_general)

    # --- PASUL 5: graficele ----------------------------------------------
    # Stabilim pentru ce case facem grafice, in functie de parametrul `case`
    lista_valide = sorted(case_valide)
    if case is None:
        de_desenat = lista_valide[:1]
    elif isinstance(case, str) and case.lower() == "toate":
        de_desenat = lista_valide
    elif isinstance(case, (list, tuple, set)):
        de_desenat = [int(c) for c in case]
    else:
        de_desenat = [int(case)]

    for id_casa in de_desenat:
        if id_casa not in lista_valide:
            print(f"\n!! Casa {id_casa} nu e in lista de case valide. "
                  f"Case disponibile: {lista_valide}")
            continue
        genereaza_grafice(id_casa, consum_orar_aparat, consum_orar_casa,
                          df_aparate, aparat=aparat)

    # Returnam totul intr-un dictionar, ca sa poti face grafice in plus
    # fara sa mai astepti inca o data toata prelucrarea.
    return {
        "status_case": status_case,
        "raport_lunar": raport_lunar,
        "consum_orar_aparat": consum_orar_aparat,
        "consum_orar_casa": consum_orar_casa,
        "df_aparate": df_aparate,
        "case_valide": lista_valide,
    }


# =========================================================================
# SCURTATURI, ca sa nu rulezi toata prelucrarea de fiecare data
# =========================================================================
def deseneaza(rezultate, id_casa, aparat=None):
    """
    Face graficele pentru inca o casa / inca un aparat, folosind datele deja
    prelucrate de main(). Dureaza o secunda, nu 40.

    FOLOSIRE:
        rezultate = main()
        deseneaza(rezultate, 2000904)
        deseneaza(rezultate, 2000904, aparat="fridge")
    """
    return genereaza_grafice(
        id_casa,
        rezultate["consum_orar_aparat"],
        rezultate["consum_orar_casa"],
        rezultate["df_aparate"],
        aparat=aparat,
    )


def arata_optiunile(rezultate):
    """
    Afiseaza ce case si ce aparate poti cere, ca sa stii ce sa scrii in
    deseneaza() sau in main().

    FOLOSIRE:
        rezultate = main()
        arata_optiunile(rezultate)
    """
    aparate = rezultate["consum_orar_aparat"].merge(
        rezultate["df_aparate"][["ID", "HouseIDREF", "Name"]],
        left_on=["ApplianceIDREF", "HouseIDREF"],
        right_on=["ID", "HouseIDREF"],
        how="left",
    )
    consum = (
        aparate.groupby(["HouseIDREF", "Name"])["Consum_Orar_Wh"].sum()
        .div(1000).round(0)
    )

    print("\n=== Ce poti cere ===")
    for id_casa in rezultate["case_valide"]:
        lista = consum.loc[id_casa].sort_values(ascending=False)
        print(f"\nCasa {id_casa}  (total {lista.sum():.0f} kWh/an)")
        for nume, kwh in lista.items():
            print(f"    {kwh:6.0f} kWh   {nume}")
    print("\nExemplu: deseneaza(rezultate, 2000904, aparat=\"fridge\")")


# =========================================================================
#
#                        ZONA DE COMENZI
#
#   AICI SCRII CE VREI SA RULEZE cand apesi Run in VS Code.
#
#   Ca sa activezi o comanda, sterge diezul "#" din fata ei.
#   Ca sa o dezactivezi la loc, pune diezul inapoi.
#
#   ATENTIE: liniile trebuie sa ramana aliniate cum sunt, cu 4 spatii
#   la inceput. Python se supara daca strici alinierea.
#
# =========================================================================
if __name__ == "__main__":

    # -----------------------------------------------------------------
    # PASUL OBLIGATORIU: asta ruleaza toata prelucrarea.
    # Dureaza vreo 40 de secunde. Trebuie sa fie mereu activ.
    # Rezultatul se pastreaza in variabila "rezultate".
    # -----------------------------------------------------------------
    rezultate = main()

    # -----------------------------------------------------------------
    # OPTIONAL 1: vezi ce case si ce aparate ai la dispozitie
    # (sterge diezul de pe randul de mai jos ca sa il activezi)
    # -----------------------------------------------------------------
    arata_optiunile(rezultate)

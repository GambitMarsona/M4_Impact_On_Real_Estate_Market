# 📊 Analiza wpływu planowanej linii metra M4 na rynek nieruchomości w Warszawie

Projekt badawczy dotyczący predykcji wpływu planowanej linii metra **M4** na ceny mieszkań w Warszawie, z wykorzystaniem metod **uczenia maszynowego**, **analizy przestrzennej** oraz **web scrapingu**.

Repozytorium stanowi zaplecze obliczeniowe do pracy licencjackiej:
> *„Analiza wpływu planowanej linii metra M4 na rynek nieruchomości w Warszawie”*  
Autor: **Marek Polit**, SGH, 2025.
Promotor: Profesor SGH - Małgorzata Wrzosek

---

## 🎯 Cel projektu

Celem projektu jest:
- estymacja wpływu **odległości od stacji metra** na **cenę za m² mieszkania**,
- symulacja potencjalnych zmian cen po uruchomieniu linii **M4**,
- porównanie skuteczności modeli:
  - regresji klasycznej,
  - drzew decyzyjnych,
  - Random Forest,
  - XGBoost,
  - Support Vector Regression.

Projekt odpowiada również na pytanie:
> Czy bliskość metra istotnie wpływa na wartość nieruchomości?

---

## 🗂 Zakres danych

Źródłem danych są ogłoszenia z portalu **Otodom**, pobierane poprzez **web scraping**.

Zakres danych:
- ok. **14 000 ofert mieszkań** z Warszawy,
- dane adresowe,
- cechy techniczne nieruchomości,
- udogodnienia binarne,
- ceny całkowite i **ceny za m²**,
- dane geograficzne (szer. i dł. geograficzna).

---

## 🛠 Stack technologiczny

- **Python**
- **Jupyter Notebook**
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`, `seaborn`
- `googlemaps`
- `selenium`
- `beautifulsoup4`
- `geopandas`
- `shapely`

---

## 🌍 Dane przestrzenne

Do analizy przestrzennej wykorzystano:
- geokodowanie adresów przez **Google Maps API**,
- dane obiektów miejskich z **OpenStreetMap** poprzez **Overpass Turbo**,
- algorytm **BallTree** do szybkiego liczenia odległości.

Wyznaczone odległości:
- do najbliższej stacji metra **M1**
- do najbliższej stacji metra **M2**
- do najbliższego supermarketu
- do najbliższych terenów zielonych

---

## 🧠 Inżynieria cech

Finalny zbiór danych zawiera m.in.:

### Zmienne ciągłe:
- cena,
- cena za m²,
- powierzchnia,
- czynsz,
- rok budowy,
- długość i szerokość geograficzna,
- odległość do metra, sklepów i zieleni.

### Zmienne kategoryczne:
- rynek (pierwotny / wtórny),
- stan wykończenia,
- dzielnica.

### Zmienne binarne (udogodnienia):
- winda,
- balkon,
- garaż,
- internet,
- klimatyzacja,
- monitoring,
- ochrona,
- piwnica,
- taras,
- itp.

---

## 🧹 Czyszczenie danych

W projekcie zastosowano:
- usuwanie braków danych dla kluczowych zmiennych,
- imputację medianą,
- usuwanie zmiennych silnie skorelowanych (|r| > 0.8),
- usuwanie obserwacji odstających:
  - ceny,
  - ceny za m²,
  - powierzchni,
  - roku budowy.

---

## 📈 Analiza eksploracyjna

W ramach EDA wykonano:
- analizę rozkładów cen,
- analizę wpływu:
  - stanu wykończenia,
  - wieku budynku,
  - odległości od metra,
- transformacje logarytmiczne zmiennych odległości,
- segmentację budynków na:
  - przed 1950,
  - 1950–2004,
  - po 2004.

---

## 🤖 Modele predykcyjne

W projekcie zaimplementowano i porównano:
- regresję liniową (model bazowy),
- drzewa decyzyjne,
- Random Forest Regression,
- Support Vector Regression (SVR),
- Gradient Boosting / XGBoost.

Modele oceniane są m.in. przez:
- RMSE,
- MAE,
- R².

---

## 🚇 Symulacja wpływu linii M4

Projekt zawiera:
- symulację zmian cen w oparciu o nowe lokalizacje stacji M4,
- predykcje zmian wartości mieszkań w promieniu do kilku kilometrów,
- wizualizacje przestrzenne prognoz.

---

## 📊 Wizualizacje

W repozytorium znajdują się:
- mapy odległości,
- wykresy gęstości,
- rozkłady cen,
- porównania segmentów rynku,
- wpływ metra na ceny.


# 📊 Analiza wpływu planowanej linii metra M4 na rynek nieruchomości w Warszawie
Autor: **Marek Polit**, SGH, 2025.
Promotor: Profesor SGH - Małgorzata Wrzosek



# Wizualizacje – wpływ linii M4 na rynek nieruchomości

<!-- RZĄD 1 – MAPA + BOX + TREND -->
<table>
<tr>
<td width="34%"><img src="https://github.com/user-attachments/assets/9a098364-24dd-4ad6-8c82-24a339ba0dff" width="100%"/></td>
<td width="33%"><img src="https://github.com/user-attachments/assets/47a45621-12ed-45b5-875c-e3b35e5cdcb8" width="100%"/></td>
<td width="33%"><img src="https://github.com/user-attachments/assets/0829548f-15e3-4c54-ad90-1d2096b45c12" width="100%"/></td>
</tr>
</table>

---

<!-- RZĄD 2 – INTERPOLACJA + ROZKŁADY + PRED VS REAL -->
<table>
<tr>
<td width="33%"><img src="https://github.com/user-attachments/assets/e5e1acdd-199e-4889-b6ee-eac19ac83ecd" width="100%"/></td>
<td width="33%"><img src="https://github.com/user-attachments/assets/0a60a560-908b-40d6-9be2-9fc25a1e9545" width="100%"/></td>
<td width="34%"><img src="https://github.com/user-attachments/assets/26384b0d-d409-45c0-b43a-00fdbca07cf4" width="100%"/></td>
</tr>
</table>

---

<!-- RZĄD 3 – ROK + STAN + UDOGODNIENIA + CHMURA -->
<table>
<tr>
<td width="25%"><img src="https://github.com/user-attachments/assets/abed8278-3206-4504-83f1-2810674c4b5f" width="100%"/></td>
<td width="25%"><img src="https://github.com/user-attachments/assets/9396765b-612a-4e17-b833-7cfb353527a5" width="100%"/></td>
<td width="25%"><img src="https://github.com/user-attachments/assets/05362a5b-b048-4602-a047-14344fb1f432" width="100%"/></td>
<td width="25%"><img src="https://github.com/user-attachments/assets/fa25480f-0f97-43dd-9a0e-1e1e2013a584" width="100%"/></td>
</tr>
</table>

---

<!-- RZĄD 4 – FEATURE IMPORTANCE + DRZEWO + RF -->
<table>
<tr>
<td width="34%"><img src="https://github.com/user-attachments/assets/c36398b2-6a41-49ff-b6b0-11c8334b1fa5" width="100%"/></td>
<td width="33%"><img src="https://github.com/user-attachments/assets/e66760a3-2ba1-468b-b71b-3d4ba2d98c59" width="100%"/></td>
<td width="33%"><img src="https://github.com/user-attachments/assets/fd24991c-f2ed-4d5f-bf74-3f0676ad02df" width="100%"/></td>
</tr>
</table>

---

<!-- RZĄD 5 – MSE + METRYKI + MAPA ZBIORCZA -->
<table>
<tr>
<td width="33%"><img src="https://github.com/user-attachments/assets/b3ecc075-f9e7-426a-b2db-ffd30dd61ddd" width="100%"/></td>
<td width="33%"><img src="https://github.com/user-attachments/assets/8804867a-e611-46c7-82c1-9d3472dbd66c" width="100%"/></td>
<td width="34%"><img src="https://github.com/user-attachments/assets/0def12be-42a9-438f-9eab-ff133448d586" width="100%"/></td>
</tr>
</table>



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

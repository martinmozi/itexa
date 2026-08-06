# XGBoost krok za krokom — ISO 8583 transakcie

> **Poradie čítania:** ← [Random Forest a XGBoost](02-random-forest-a-xgboost.md) · **lekcia 2 — podrobný príklad** · [Feed-forward siete (MLP)](04-feed-forward-siete.md) →

Príklad s bytom bol zámerne zjednodušený. Poďme si teraz celý mechanizmus prejsť **krok za krokom na skutočnom type dát** — na kartových transakciách v štandarde **ISO 8583**. To je protokol, ktorým si terminál, akceptačná banka a vydavateľ karty vymieňajú autorizačné správy. Každá správa je vlastne riadok tabuľky: očíslované dátové elementy (*data elements*, DE) ako DE4 = suma, DE7 = čas, DE18 = typ obchodníka. Presne to, na čo je XGBoost stavaný.

**Úloha:** pre každú prichádzajúcu autorizáciu odhadnúť pravdepodobnosť, že ide o **podvod** (transakcia skončí reklamáciou / chargebackom).

## Dáta

Zoberme 10 transakcií jednej karty. Prvé stĺpce sú priamo z ISO 8583 správy, `tx/60 min` je **odvodený** príznak (o tých viac nižšie), posledný stĺpec `y` je pravda, ktorú sme sa dozvedeli až spätne z reklamácií:

| # | DE4 suma | DE7 čas | DE18 MCC | DE22 vstup | DE43 krajina | tx/60 min | y (podvod) |
|---|---|---|---|---|---|---|---|
| 1 | 12,40 € | 08:12 | 5411 potraviny | čip | SK | 1 | 0 |
| 2 | 34,90 € | 12:45 | 5812 reštaurácia | bezkontakt | SK | 2 | 0 |
| 3 | **890,00 €** | **03:17** | 5732 elektronika | **e-commerce** | **US** | **5** | **1** |
| 4 | 7,20 € | 17:02 | 5814 fastfood | bezkontakt | SK | 1 | 0 |
| 5 | **1 250,00 €** | **02:58** | 5944 klenoty | **e-commerce** | **CN** | **6** | **1** |
| 6 | 210,00 € | 19:30 | 5651 odevy | e-commerce | AT | 1 | 0 |
| 7 | 45,00 € | 09:05 | 5541 čerpacia st. | čip | SK | 1 | 0 |
| 8 | **640,00 €** | **03:41** | 7995 stávky | **e-commerce** | **MT** | **4** | **1** |
| 9 | 88,50 € | 14:20 | 5912 lekáreň | čip | SK | 1 | 0 |
| 10 | 15,00 € | 21:10 | 4111 doprava | bezkontakt | SK | 2 | 0 |

Riadok 6 je tu naschvál: **poctivý** nákup v rakúskom e-shope. Bez neho by stačilo pravidlo „e-commerce = podvod" a nemali by sme čo trénovať.

> V reálnej prevádzke je podvodov rádovo 0,05 – 0,3 % transakcií. Tu ich máme 30 %, aby sa dali čísla ukázať na papieri. Ako sa s reálnou nevyváženosťou pracuje, je v bode 3 na konci dokumentu.

## Krok 0 — nultý odhad a prvé rezíduá

Model ešte nič nevie, tak začne tým najhlúpejším možným odhadom: **priemerom cieľovej premennej**. Tri podvody z desiatich, teda pre **každý** riadok rovnako:

```text
p₀ = 0,30      → každá transakcia dostane 30 % pravdepodobnosť podvodu
```

A teraz to hlavné slovo celého dokumentu:

> **Rezíduum = skutočnosť − to, čo model práve teraz predpovedá.**
> Je to **nedoplatok modelu** — koľko mu ešte chýba do pravdy. Nie „ako veľmi sa mýli" v absolútnej hodnote, ale **so znamienkom a v jednotkách cieľa**: kladné rezíduum znamená „prihoď", záporné „uber".

Pre náš nultý model:

| riadok | y | predpoveď p₀ | **rezíduum r = y − p₀** | čo to hovorí |
|---|---|---|---|---|
| 3, 5, 8 (podvody) | 1 | 0,30 | **+0,70** | „toto bol podvod a ty si dal iba 30 % — prihoď 0,70" |
| 1, 2, 4, 6, 7, 9, 10 | 0 | 0,30 | **−0,30** | „toto bolo v poriadku a ty strašíš 30 % — uber 0,30" |

Všimnite si dve veci. Po prvé, súčet rezíduí je nula: 3 × 0,70 − 7 × 0,30 = 0. Presne to znamená, že priemer je najlepší možný odhad, keď o riadkoch nič iné nevieme. Po druhé, **rezíduá sú teraz nový cieľ**. Druhý model sa už nebude učiť „podvod / nie podvod", ale bude sa učiť predpovedať tieto čísla: +0,70 a −0,30.

## Krok 1 — prvý strom sa učí rezíduá

XGBoost postaví plytký strom a hľadá otázku (split), ktorá rozdelí riadky tak, aby v každej vetve boli **podobné rezíduá**. Skúsi rôzne prahy na všetkých stĺpcoch — `DE4 > 500 €`, `DE7 < 05:00`, `DE22 = e-commerce`, `tx/60 min ≥ 3` — a vyberie ten, ktorý najviac zníži chybu. Nech vyhrá:

```text
                 DE22 = e-commerce ?
                 /                 \
              áno                   nie
        riadky 3,5,6,8          riadky 1,2,4,7,9,10
        rezíduá:               rezíduá:
        +0,70 +0,70            −0,30 −0,30 −0,30
        −0,30 +0,70            −0,30 −0,30 −0,30
```

Aké číslo dá strom do listu? **Nie priemer rezíduí**, ako by človek čakal, ale trochu opatrnejšiu hodnotu — XGBoost delí súčet rezíduí súčtom „istoty" modelu plus regularizačným členom λ:

```text
             Σ rezíduí v liste
    w  =  ─────────────────────────         (λ = 1, štandardná regularizácia)
           Σ p·(1−p)  +  λ
```

Člen `p·(1−p)` hovorí, **aký citlivý je model v danom bode**: pri p = 0,5 je najväčší (model váha, malý posun veľa zmení), pri p blízko 0 alebo 1 je takmer nulový (model je si istý, treba silnejší tlak, aby sa pohol). A λ zámerne brzdí listy, v ktorých je málo riadkov — bez neho by sa strom vrhal na náhodné výnimky.

Dosadíme (p = 0,30 pre všetky riadky, teda p·(1−p) = 0,21):

| list | Σ rezíduí | Σ p(1−p) | **w = výstup listu** |
|---|---|---|---|
| e-commerce (4 riadky) | 0,70+0,70−0,30+0,70 = **+1,80** | 4 × 0,21 = 0,84 | 1,80 / (0,84+1) = **+0,978** |
| ostatné (6 riadkov) | 6 × (−0,30) = **−1,80** | 6 × 0,21 = 1,26 | −1,80 / (1,26+1) = **−0,797** |

Tento výstup sa **nepripočíta celý**. Vynásobí sa **learning rate** (`eta`, tu 0,3) — model spraví len tretinu navrhovaného kroku, aby sa jedným stromom neprestrelilo. (V reálnom nasadení býva `eta` ešte menšia, typicky 0,05; tu sme ju zvolili väčšiu, aby bol posun po dvoch stromoch vidno.) Po prevode späť na pravdepodobnosť dostaneme:

| riadok | p pred | p po 1. strome | nové rezíduum |
|---|---|---|---|
| 3, 5, 8 (podvod, e-com) | 0,300 | **0,365** | +0,635 |
| 6 (poctivý, e-com) | 0,300 | **0,365** | **−0,365** |
| 1, 2, 4, 7, 9, 10 | 0,300 | **0,252** | −0,252 |

**Toto je tá najdôležitejšia tabuľka celého dokumentu.** Pozrite sa, čo sa stalo:

- Podvodom rezíduum **kleslo** z +0,70 na +0,635 — model sa priblížil, ale ešte zďaleka nedošiel.
- Poctivým offline transakciám kleslo z −0,30 na −0,252 — tiež zlepšenie.
- Ale riadku 6 rezíduum **narástlo** z −0,30 na −0,365. Prvý strom mu **uškodil**, lebo ho hodil do jedného vreca s podvodmi. A práve preto má teraz najväčšie rezíduum spomedzi poctivých riadkov — čo je zároveň **inštrukcia pre druhý strom**: „tu je moja najväčšia bolesť, poď to opraviť."

Takto boosting funguje: **rezíduá sú spôsob, akým si stromy medzi sebou odovzdávajú, čo ešte treba dorobiť.**

## Krok 2 — druhý strom opravuje, čo prvý pokazil

Druhý strom dostane presne tie nové rezíduá a hľadá split, ktorý ich rozdelí. Delenie podľa `DE22` by mu už nepomohlo — v e-commerce vetve sú teraz aj +0,635 aj −0,365, teda rezíduá s **opačným znamienkom**, a jeden list ich nedokáže obslúžiť naraz. Model musí siahnuť po inom stĺpci, ktorý riadok 6 odlíši od podvodov. Ponúka sa **rýchlosť míňania**:

```text
              tx/60 min ≥ 4 ?
              /            \
           áno              nie
     riadky 3,5,8      riadky 1,2,4,6,7,9,10
     rezíduá:          rezíduá:
     +0,635 ×3         −0,365 (r.6), −0,252 ×6
```

| list | Σ rezíduí | Σ p(1−p) | w | príspevok (× 0,3) |
|---|---|---|---|---|
| tx/60 min ≥ 4 | +1,905 | 0,695 | **+1,124** | +0,337 |
| ostatné | −1,879 | 1,364 | **−0,795** | −0,238 |

A výsledok po dvoch stromoch:

| riadok | p₀ | po 1. strome | **po 2. strome** | rezíduum |
|---|---|---|---|---|
| 3, 5, 8 (podvody) | 0,300 | 0,365 | **0,446** | +0,554 |
| 6 (poctivý e-shop) | 0,300 | 0,365 | **0,312** | −0,312 |
| 1, 2, 4, 7, 9, 10 | 0,300 | 0,252 | **0,210** | −0,210 |

Riadok 6 je zachránený — druhý strom mu zobral, čo mu prvý neprávom pridal, a jeho rezíduum kleslo pod pôvodnú hodnotu. Podvody idú hore, poctivé transakcie dole.

## Ako to pokračuje

| po koľkých stromoch | p (podvody 3,5,8) | p (poctivý e-shop, r. 6) | p (bežné offline) |
|---|---|---|---|
| 0 | 0,300 | 0,300 | 0,300 |
| 1 | 0,365 | 0,365 | 0,252 |
| 2 | 0,446 | 0,312 | 0,210 |
| 10 | 0,76 | 0,29 | 0,07 |
| 200 | 0,99 | 0,03 | 0,001 |

Každý ďalší strom sa pýta na niečo, čo tie pred ním nedokázali rozlíšiť — MCC 7995 (stávkové kancelárie), nočnú hodinu v DE7, nesúlad krajiny obchodníka s krajinou vydania karty, sumu vysoko nad zvykom karty. Rezíduá sa zmenšujú, kroky sú čoraz jemnejšie a tréning sa zastaví (**early stopping**) vo chvíli, keď sa chyba na validačných dátach prestane zlepšovať — to je moment, keď by ďalšie stromy už len dolaďovali šum.

**Finálna predpoveď nie je hlasovanie, ale súčet:** východiskový odhad + 0,3 × (príspevok stromu 1) + 0,3 × (príspevok stromu 2) + … Preto sa boostingu hovorí *aditívny* model.

## Prečo sa tomu hovorí *gradient* boosting

Krátka, ale užitočná poznámka. Model neminimalizuje rezíduá „lebo to tak vyzerá logicky" — rezíduum je presne **záporný gradient chybovej funkcie**. Pri klasifikácii s log-loss vyjde derivácia chyby podľa predpovede rovná `p − y`, takže záporný gradient je `y − p`, teda **presne to, čo sme celý čas počítali**. Pri regresii so štvorcovou chybou vyjde `y − ŷ`, teda „skutočná cena mínus odhad" z príkladu s bytom.

Gradient boosting je teda **gradientný zostup, kde krokom nie je úprava čísla, ale pridanie celého stromu.** Rovnaká myšlienka ako pri trénovaní neurónových sietí (pozri [01-adam-optimalizator.md](../03-ucenie/01-adam-optimalizator.md)), len parametrom je model sám.

## Čo z toho plynie pre prax na ISO 8583 dátach

1. **Surové DE polia nestačia.** Najsilnejšie príznaky sú **odvodené** — počet transakcií kartou za 1 h / 24 h, pomer sumy k priemernej sume karty za 90 dní, čas od predchádzajúcej transakcie, počet rôznych krajín za deň, či MCC karta ešte nikdy nepoužila. Práve stĺpec `tx/60 min` zachránil náš druhý strom. Bez feature engineeringu nepomôže ani najlepší model.
2. **Vysoká kardinalita.** MCC má stovky hodnôt, DE43 desiatky krajín. One-hot to rozfúkne; použite `enable_categorical=True` (XGBoost), CatBoost, alebo target encoding **počítaný len z trénovacieho okna**.
3. **Nevyváženosť tried.** Pri 0,1 % podvodov nastavte `scale_pos_weight` a **nesledujte accuracy** — model „všetko je v poriadku" má 99,9 %. Sledujte **PR-AUC** a recall pri prevádzkovo únosnej miere falošných poplachov.
4. **Delenie dát podľa času, nie náhodne.** Náhodné rozdelenie by dalo model, ktorý sa učí z budúcnosti (*data leakage*) — trénujte na januári až marci, testujte na apríli. Aj príznaky musia byť počítané len z toho, čo bolo v momente autorizácie známe.
5. **Latencia.** Autorizácia má rozpočet rádovo 100 ms. XGBoost s 300 plytkými stromami zvládne predikciu za jednotky milisekúnd na CPU — ďalší dôvod, prečo je v platobnej infraštruktúre populárnejší než neurónová sieť.
6. **Vysvetliteľnosť je regulačná požiadavka.** Ak transakciu zamietnete, musíte vedieť prečo. **SHAP** rozloží skóre na príspevky jednotlivých polí: „0,46 z 0,71 pridalo `tx/60 min = 6`, 0,18 nesúlad krajiny, 0,07 nočná hodina."

## Kód

```python
import xgboost as xgb
from sklearn.metrics import average_precision_score

model = xgb.XGBClassifier(
    n_estimators=2000,      # horný strop; early stopping ho zvyčajne nedosiahne
    max_depth=5,            # plytké stromy — každý je zámerne slabý model
    learning_rate=0.05,     # menší krok = viac stromov, ale lepšia generalizácia
    reg_lambda=1.0,         # λ z výpočtu listu vyššie
    scale_pos_weight=300,   # ~1 podvod na 300 poctivých transakcií
    eval_metric="aucpr",    # PR-AUC, nie accuracy
    early_stopping_rounds=50,
    enable_categorical=True,
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])   # valid = neskorší časový úsek!

print(average_precision_score(y_test, model.predict_proba(X_test)[:, 1]))

# prečo model zamietol konkrétnu transakciu
import shap
shap.plots.waterfall(shap.TreeExplainer(model)(X_test.iloc[[0]])[0])
```

---

## Kontrolné otázky

1. Čo je **rezíduum** a prečo sa druhý strom v boostingu učí niečo iné než prvý?
2. Vysvetlite, prečo poctivému e-shopovému nákupu (riadok 6) rezíduum po prvom strome **narástlo** a čo to znamenalo pre druhý strom.
3. Prečo sa do listu nedáva priemer rezíduí, ale hodnota delená členom `Σ p(1−p) + λ`? Čo robí každý z tých dvoch členov?
4. Prečo je rezíduum `y − p` presne to, čo sa v gradient boostingu má minimalizovať?
5. Prečo sa transakčné dáta nesmú deliť na train/test náhodne, ale podľa času?

---

### Súvisiace dokumenty

- [02-random-forest-a-xgboost.md](02-random-forest-a-xgboost.md) — teória, na ktorú tento príklad nadväzuje
- [04-metriky.md](../01-prehlad/04-metriky.md) — PR-AUC, precision/recall pri nevyvážených triedach
- [01-adam-optimalizator.md](../03-ucenie/01-adam-optimalizator.md) — tá istá myšlienka gradientu, len pri neurónových sieťach

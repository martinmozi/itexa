# XGBoost v praxi — tréning a inferencia krok za krokom

> **Poradie čítania:** ← [Čo sa pri učení pokazí](02-problemy-pri-uceni.md) · **lekcia 3 — praktická časť** · [Transformery a attention](../04-llm/01-transformer-siete.md) →

> **Cieľ dokumentu:** previesť celým životným cyklom modelu na tabuľkových dátach — od načítania CSV cez tréning s early stoppingom a ladenie hyperparametrov až po uloženie modelu a jeho nasadenie do inferencie (dávkovej aj online). Uprostred (sekcia 3) je **úplný algoritmus pod kapotou** aj s referenčnou implementáciou v NumPy, takže si XGBoost viete napísať sami — a potom už viete, čo presne robí každý hyperparameter.

Na mechaniku boostingu (rezíduá, výstup listu, learning rate) nadväzujeme z [03-xgboost-priklad-iso8583.md](../02-typy-modelov/03-xgboost-priklad-iso8583.md) — tam sme počítali rukou na papieri, tu to isté odvodíme všeobecne a spustíme v kóde.

Príklady pokračujú v tej istej úlohe ako podrobný príklad v lekcii 2: **detekcia podvodných kartových transakcií** (ISO 8583). Je to typická tabuľková úloha s nevyváženými triedami a časovou osou — teda presne tá, kde sa dá pokaziť najviac vecí.

**Inštalácia** (verzie 2.x a 3.x sa v tomto dokumente správajú rovnako):

```bash
uv add xgboost scikit-learn pandas pyarrow    # bez uv:  pip install xgboost scikit-learn pandas pyarrow
python -c "import xgboost as xgb; print(xgb.__version__)"
```

---

## 1. Dáta — čo XGBoost očakáva a čo nie

Pekné na stromoch je, koľko prípravy **netreba**:

| Úkon | Treba pri neurónovej sieti | Treba pri XGBoost |
|---|---|---|
| škálovanie / normalizácia stĺpcov | **áno** (inak sa sieť učí zle) | **nie** — strom sa pýta „je hodnota > prah", škála je mu jedno |
| doplnenie chýbajúcich hodnôt | áno | **nie** — `NaN` je platný vstup, strom sa naučí, kam ich posielať |
| one-hot kódovanie kategórií | áno | **nie**, ak použijete `enable_categorical=True` |
| odstránenie odľahlých hodnôt | často | zriedka — strom ich izoluje do vlastného listu |
| monotónne transformácie (log sumy) | pomáhajú | **nemajú žiadny efekt** — poradie hodnôt sa nemení |

Čo naopak treba **vždy**: aby každý riadok bol jedno pozorovanie, aby stĺpce boli číselné alebo `category`, a aby ste vedeli, ktorý stĺpec je cieľ.

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("transakcie.parquet")   # ~2 mil. riadkov, ISO 8583 polia + odvodené príznaky

CIEL = "je_podvod"
KATEGORICKE = ["de18_mcc", "de22_vstup", "de43_krajina", "de49_mena"]
CISELNE = ["de4_suma", "hodina", "den_v_tyzdni", "tx_60min", "tx_24h",
           "pomer_k_priemeru_90d", "min_od_predchadzajucej", "pocet_krajin_24h"]

# kategórie musia mať dtype 'category', inak ich XGBoost odmietne
for stlpec in KATEGORICKE:
    df[stlpec] = df[stlpec].astype("category")

X = df[CISELNE + KATEGORICKE]
y = df[CIEL].astype(int)
```

> **Odvodené príznaky sú dôležitejšie než model.** Surové DE polia samy osebe veľa neprezradia — silu majú až príznaky typu `tx_60min` (koľko transakcií kartou za poslednú hodinu) alebo `pomer_k_priemeru_90d`. Platí pravidlo: **každý príznak musí byť počítaný len z toho, čo bolo v okamihu autorizácie známe.** Ak do `pomer_k_priemeru_90d` prepustíte priemer počítaný cez celé dáta vrátane budúcnosti, model bude na testovacích dátach vynikajúci a v prevádzke bezcenný. To je **data leakage** a je to najčastejšia príčina modelu, ktorý „fungoval a potom nie".

### Delenie dát — podľa času, nie náhodne

```python
df = df.sort_values("cas_transakcie")

# train: prvých 70 % času · valid: ďalších 15 % · test: posledných 15 %
n = len(df)
i_tr, i_va = int(0.70 * n), int(0.85 * n)

X_train, y_train = X.iloc[:i_tr],      y.iloc[:i_tr]
X_valid, y_valid = X.iloc[i_tr:i_va],  y.iloc[i_tr:i_va]
X_test,  y_test  = X.iloc[i_va:],      y.iloc[i_va:]

print(f"train {len(X_train):,} | valid {len(X_valid):,} | test {len(X_test):,}")
print("podiel podvodov:", y_train.mean().round(5), y_test.mean().round(5))
```

Tri množiny, tri rozdielne úlohy — a je kritické ich nepomiešať:

- **train** — z neho sa stavajú stromy,
- **valid** — na ňom sa rozhoduje, **kedy prestať** (early stopping) a ktoré hyperparametre sú lepšie,
- **test** — siahnete naň **raz**, na konci, keď už nič nemeníte. Každé rozhodnutie urobené podľa testu z neho robí ďalšiu validačnú množinu a odhad presnosti sa stáva optimistickým.

---

## 2. Dve API, ktoré XGBoost ponúka

Toto býva prvý zdroj zmätku: dokumentácia aj príklady na internete miešajú dva rôzne štýly volania tej istej knižnice.

| | **scikit-learn API** | **natívne API** |
|---|---|---|
| trieda / funkcia | `XGBClassifier`, `XGBRegressor` | `xgb.train(...)` → objekt `Booster` |
| dáta | priamo `DataFrame` / `ndarray` | `xgb.DMatrix` (vlastný formát) |
| hyperparametre | argumenty konštruktora | slovník `params` |
| zapadá do | `Pipeline`, `GridSearchCV`, `cross_val_score` | vlastné slučky, `xgb.cv`, low-level kontrola |
| odporúčanie | **začnite tu** | keď potrebujete `xgb.cv`, custom loss alebo najrýchlejšiu inferenciu |

Sú to len dva obaly nad tým istým jadrom — a kedykoľvek sa dá prejsť z jedného do druhého cez `model.get_booster()`. Nižšie ukazujeme oba; zvyšok dokumentu používa sklearn API, pretože je čitateľnejšie.

```python
import xgboost as xgb

# --- sklearn API ---------------------------------------------------------
model = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                          enable_categorical=True)
model.fit(X_train, y_train)
p = model.predict_proba(X_valid)[:, 1]

# --- natívne API (to isté) ----------------------------------------------
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
booster = xgb.train({"max_depth": 5, "eta": 0.05, "objective": "binary:logistic"},
                    dtrain, num_boost_round=500)
p = booster.predict(dvalid)
```

> **Pozor na názvoslovie:** ten istý hyperparameter má v každom API iné meno. `learning_rate` = `eta`, `n_estimators` = `num_boost_round`, `reg_lambda` = `lambda`, `reg_alpha` = `alpha`. Keď kopírujete kód z internetu, najprv zistite, v ktorom svete je napísaný.

---

## 3. Pod kapotou — algoritmus, ktorý si viete naprogramovať

Dokument [03-xgboost-priklad-iso8583.md](../02-typy-modelov/03-xgboost-priklad-iso8583.md) ukázal mechaniku na papieri: rezíduá, výstup listu, learning rate. Táto sekcia je o stupeň nižšie — je to **úplná špecifikácia**, podľa ktorej si viete XGBoost napísať sami, v rovnakom duchu ako [01-adam-optimalizator.md](01-adam-optimalizator.md) pri Adamovi. Na konci je referenčná implementácia v NumPy (~90 riadkov) a test, ktorý ju porovná so skutočným XGBoostom.

Prečo to vedieť, keď knižnica existuje? Lebo **každý hyperparameter zo sekcie 5 je jedno písmeno v týchto vzorcoch** — a kto videl vzorec, nemusí ladenie hádať.

### 3.1 Čo sa vlastne minimalizuje

Model je **súčet stromov**, nie ich priemer:

```text
    F₀(x)  = základný odhad (logit priemeru cieľa)
    Fₘ(x)  = Fₘ₋₁(x) + η · fₘ(x)          ... η = learning rate, fₘ = m-tý strom
```

Hľadá sa taká postupnosť stromov, ktorá minimalizuje

```text
    L  =  Σᵢ l(yᵢ, ŷᵢ)   +   Σₖ Ω(fₖ)          kde   Ω(f) = γ·T + ½·λ·Σⱼ wⱼ²
```

Prvý člen je chyba (log-loss pri klasifikácii, štvorcová chyba pri regresii), druhý je **regularizácia samotného stromu**: `T` je počet listov a `wⱼ` hodnoty v nich. Práve tento druhý člen odlišuje XGBoost od starších implementácií gradient boostingu — cena za zložitosť stromu je priamo v optimalizovanej funkcii, nie v dodatočnom orezávaní.

### 3.2 Taylorov rozvoj druhého rádu — odkiaľ sa berú `g` a `h`

Keď staviame `m`-tý strom, predchádzajúce sú už pevné. Rozvinieme chybu okolo aktuálnej predpovede do **druhého rádu**:

```text
    L⁽ᵐ⁾  ≈  Σᵢ [ gᵢ · f(xᵢ)  +  ½ · hᵢ · f(xᵢ)² ]  +  Ω(f)  +  konštanta

           ∂l(yᵢ, ŷ)                    ∂²l(yᵢ, ŷ)
    gᵢ  =  ─────────  (gradient)  hᵢ  =  ──────────  (hessián, „zakrivenie")
             ∂ŷ                            ∂ŷ²
```

To je celá matematika, ktorú XGBoost potrebuje: **z chybovej funkcie mu stačia dve čísla na riadok.** Vďaka tomu je algoritmus univerzálny — zmeníte úlohu, zmeníte len `g` a `h`:

| Úloha | `objective` | predpoveď | `gᵢ` | `hᵢ` |
|---|---|---|---|---|
| regresia, štvorcová chyba | `reg:squarederror` | `ŷ` priamo | `ŷᵢ − yᵢ` | `1` |
| binárna klasifikácia | `binary:logistic` | `p = σ(ŷ)` | `pᵢ − yᵢ` | `pᵢ(1−pᵢ)` |
| počty udalostí | `count:poisson` | `μ = e^ŷ` | `μᵢ − yᵢ` | `μᵢ` |
| vlastná chyba | vlastná funkcia | — | vy dodáte | vy dodáte |

Všimnite si riadok s regresiou: `−g = y − ŷ`, teda **presne rezíduum** z príkladu s bytom. Pri klasifikácii je `−g = y − p`, presne „nedoplatok modelu" z ISO 8583 dokumentu. Rezíduum nie je samostatná myšlienka — je to **záporný gradient**, a preto sa tomu hovorí *gradient* boosting. XGBoost ide o krok ďalej a používa aj druhú deriváciu; presnejší názov by bol *Newtonov* boosting.

### 3.3 Optimálna hodnota listu — a odkiaľ je vzorec `−G/(H+λ)`

Zafixujme na chvíľu **tvar** stromu (kam ktorý riadok padne) a pýtajme sa len, aké číslo dať do listov. Označme pre list `j`:

```text
    Gⱼ = Σ gᵢ ,   Hⱼ = Σ hᵢ        (súčty cez riadky, ktoré padli do listu j)
```

Keďže všetky riadky v liste dostanú tú istú hodnotu `wⱼ`, cieľová funkcia sa rozpadne na nezávislé kvadratické rovnice — jednu na list:

```text
    L(w)  =  Σⱼ [ Gⱼ·wⱼ  +  ½·(Hⱼ + λ)·wⱼ² ]  +  γ·T
```

Kvadratická funkcia `a·w + ½·b·w²` má minimum v `w = −a/b`, takže:

```text
              Gⱼ                                    1     Gⱼ²
    wⱼ*  =  − ──────           a po dosadení:  L* = −─ · Σ ────── + γ·T
             Hⱼ + λ                                  2   Hⱼ + λ
```

`L*` sa volá **skóre štruktúry** (*structure score*) — číslo, ktoré hovorí, aký dobrý je daný tvar stromu. Čím nižšie, tým lepšie.

A teraz kontrola, že sedí s tým, čo sme počítali ručne v lekcii 2: pri log-loss je `−G = Σ(y − p) = Σ rezíduí` a `H = Σ p(1−p)`, takže

```text
    w  =  Σ rezíduí / ( Σ p(1−p) + λ )
```

— presne vzorec z [dokumentu o ISO 8583](../02-typy-modelov/03-xgboost-priklad-iso8583.md). Teraz už viete, prečo je v menovateli práve `p(1−p)`: je to **druhá derivácia log-loss**, teda miera, ako veľmi sa chyba zmení, keď sa predpoveď pohne. Kde je model neistý (`p ≈ 0,5`), je zakrivenie veľké a krok sa tlmí; kde je istý, je zakrivenie takmer nulové a treba silnejší posun.

### 3.4 Zisk zo splitu — jediné kritérium, podľa ktorého strom rastie

Rozdelenie uzla má zmysel vtedy, keď **zníži skóre štruktúry**. Rozdiel pred a po rozdelení je:

```text
             1   ⎡  G_L²      G_R²      (G_L+G_R)²  ⎤
    Zisk  =  ─ · ⎢ ────── + ────── − ─────────────── ⎥  −  γ
             2   ⎣ H_L+λ    H_R+λ    H_L+H_R+λ      ⎦
```

Tri veci sú v tomto jedinom vzorci a stoja za to ich vysloviť nahlas:

- **λ (`reg_lambda`)** je v menovateli. Čím väčšia, tým menší zisk zo splitov v malých uzloch (kde je `H` malé) — a tým aj menšie hodnoty listov. Je to tlmič, nie zákaz.
- **γ (`gamma`)** je paušálny **poplatok za nový list**. Split, ktorý neprinesie zisk aspoň `γ`, sa neuskutoční. To je pre-pruning zabudovaný do kritéria.
- **`min_child_weight`** je minimálne `H` v dieťati. Pri log-loss je `h = p(1−p) ≤ 0,25`, takže `min_child_weight = 1` znamená zhruba „aspoň štyri riadky, o ktorých si model nie je istý". Nie je to počet riadkov — je to **efektívny počet**, vážený neistotou. Preto v silne nevyváženej úlohe (kde `p` je blízko nuly) tá istá hodnota zodpovedá oveľa väčšiemu počtu riadkov.

Ak je najlepší zisk `≤ 0`, uzol zostáva listom. Toto je celé rozhodovanie o raste stromu.

### 3.5 Hľadanie najlepšieho splitu — presný algoritmus

Kandidátov je konečne veľa: pre každý stĺpec a každý prah medzi dvoma susednými hodnotami jeden. Naivné počítanie by pre každý kandidát sčítalo celý uzol, čo je `O(n²)`. Trik je **zoradiť a ísť prefixovými súčtami**:

```text
pre každý stĺpec j:
    zoraď riadky uzla podľa hodnoty x[:, j]
    G_L ← 0 ;  H_L ← 0
    pre k = 1 … n−1:                       # deliaci bod za k-tym riadkom
        G_L += g[k] ;  H_L += h[k]
        G_R = G − G_L ;  H_R = H − H_L
        ak x[k] == x[k+1]: pokračuj        # medzi rovnakými hodnotami sa nedelí
        ak H_L < min_child_weight alebo H_R < min_child_weight: pokračuj
        spočítaj Zisk a zapamätaj si najlepší
```

Jeden prechod na stĺpec, teda `O(n·d)` na uzol po zoradení. To je **exact greedy** algoritmus (`tree_method="exact"`) — prejde všetky možné prahy a je presný.

### 3.6 Histogramová metóda — prečo je `hist` dnes predvolená

Pri miliónoch riadkov je zoraďovanie v každom uzle drahé. Metóda `hist` to obchádza:

1. **Raz na začiatku** sa každý stĺpec rozdelí na `max_bin` (štandardne 256) košov podľa kvantilov. Každá hodnota sa nahradí číslom koša — z `float` sa stane `uint8`.
2. V uzle sa spraví **histogram**: pre každý kôš sa nasčíta `G` a `H`. To je jeden prechod `O(n)` bez zoraďovania.
3. Kandidátov na split je potom len 255 na stĺpec, nie `n−1`.

K tomu ešte jeden veľmi účinný trik: **histogram súrodenca sa nepočíta, ale odčíta.** Ak poznáme histogram rodiča a jedného dieťaťa, druhé dieťa je ich rozdiel — takže sa vždy počíta len ten menší z dvojice a práca klesne na polovicu.

| | `exact` | `hist` |
|---|---|---|
| kandidáti na split | všetky prahy | `max_bin − 1` |
| zložitosť na uzol | `O(n·d)` + zoradenie | `O(n·d)` bez zoradenia, potom `O(max_bin·d)` |
| pamäť | pôvodné hodnoty | `uint8` kódy — až 8× menej |
| presnosť | presná | zaokrúhlená na koše (v praxi nerozlíšiteľné) |

`max_bin` je teda výmena presnosti za rýchlosť: 512 košov býva o kúsok presnejšie, 128 rýchlejšie.

### 3.7 Chýbajúce hodnoty — naučený smer, nie imputácia

Toto je jeden z dôvodov, prečo XGBoost na reálnych dátach vyhráva. Chýbajúce hodnoty sa **nedopĺňajú**; pri každom splite sa vyskúšajú obe možnosti:

```text
    variant A:  chýbajúce idú doľava   → spočítaj zisk
    variant B:  chýbajúce idú doprava  → spočítaj zisk
    ulož lepší z nich ako „default direction" uzla
```

V uzle sa teda uloží nielen `(stĺpec, prah)`, ale aj **smer pre chýbajúce**. Model sa tým naučí, čo chýbajúca hodnota v danom kontexte znamená — pri transakciách napríklad „chýbajúce PSČ držiteľa karty sa správa ako zahraničná e-commerce".

Existuje ešte jeden kandidát, na ktorý sa pri vlastnej implementácii ľahko zabudne (a ktorý XGBoost skutočne zvažuje): split, ktorý **oddelí chýbajúce od nechýbajúcich** — prah nad maximom stĺpca, všetky prítomné hodnoty na jednu stranu, chýbajúce na druhú. Ak je „nevyplnené" samo osebe silným signálom, je to najlepší možný split. Bez neho sa strom pri dátach s `NaN` rozíde od skutočného XGBoostu.

### 3.8 Referenčná implementácia (NumPy)

Všetko vyššie zhrnuté do funkčného kódu. Podporuje log-loss, `max_depth`, `eta`, `lambda`, `gamma`, `min_child_weight` aj chýbajúce hodnoty:

```python
import numpy as np

class Uzol:
    __slots__ = ("stlpec", "prah", "vlavo", "vpravo", "chybajuce_vlavo", "w")
    def __init__(self):
        self.stlpec = None; self.prah = None
        self.vlavo = None;  self.vpravo = None
        self.chybajuce_vlavo = True; self.w = 0.0

def _skore(G, H, lam):
    return (G * G) / (H + lam)               # člen G²/(H+λ) zo skóre štruktúry

def najdi_split(X, g, h, lam, gamma, min_child_weight):
    G, H = g.sum(), h.sum()
    zaklad = _skore(G, H, lam)
    najlepsi = {"zisk": 0.0}

    def skus(GL, HL, stlpec, prah, chybajuce_vlavo):
        nonlocal najlepsi
        GR, HR = G - GL, H - HL
        if HL < min_child_weight or HR < min_child_weight:
            return
        zisk = 0.5 * (_skore(GL, HL, lam) + _skore(GR, HR, lam) - zaklad) - gamma
        if zisk > najlepsi["zisk"] + 1e-12:
            najlepsi = {"zisk": float(zisk), "stlpec": stlpec,
                        "prah": float(prah), "chybajuce_vlavo": chybajuce_vlavo}

    for j in range(X.shape[1]):
        x = X[:, j]
        chyba = np.isnan(x)
        G_ch, H_ch = g[chyba].sum(), h[chyba].sum()
        xs_all, gs_all, hs_all = x[~chyba], g[~chyba], h[~chyba]
        if xs_all.size == 0:
            continue
        poradie = np.argsort(xs_all, kind="mergesort")
        xs, gs, hs = xs_all[poradie], gs_all[poradie], hs_all[poradie]
        Gp, Hp = np.cumsum(gs), np.cumsum(hs)          # prefixové súčty

        # (a) bežné prahy medzi dvoma rôznymi hodnotami, obidva smery pre chýbajúce
        if xs.size >= 2:
            for k in np.flatnonzero(xs[:-1] < xs[1:]):
                prah = 0.5 * (xs[k] + xs[k + 1])
                skus(Gp[k] + G_ch, Hp[k] + H_ch, j, prah, True)
                skus(Gp[k],        Hp[k],        j, prah, False)

        # (b) split „chýba / nechýba" — len ak v uzle nejaké chýbajúce sú
        if H_ch > 0:
            skus(Gp[-1], Hp[-1], j, xs[-1] + 1.0, False)
            skus(G_ch,   H_ch,   j, xs[0],        True)

    return najlepsi

def postav(X, g, h, hlbka, max_hlbka, lam, gamma, min_child_weight):
    uzol = Uzol()
    uzol.w = -g.sum() / (h.sum() + lam)                # w* = −G/(H+λ)
    if hlbka >= max_hlbka or len(g) < 2:
        return uzol
    s = najdi_split(X, g, h, lam, gamma, min_child_weight)
    if s["zisk"] <= 0:                                  # žiadny split sa neoplatí
        return uzol
    x = X[:, s["stlpec"]]
    ide_vlavo = np.where(np.isnan(x), s["chybajuce_vlavo"], x < s["prah"])
    uzol.stlpec, uzol.prah = s["stlpec"], s["prah"]
    uzol.chybajuce_vlavo = s["chybajuce_vlavo"]
    uzol.vlavo  = postav(X[ide_vlavo],  g[ide_vlavo],  h[ide_vlavo],
                         hlbka + 1, max_hlbka, lam, gamma, min_child_weight)
    uzol.vpravo = postav(X[~ide_vlavo], g[~ide_vlavo], h[~ide_vlavo],
                         hlbka + 1, max_hlbka, lam, gamma, min_child_weight)
    return uzol

def predikuj_strom(uzol, X):
    if uzol.stlpec is None:
        return np.full(len(X), uzol.w)
    x = X[:, uzol.stlpec]
    vlavo = np.where(np.isnan(x), uzol.chybajuce_vlavo, x < uzol.prah)
    out = np.empty(len(X))
    if vlavo.any():    out[vlavo]  = predikuj_strom(uzol.vlavo,  X[vlavo])
    if (~vlavo).any(): out[~vlavo] = predikuj_strom(uzol.vpravo, X[~vlavo])
    return out

def trenuj(X, y, n_stromov=100, eta=0.3, max_hlbka=3, lam=1.0, gamma=0.0,
           min_child_weight=1.0, base_score=0.5):
    F = np.full(len(y), np.log(base_score / (1 - base_score)), dtype=float)
    stromy = []
    for _ in range(n_stromov):
        p = 1.0 / (1.0 + np.exp(-F))          # aktuálna predpoveď
        g = p - y                              # gradient log-loss
        h = p * (1.0 - p)                      # hessián log-loss
        strom = postav(X, g, h, 0, max_hlbka, lam, gamma, min_child_weight)
        F += eta * predikuj_strom(strom, X)    # shrinkage: len η-tina kroku
        stromy.append(strom)
    return {"stromy": stromy, "eta": eta, "base_score": base_score}

def predikuj(model, X):
    F = np.full(len(X), np.log(model["base_score"] / (1 - model["base_score"])))
    for strom in model["stromy"]:
        F += model["eta"] * predikuj_strom(strom, X)
    return 1.0 / (1.0 + np.exp(-F))
```

Celá tréningová slučka má päť riadkov a stojí za to ich prečítať ešte raz: **spočítaj predpoveď → spočítaj `g` a `h` → postav strom na týchto dvoch číslach → pripočítaj η-tinu jeho výstupu → opakuj.** Nič viac v gradient boostingu nie je.

### 3.9 Kontrola správnosti — porovnanie so skutočným XGBoostom

Implementáciu, ktorá „vyzerá rozumne", treba overiť proti referencii. Pri rovnakých parametroch musia obe dávať **tie isté pravdepodobnosti**:

```python
import numpy as np, xgboost as xgb

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 5))
X[rng.random(X.shape) < 0.1] = np.nan                    # 10 % chýbajúcich hodnôt
logit = 1.5*np.nan_to_num(X[:,0]) - 2.0*np.nan_to_num(X[:,1])*np.nan_to_num(X[:,2]) + 0.5
y = (rng.random(len(X)) < 1/(1+np.exp(-logit))).astype(float)

moj = trenuj(X, y, n_stromov=30, eta=0.2, max_hlbka=4,
             lam=1.0, gamma=0.0, min_child_weight=1.0, base_score=0.5)
p_moj = predikuj(moj, X)

ref = xgb.XGBClassifier(n_estimators=30, learning_rate=0.2, max_depth=4,
                        reg_lambda=1.0, gamma=0.0, min_child_weight=1.0,
                        base_score=0.5, tree_method="exact",
                        subsample=1.0, colsample_bytree=1.0)
ref.fit(X, y)
p_ref = ref.predict_proba(X)[:, 1]

print("max |rozdiel| :", np.abs(p_moj - p_ref).max())    # 1.09e-07
print("stredný       :", np.abs(p_moj - p_ref).mean())   # 2.16e-08
```

Rozdiel rádu **10⁻⁷** je presne to, čo čakáme: XGBoost počíta vnútorne v `float32`, my vo `float64`. Keby sa implementácia v čomkoľvek podstatnom líšila, rozdiel by bol rádovo väčší — je to teda veľmi citlivý test.

Tri veci, ktoré treba pri porovnávaní nastaviť, inak sa zhoda nedostaví:

- **`base_score=0.5` explicitne.** XGBoost si od verzie 2.0 východiskový odhad počíta z dát; my sme ho zafixovali.
- **`tree_method="exact"`.** Predvolená `hist` binuje hodnoty do košov, takže prahy vyjdú inak.
- **`subsample=1.0`, `colsample_bytree=1.0`** — žiadna náhoda, inak nie je čo porovnávať.

A dve miesta, kde sa naša implementácia od XGBoostu úmyselne líši:

1. **γ.** My ho odpočítavame priamo od zisku (vzorec z pôvodného článku). XGBoost ho aplikuje ako **post-pruning**: strom najprv dorastie a až potom sa zdola nahor odstraňujú splity, ktorých `loss_chg` je menší než γ — pričom `loss_chg` počíta **bez tej polovice** vo vzorci. Dôsledok: naše `gamma` zodpovedá zhruba `gamma/2` v XGBooste a pri veľkých hodnotách sa stromy rozídu. Zhoda platí pre `gamma=0`.
2. **Remízy.** Keď dva splity dajú presne rovnaký zisk (stáva sa to v malých uzloch, typicky pri `max_depth ≥ 6` a `min_child_weight = 1`), obe implementácie si vyberú iný — rovnako dobrý — split a od toho miesta sa stromy líšia. Na reálnych dátach so spojitými hodnotami je to zriedkavé.

### 3.10 Čo skutočný XGBoost pridáva navyše

Deväťdesiat riadkov vyššie je jadro. Zvyšok knižnice je inžinierstvo — a stojí za to vedieť, čo tam pribudlo:

| Vlastnosť | Čo rieši |
|---|---|
| **weighted quantile sketch** | návrh kandidátov na prahy pri dátach, ktoré sa nezmestia do pamäte (`tree_method="approx"`), s váhami podľa `h` |
| **cache-aware prístup** | poradie čítania riadkov tak, aby sedelo na cache procesora — hlavný zdroj zrýchlenia oproti naivnej implementácii |
| **blokové ukladanie, out-of-core** | komprimované stĺpcové bloky na disku, tréning nad dátami väčšími než RAM |
| **paralelizmus** | stĺpce sa spracúvajú súbežne; na GPU celý histogram naraz (`device="cuda"`) |
| **`colsample_*`, `subsample`** | náhodný výber stĺpcov (na strom / úroveň / uzol) a riadkov — regularizácia navyše |
| **`grow_policy="lossguide"`** | rast po najlepšom liste namiesto po úrovniach (ako LightGBM), s `max_leaves` |
| **kategorické delenia** | `enable_categorical` — koše sa zoradia podľa `G/H` a delí sa na podmnožiny, bez one-hot |
| **monotónne a interakčné obmedzenia** | `monotone_constraints` vynúti „vyššia suma ⇒ nie nižšie skóre" — často regulačná požiadavka |
| **`pred_contribs`** | presné SHAP hodnoty priamo zo štruktúry stromov (sekcia 10) |

Ak si chcete implementáciu rozšíriť, najväčšiu hodnotu za najmenej práce dajú v tomto poradí: **histogramy** (rýchlosť), **`subsample`/`colsample`** (presnosť) a **early stopping** (všetko ostatné).

---

## 4. Prvý tréning — baseline s early stoppingom

Nezačínajte ladením. Začnite **rozumným východiskovým modelom**, ktorý dobehne, a až potom sa ho snažte poraziť.

```python
import xgboost as xgb

# pomer tried: koľko poctivých transakcií pripadá na jeden podvod
pomer = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {pomer:.0f}")        # napr. 420

model = xgb.XGBClassifier(
    # --- kedy sa zastaviť ---
    n_estimators=5000,              # horný strop, nie cieľ — early stopping ho nedosiahne
    learning_rate=0.05,             # malý krok; menšia hodnota = viac stromov, lepšia generalizácia
    early_stopping_rounds=100,      # 100 kôl bez zlepšenia na valid → koniec
    # --- tvar stromov ---
    max_depth=6,
    min_child_weight=5,             # list musí mať dosť „hmoty", inak vznikne na 2 riadkoch
    # --- náhodnosť (regularizácia) ---
    subsample=0.8,                  # každý strom vidí 80 % riadkov
    colsample_bytree=0.8,           # a 80 % stĺpcov
    # --- regularizácia ---
    reg_lambda=1.0,                 # λ z výpočtu výstupu listu
    gamma=0.0,                      # minimálny zisk, aby sa split vôbec urobil
    # --- úloha a metriky ---
    objective="binary:logistic",
    eval_metric=["aucpr", "logloss"],   # early stopping sleduje poslednú metriku v zozname
    scale_pos_weight=pomer,
    # --- technické ---
    enable_categorical=True,
    tree_method="hist",             # štandard; na GPU pridajte device="cuda"
    n_jobs=-1,
    random_state=42,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=50,                     # výpis každých 50 stromov
)

print("najlepšia iterácia:", model.best_iteration, "| skóre:", round(model.best_score, 5))
```

Výpis počas tréningu vyzerá takto a **oplatí sa mu rozumieť**:

```text
[0]    validation_0-aucpr:0.11243  validation_0-logloss:0.66934  validation_1-aucpr:0.09871  validation_1-logloss:0.67012
[50]   validation_0-aucpr:0.58120  validation_0-logloss:0.18442  validation_1-aucpr:0.51004  validation_1-logloss:0.19973
[100]  validation_0-aucpr:0.67934  validation_0-logloss:0.12006  validation_1-aucpr:0.58217  validation_1-logloss:0.14120
[300]  validation_0-aucpr:0.81220  validation_0-logloss:0.06553  validation_1-aucpr:0.62884  validation_1-logloss:0.12980
[420]  validation_0-aucpr:0.86004  validation_0-logloss:0.05120  validation_1-aucpr:0.62031  validation_1-logloss:0.13455
```

`validation_0` je prvá položka v `eval_set` (train), `validation_1` druhá (valid). Okolo iterácie 300 sa **nožnice otvárajú**: train sa ďalej zlepšuje, valid už nie — to je **preučenie** a presne v tom bode chceme skončiť. O to sa stará `early_stopping_rounds`: keď 100 kôl po sebe nepríde zlepšenie, tréning sa zastaví a `model.best_iteration` ukazuje na najlepšie miesto.

> **Dôležité:** model si stromy po `best_iteration` **nezmaže**, len si pamätá, kde bolo najlepšie. Sklearn API ich pri `predict`/`predict_proba` automaticky ignoruje. Pri natívnom `Booster` si to musíte vypýtať sami: `booster.predict(dmat, iteration_range=(0, booster.best_iteration + 1))`. Na tomto sa dá stratiť pár percent presnosti bez toho, aby ste to zbadali.

Krivku učenia si vytiahnete z modelu a vykreslíte — v notebooku je to prvá vec, ktorú chcete vidieť:

```python
import matplotlib.pyplot as plt

h = model.evals_result()
plt.plot(h["validation_0"]["aucpr"], label="train")
plt.plot(h["validation_1"]["aucpr"], label="valid")
plt.axvline(model.best_iteration, ls="--", c="gray", label="best_iteration")
plt.xlabel("počet stromov"); plt.ylabel("PR-AUC"); plt.legend(); plt.show()
```

---

## 5. Hyperparametre — čo naozaj ladiť a v akom poradí

XGBoost ich má desiatky, reálny vplyv má týchto osem:

| Parameter | Čo robí | Typický rozsah | Kam ísť pri **preučení** |
|---|---|---|---|
| `learning_rate` (`eta`) | veľkosť kroku každého stromu | 0,01 – 0,3 | **znížiť** (a nechať viac stromov) |
| `n_estimators` | počet stromov | strop 2000–10000 + early stopping | rieši early stopping |
| `max_depth` | hĺbka stromu = zložitosť interakcií | 3 – 10 | **znížiť** |
| `min_child_weight` | minimálne `H = Σh` v liste, teda **efektívny** počet riadkov (sekcia 3.4) | 1 – 20 | **zvýšiť** |
| `subsample` | podiel riadkov na strom | 0,5 – 1,0 | **znížiť** |
| `colsample_bytree` | podiel stĺpcov na strom | 0,5 – 1,0 | **znížiť** |
| `reg_lambda` | L2 regularizácia výstupu listu — λ v menovateli `w* = −G/(H+λ)` (sekcia 3.3) | 0,5 – 10 | **zvýšiť** |
| `gamma` | minimálny zisk potrebný na split — poplatok za nový list vo vzorci zisku (sekcia 3.4) | 0 – 5 | **zvýšiť** |

Vzťah medzi `learning_rate` a počtom stromov je základný: **polovičný learning rate ≈ dvojnásobok stromov** pri zhruba rovnakej alebo mierne lepšej presnosti a dvojnásobnom čase tréningu. Preto sa ladí s väčším `eta` (0,1) a finálny model sa natrénuje s menším (0,03).

Praktický postup, ktorý ušetrí hodiny:

1. **Nechajte `eta = 0,1`** a early stopping nech určí počet stromov. Máte baseline.
2. **Zložitosť stromu:** skúšajte `max_depth` ∈ {4, 6, 8} × `min_child_weight` ∈ {1, 5, 20}. Tu býva najväčší skok.
3. **Náhodnosť:** `subsample` a `colsample_bytree` ∈ {0,6; 0,8; 1,0}.
4. **Regularizácia:** `reg_lambda` a `gamma`, ak model stále preučuje.
5. **Až nakoniec** znížte `eta` na 0,03 a nechajte model dobehnúť s väčším počtom stromov.

Automatizované hľadanie sa oplatí — ale nie mriežkou (*grid search*), ktorá premrhá čas na nezaujímavých kombináciách. Náhodné hľadanie alebo Optuna nájdu lepšie nastavenie rýchlejšie:

```python
import optuna

def cielova_funkcia(trial):
    params = dict(
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 30, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 20, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
    )
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.1,
                          early_stopping_rounds=50, eval_metric="aucpr",
                          scale_pos_weight=pomer, enable_categorical=True,
                          tree_method="hist", n_jobs=-1, random_state=42, **params)
    m.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return m.best_score          # PR-AUC na validácii — maximalizujeme

study = optuna.create_study(direction="maximize")
study.optimize(cielova_funkcia, n_trials=50)
print(study.best_params, round(study.best_value, 4))
```

> **Nevyváženosť tried** sa rieši parametrom `scale_pos_weight` (váha pozitívnej triedy) — ten však **posúva pravdepodobnosti**: model prestane vracať kalibrované čísla a začne vracať skóre. Ak potrebujete, aby „0,03" naozaj znamenalo 3 % riziko (napr. pri výpočte očakávanej straty), nechajte `scale_pos_weight = 1` a radšej posuňte **rozhodovací prah** (sekcia 7) alebo model dodatočne kalibrujte (`sklearn.calibration.CalibratedClassifierCV`).

---

## 6. Krížová validácia — keď je jedna validačná množina málo

Jedno rozdelenie na train/valid je náchylné na náhodu: vyjde vám, že `max_depth=8` je lepšia než `6`, a pritom ste len trafili šťastnú vzorku. Pri menších dátach (do stoviek tisíc riadkov) použite krížovú validáciu.

Natívne API má na to hotovú funkciu, ktorá zároveň nájde optimálny počet stromov:

```python
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

vysledok = xgb.cv(
    params={"objective": "binary:logistic", "eval_metric": "aucpr",
            "max_depth": 6, "eta": 0.05, "subsample": 0.8,
            "scale_pos_weight": pomer, "tree_method": "hist"},
    dtrain=dtrain,
    num_boost_round=3000,
    nfold=5,
    stratified=True,             # každý fold má rovnaký podiel podvodov
    early_stopping_rounds=100,
    seed=42,
)
print(vysledok.tail(3))                    # stĺpce: train/test-aucpr-mean, -std
print("optimálny počet stromov:", len(vysledok))
```

Pri **časových dátach** však náhodné foldy klamú — model by sa učil z budúcnosti. Použite rozširujúce sa okno:

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score

skore = []
for train_idx, valid_idx in TimeSeriesSplit(n_splits=5).split(X_train):
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=6,
                          early_stopping_rounds=100, eval_metric="aucpr",
                          scale_pos_weight=pomer, enable_categorical=True,
                          tree_method="hist", n_jobs=-1)
    m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
          eval_set=[(X_train.iloc[valid_idx], y_train.iloc[valid_idx])], verbose=False)
    skore.append(average_precision_score(y_train.iloc[valid_idx],
                                         m.predict_proba(X_train.iloc[valid_idx])[:, 1]))

print([round(s, 4) for s in skore], "→ priemer", round(np.mean(skore), 4))
```

Rozptyl medzi foldmi je sám osebe informácia: ak sa skóre hýbe medzi 0,45 a 0,68, rozdiel dvoch nastavení o 0,01 nič neznamená.

---

## 7. Vyhodnotenie a voľba prahu

Model vracia **pravdepodobnosť**, nie rozhodnutie. Rozhodnutie z nej urobí až **prah** — a jeho voľba je obchodná, nie štatistická.

```python
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix)

p_test = model.predict_proba(X_test)[:, 1]

print("PR-AUC :", round(average_precision_score(y_test, p_test), 4))   # toto sledujte
print("ROC-AUC:", round(roc_auc_score(y_test, p_test), 4))             # pri 0,2 % podvodov klame
```

Pri 0,2 % pozitívnych prípadov má model „všetko je v poriadku" **accuracy 99,8 %** a je úplne bezcenný. Preto sa pri nevyvážených úlohách sleduje **PR-AUC** (priemerná presnosť) a recall pri prevádzkovo únosnej miere falošných poplachov — podrobne v [04-metriky.md](../01-prehlad/04-metriky.md).

Prah vyberáme podľa toho, koľko poplachov denne dokáže oddelenie rizika spracovať:

```python
precision, recall, prahy = precision_recall_curve(y_test, p_test)

# koľko transakcií denne smieme zablokovať? nech je kapacita 0,5 % objemu
kapacita = 0.005
prah = np.quantile(p_test, 1 - kapacita)

y_hat = (p_test >= prah).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()

print(f"prah {prah:.4f} → zachytených {tp}/{tp+fn} podvodov (recall {tp/(tp+fn):.1%}), "
      f"falošných poplachov {fp} (precision {tp/(tp+fp):.1%})")
```

> V praxi býva prahov viac: nad 0,9 transakciu **zamietnuť**, medzi 0,5 a 0,9 vyžiadať **3-D Secure overenie**, pod 0,5 pustiť. Model dodáva skóre, pravidlá nad ním dodáva biznis.

---

## 8. Uloženie modelu — a čo ešte musíte uložiť s ním

```python
model.save_model("model/xgb_fraud_v3.ubj")      # UBJSON — kompaktný binárny formát
# model.save_model("model/xgb_fraud_v3.json")   # JSON — čitateľný, väčší, dobrý na diff
```

**Nepoužívajte `pickle`.** Pickle uloží Python objekt aj s väzbou na konkrétnu verziu XGBoostu a scikit-learnu; po upgrade sa nemusí dať načítať. Formáty `.ubj` a `.json` sú oficiálne stabilné naprieč verziami a dajú sa načítať aj z iného jazyka (Java, C++, R).

Samotný súbor modelu však **na inferenciu nestačí**. Uložte vedľa neho aj kontrakt vstupu:

```python
import json

kontrakt = {
    "verzia_modelu": "v3",
    "xgboost": xgb.__version__,
    "stlpce": list(X_train.columns),                      # PORADIE je záväzné
    "kategoricke": {c: list(X_train[c].cat.categories) for c in KATEGORICKE},
    "prah": float(prah),
    "best_iteration": int(model.best_iteration),
    "trenovane_do": str(df["cas_transakcie"].iloc[i_tr]),
}
with open("model/kontrakt_v3.json", "w", encoding="utf-8") as f:
    json.dump(kontrakt, f, ensure_ascii=False, indent=2)
```

Prečo to je nutné, ukazuje sekcia 9: model si pamätá stromy, nie to, ako vyzerali dáta, z ktorých vznikli.

### Doučenie na plných dátach

Keď už poznáte optimálny počet stromov, oplatí sa model natrénovať **ešte raz na train + valid spolu** — validačné dáta sú tie najnovšie a pri podvodoch je čerstvosť cenná. Early stopping už nepoužijete (nemáte na čom), počet stromov zafixujete na nájdenej hodnote:

```python
X_full = pd.concat([X_train, X_valid])
y_full = pd.concat([y_train, y_valid])

final = xgb.XGBClassifier(**{**model.get_params(),
                             "n_estimators": model.best_iteration + 1,
                             "early_stopping_rounds": None})
final.fit(X_full, y_full, verbose=False)
final.save_model("model/xgb_fraud_v3.ubj")
```

---

## 9. Inferencia

### Dávková inferencia (nočné skóre, reporty)

```python
model = xgb.XGBClassifier()
model.load_model("model/xgb_fraud_v3.ubj")

kontrakt = json.load(open("model/kontrakt_v3.json", encoding="utf-8"))
X_nove = priprav(df_nove, kontrakt)              # tá istá príprava ako pri tréningu!

p = model.predict_proba(X_nove)[:, 1]            # pravdepodobnosti
y_hat = (p >= kontrakt["prah"]).astype(int)      # rozhodnutia
```

Pri miliónoch riadkov je rýchlejšie obísť sklearn obal a volať jadro priamo:

```python
booster = model.get_booster()
p = booster.inplace_predict(X_nove)              # bez kopírovania do DMatrix
```

### Online inferencia (autorizácia v reálnom čase)

Autorizácia má rozpočet rádovo **100 ms na celú cestu** vrátane siete a databáz — na model zostávajú jednotky milisekúnd. XGBoost to zvládne, ale treba vedieť tri veci:

```python
from fastapi import FastAPI
import xgboost as xgb, pandas as pd, numpy as np, json

app = FastAPI()

booster = xgb.Booster()
booster.load_model("model/xgb_fraud_v3.ubj")
kontrakt = json.load(open("model/kontrakt_v3.json", encoding="utf-8"))

# 1) pri jednom riadku je réžia vlákien väčšia než samotný výpočet
booster.set_param({"nthread": 1})

STLPCE = kontrakt["stlpce"]
KATEGORIE = kontrakt["kategoricke"]

def na_dataframe(tx: dict) -> pd.DataFrame:
    row = pd.DataFrame([tx], columns=STLPCE)          # 2) presné poradie stĺpcov
    for c, kategorie in KATEGORIE.items():            # 3) presne tie isté kategórie
        row[c] = pd.Categorical(row[c], categories=kategorie)
    return row

@app.post("/skore")
def skore(tx: dict):
    X = na_dataframe(tx)
    p = float(booster.inplace_predict(X)[0])
    return {"skore": round(p, 4),
            "rozhodnutie": "zamietnut" if p >= kontrakt["prah"] else "povolit"}
```

Tri číslované poznámky v kóde sú presne tie tri miesta, kde sa online inferencia najčastejšie pokazí:

1. **Vlákna.** Na jednom riadku je paralelizácia kontraproduktívna — rozdeliť prácu medzi 16 jadier trvá dlhšie než ju spraviť. `nthread=1` býva na jednej transakcii aj 5× rýchlejšie. Škálujte radšej počtom procesov služby.
2. **Poradie a názvy stĺpcov.** Model pozná príznaky ako `f0, f1, f2…` v poradí z tréningu. Ak služba pošle stĺpce inak zoradené, model bez chyby vráti **nezmysel** — suma sa bude porovnávať s prahom pre hodinu. Preto sa poradie ukladá do kontraktu a validuje.
3. **Kategórie.** `pd.Categorical` bez explicitného zoznamu kategórií priradí kódy podľa toho, čo práve v dátach je — v jednoriadkovom requeste teda vždy `0`. Kategórie musia prísť z kontraktu. Neznáma hodnota (nové MCC) sa stane `NaN`, čo je v poriadku: XGBoost má pre chýbajúce hodnoty naučený smer.

Meranie latencie patrí do testov rovnako ako presnosť:

```python
import time
X1 = na_dataframe(vzorova_tx)
booster.inplace_predict(X1)                      # zahriatie
t = time.perf_counter()
for _ in range(1000):
    booster.inplace_predict(X1)
print(f"{(time.perf_counter() - t):.2f} ms na transakciu")   # typicky 0,2–2 ms
```

| Situácia | Odporúčanie |
|---|---|
| jeden riadok, nízka latencia | `Booster.inplace_predict`, `nthread=1`, model načítaný raz pri štarte |
| dávka do ~100 tisíc riadkov | `model.predict_proba`, predvolené vlákna |
| dávka v miliónoch | `inplace_predict` po častiach (chunkoch), prípadne `device="cuda"` |
| veľmi prísna latencia (< 0,5 ms) | menej stromov (`best_iteration` orezaný), `max_depth` ≤ 6, prípadne kompilácia cez Treelite |

> **GPU sa na inferenciu oplatí zriedka.** Tréning na GPU (`device="cuda"`) je pri miliónoch riadkov 5–20× rýchlejší, ale jedna transakcia na GPU trvá dlhšie než na CPU — prenos dát prevýši výpočet. Produkčné skórovanie stromov je CPU disciplína.

---

## 10. Prečo model rozhodol tak, ako rozhodol

Pri zamietnutej platbe musíte vedieť dôvod — je to aj regulačná požiadavka. Dve úrovne odpovede:

**Globálne** (ktoré príznaky sú dôležité pre model ako celok):

```python
booster = model.get_booster()
for typ in ("gain", "weight", "cover"):
    top = sorted(booster.get_score(importance_type=typ).items(),
                 key=lambda x: -x[1])[:5]
    print(typ, [(k, round(v, 1)) for k, v in top])
```

- `gain` — **koľko presnosti príznak priniesol** (toto chcete vidieť),
- `weight` — v koľkých splitoch sa vyskytol (uprednostňuje spojité premenné s mnohými prahmi),
- `cover` — koľko riadkov jeho splity ovplyvnili.

**Lokálne** (prečo práve táto transakcia):

```python
import shap

explainer = shap.TreeExplainer(model)
sv = explainer(X_test.iloc[[0]])
shap.plots.waterfall(sv[0])

# to isté bez knižnice shap — XGBoost to vie natívne
prispevky = booster.predict(xgb.DMatrix(X_test.iloc[[0]], enable_categorical=True),
                            pred_contribs=True)
# posledný stĺpec je základná hodnota (bias), zvyšok sú príspevky príznakov
```

Výstup sa dá prečítať ako veta: *„skóre 0,71 vzniklo z bázy 0,002; `tx_60min = 6` pridalo 0,46, nesúlad krajiny 0,18, nočná hodina 0,07."* Presne toto patrí do zdôvodnenia zamietnutia.

---

## 11. Katalóg chýb — čo sa pri XGBoost projekte pokazí najčastejšie

| Príznak | Pravdepodobná príčina | Riešenie |
|---|---|---|
| Validácia vynikajúca, prevádzka zlá | **data leakage** — príznak, ktorý v čase autorizácie neexistoval (napr. „reklamovaná suma") | prejdite príznaky jeden po druhom a pýtajte sa: *bolo toto známe v okamihu transakcie?* |
| Test lepší než validácia | prah alebo hyperparametre ladené na teste | test sa používa **raz**, na konci |
| Model predpovedá stále jednu triedu | extrémna nevyváženosť | `scale_pos_weight`, PR-AUC namiesto accuracy, posun prahu |
| Presnosť v prevádzke nesedí s offline testom | iné poradie stĺpcov alebo iné kódovanie kategórií | kontrakt vstupu (sekcia 8) + test, ktorý porovná skóre služby so skóre z notebooku na tých istých riadkoch |
| `ValueError: DataFrame.dtypes for data must be int, float, bool or category` | textový stĺpec | `astype("category")` + `enable_categorical=True` |
| Tréning trvá hodiny | veľa stromov pri malom `eta`, `max_depth` 12+ | `tree_method="hist"`, `device="cuda"`, ladiť s `eta=0,1` a doladiť až finálny model |
| Skóre po retréningu úplne iné | zmenený zoznam kategórií, iná perióda dát | kontrakt verziujte spolu s modelom, porovnávajte distribúciu skóre starého a nového modelu na tej istej vzorke |
| Model po pár mesiacoch degraduje | **drift** — podvodníci zmenili taktiku, prišli nové MCC | monitorujte PR-AUC na priebežne prichádzajúcich reklamáciách a retrénujte podľa plánu (mesačne) |
| Pomalá odpoveď služby | model sa načítava pri každom requeste, alebo `nthread` = počet jadier | načítať raz pri štarte, `nthread=1` |

---

## 12. Zhrnutie — celý postup na jednej strane

```text
0. princíp     → g a h z chybovej funkcie · w* = −G/(H+λ) · zisk = ½[G²/(H+λ) …] − γ
1. dáta        → kategórie ako 'category', chýbajúce hodnoty nechať ako NaN, žiadne škálovanie
2. delenie     → podľa času: train / valid / test (test odložiť a nesiahať naň)
3. baseline    → eta 0,1 · depth 6 · n_estimators 5000 · early_stopping_rounds 100
4. krivka      → evals_result(): kde sa otvárajú nožnice train vs. valid
5. ladenie     → depth & min_child_weight → subsample & colsample → lambda & gamma → znížiť eta
6. metrika     → PR-AUC (nie accuracy), prah podľa kapacity na falošné poplachy
7. finál       → doučiť na train+valid s best_iteration, uložiť .ubj + kontrakt vstupu
8. inferencia  → rovnaké stĺpce, rovnaké kategórie, inplace_predict, nthread=1
9. vysvetlenie → SHAP / pred_contribs pri každom zamietnutí
10. prevádzka  → monitorovať drift a PR-AUC, retrénovať podľa plánu
```

---

## Kontrolné otázky

1. Prečo sa pri XGBoost nemusia stĺpce škálovať ani sa nemusia dopĺňať chýbajúce hodnoty — a čo z toho vyplýva pre porovnanie s neurónovou sieťou na tých istých dátach?
2. Odvoďte hodnotu listu `w* = −G/(H+λ)`. Z akej funkcie sa minimalizuje a prečo je výsledkom práve podiel týchto dvoch súčtov?
3. Napíšte `g` a `h` pre štvorcovú chybu a pre log-loss. Prečo je `−g` pri štvorcovej chybe presne rezíduum a čo je `h` pri klasifikácii?
4. Napíšte vzorec zisku zo splitu a ukážte v ňom λ, γ a `min_child_weight`. Čo presne obmedzuje každý z nich?
5. Prečo je `min_child_weight = 1` pri silne nevyváženej úlohe oveľa prísnejšie obmedzenie než pri vyváženej?
6. Ako sa v exact greedy algoritme dosiahne, že vyhodnotenie všetkých prahov jedného stĺpca stojí jeden prechod? Čo na tom mení histogramová metóda a čo je trik s odčítaním histogramu súrodenca?
7. Ako XGBoost rozhoduje, kam poslať chýbajúce hodnoty? Aký ďalší kandidát na split z toho vyplýva a kedy vyhrá?
8. Na čo slúžia tri oddelené množiny (train / valid / test) a čo presne sa pokazí, ak podľa testovacej množiny vyberiete prah?
9. Čo robí `early_stopping_rounds` a prečo model po ňom neobsahuje len `best_iteration` stromov? Kedy vás to môže pri natívnom API stáť presnosť?
10. Máte model s `eta = 0,1` a 400 stromami. Aké nastavenie skúsite, ak chcete o niečo lepšiu generalizáciu, a ako sa zmení čas tréningu?
11. Prečo je pri 0,2 % podvodov accuracy nepoužiteľná metrika a čo sledujete namiesto nej?
12. Vymenujte tri veci, ktoré musíte uložiť popri súbore modelu, aby sa inferencia správala rovnako ako tréning. Čo sa stane, ak vynecháte poradie stĺpcov?
13. Prečo je pri skórovaní jednej transakcie `nthread=1` rýchlejšie než plná paralelizácia a prečo sa GPU na online inferenciu neoplatí?
14. Model má na validácii PR-AUC 0,71, v prevádzke po nasadení 0,22. Aké tri príčiny preveríte ako prvé a ako ich rozlíšite?

---

### Súvisiace dokumenty

- [03-xgboost-priklad-iso8583.md](../02-typy-modelov/03-xgboost-priklad-iso8583.md) — mechanika boostingu prepočítaná rukou na tých istých dátach
- [02-random-forest-a-xgboost.md](../02-typy-modelov/02-random-forest-a-xgboost.md) — bagging vs. boosting, prečo stromy vyhrávajú na tabuľkách
- [04-metriky.md](../01-prehlad/04-metriky.md) — PR-AUC, precision/recall, voľba prahu pri nevyvážených triedach
- [03-generalizacia-a-preucenie.md](../01-prehlad/03-generalizacia-a-preucenie.md) — preučenie a delenie dát
- [02-problemy-pri-uceni.md](02-problemy-pri-uceni.md) — tie isté otázky, len pri neurónových sieťach
- [01-vyvojove-prostredie.md](../00-prostredie/01-vyvojove-prostredie.md) — prostredie, notebooky, na čom to spustiť

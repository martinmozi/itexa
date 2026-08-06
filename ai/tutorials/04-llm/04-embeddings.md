# Embeddingy a príprava dát pre RAG

> **Poradie čítania:** ← [Prehľad súčasných modelov](03-llm-modely.md) · **lekcia 6** · [RAG](05-rag.md) →

> **Tutoriál:** ako sa z obyčajného textu stane vektor, ktorý sa dá vyhľadávať, a kde v celom RAG pipeline hrá úlohu malý (embedding / rerank) LLM model, ktorý pri inferencii reálne potrebuje slušný kus CPU – a niekedy aj GPU.

Tento dokument sleduje **cestu jedného kúsku textu** od surových znakov až po hotový vektor: tokenizácia → token embeddingy → transformer vrstvy → pooling → normalizácia. Všetko na konkrétnych číslach, ktoré sa dajú prepočítať ceruzkou.

Čo sa s hotovým vektorom ďalej robí — indexovanie, vyhľadávanie, reranking a celá RAG pipeline — je v nadväzujúcom dokumente **[05-rag.md](05-rag.md)**.

Predpokladá znalosť [transformerov](01-transformer-siete.md) z lekcie 4: attention, multi-head, positional encoding a reziduálne spojenia tu už len použijeme a doplníme im čísla.

> **Ako čítať tento text:** kľúčové kroky sú rozpísané na malých, ručne prepočítateľných príkladoch (miniatúrny slovník, `hidden_dim = 4`, jedna krátka veta). Čísla naprieč Časťou 1 tvoria **jeden súvislý bežecký príklad** – tú istú vetu potiahneme cez tokenizáciu, embedding maticu, attention, pooling aj normalizáciu. Reálne modely majú tie isté operácie, len s rozmermi 1024+ a miliardami parametrov. Ak si príklady prepočítate na papieri, viete podľa nich celý proces aj naprogramovať.

---

## Vstup: text → čísla

Predstavme si vetu z firemného dokumentu:

```text
"Zamestnanec má nárok na 25 dní dovolenky."
```

**Cieľ celého procesu:** dostať z tejto vety **jeden vektor** – pole napr. `1024` desatinných čísel – tak, aby vety s podobným významom mali podobné vektory (blízko v priestore) a vety s iným významom mali vzdialené vektory. Poďme sa pozrieť, ako sa to reálne deje vo vnútri.

---

## Krok 1: Tokenizácia

Model nepracuje priamo so slovami, ale s **tokenmi** – kúskami textu, ktoré nemusia byť celé slová. Tokenizér (napr. BPE, WordPiece, SentencePiece) má svoj naučený slovník (*vocabulary*), typicky `30 000 – 100 000+` položiek, a vetu podľa neho rozreže:

```text
"Zamestnanec má nárok na 25 dní dovolenky."
   → ["Zamest", "nanec", " má", " nárok", " na", " 25", " dní", " dovolen", "ky", "."]
   → token ID:  [4521, 892, 33, 1204, 12, 8842, 445, 2210, 87, 5]
```

Toto je len **ilustračné** rozdelenie – presný výsledok závisí od konkrétneho tokenizéra.

### Ako sa slovník vôbec naučí (BPE krok po kroku)

Najrozšírenejší algoritmus je **BPE** (*Byte-Pair Encoding*). Slovník sa netvorí ručne – učí sa štatisticky z trénovacieho korpusu, a to takto:

1. **Inicializácia.** Slovník začína ako množina všetkých jednotlivých znakov (pri byte-level BPE dokonca 256 bajtov). Každé slovo v korpuse sa zapíše ako postupnosť znakov, na konci s markerom hranice slova, napr. `</w>`.
2. **Počítanie párov.** Prejde sa celý korpus a spočíta sa, ako často sa v ňom vyskytuje **každá dvojica susedných symbolov**.
3. **Zlúčenie (merge).** Najčastejší pár sa zlúči do jedného nového symbolu a pridá sa do slovníka. Toto zlúčenie sa zapíše do zoznamu *merge rules* (na poradí záleží).
4. **Opakovanie.** Kroky 2–3 sa opakujú, kým slovník nedosiahne cieľovú veľkosť (napr. 50 000).

**Miniatúrny príklad.** Nech je náš korpus len tri slová s frekvenciami: `nízka` (5×), `nízko` (2×), `nízkosť` (1×). Rozpíšeme na znaky:

```text
n í z k a </w>      (5×)
n í z k o </w>      (2×)
n í z k o s ť </w>  (1×)
```

Spočítame páry (frekvencia = koľkokrát sa slovo vyskytuje):

```text
(n,í)  = 5+2+1 = 8      (í,z)  = 8      (z,k)  = 8
(k,a)  = 5             (k,o)  = 2+1 = 3  (o,</w>) = 2
(o,s)  = 1             (s,ť)  = 1        ...
```

Najčastejší je `(n,í)=8` → zlúčime na `ní`. Prepíšeme korpus a znova počítame; teraz vyhrá `(ní,z)=8` → `níz`, potom `(níz,k)=8` → `nízk`. Po troch merge pravidlách máme podreťazec `nízk` ako jeden token, ktorý zdieľajú všetky tri slová. Zriedke zakončenia (`osť`) zostanú rozbité na menšie kúsky. Presne **preto** sa časté kmene slov stanú jedným tokenom a zriedkavé slová sa poskladajú z viacerých – slovník je kompromis medzi „všetko sú znaky" (krátky slovník, dlhé sekvencie) a „všetko sú slová" (obrovský slovník, problém s neznámymi slovami).

### Ako sa nová veta rozreže (inferencia)

Pri reálnom použití sa merge pravidlá aplikujú **v tom istom poradí**, v akom sa naučili. Slovo sa rozbije na znaky a postupne sa aplikuje prvé použiteľné pravidlo, potom ďalšie, atď. WordPiece namiesto toho robí **greedy longest-match**: od začiatku slova hľadá najdlhší reťazec, ktorý je v slovníku, ten odreže a pokračuje od zvyšku. Príklad greedy segmentácie proti slovníku `{"dovolen", "ku", "ky", "do", "vo", "len", ...}`:

```text
"dovolenky"
  → skúsi "dovolenky" (nie v slovníku) → skráti
  → skúsi "dovolenk"  (nie) → ... → "dovolen" (ÁNO) → odreže
  zvyšok "ky"
  → "ky" (ÁNO) → odrež
  výsledok: ["dovolen", "ky"]
```

> **Dôležitý postreh:** slovenčina/čeština sa pri mnohých (najmä anglicky trénovaných) modeloch rozreže na **viac** tokenov ako ekvivalentná anglická veta, lebo slovník bol trénovaný hlavne na angličtine – teda časté anglické kmene majú svoj token, kým slovenské sa musia poskladať z drobných kúskov. Napr. anglické „vacation" môže byť 1 token, kým „dovolenka" pokojne 3–4. To má priamy dopad na to, koľko textu sa zmestí do jedného chunku.

### Špeciálne tokeny

Okrem obsahových tokenov má slovník aj **riadiace (špeciálne) tokeny**, ktoré nesú žiadne slovo, ale majú funkčnú úlohu:

| Token | Význam |
|---|---|
| `[CLS]` / `<s>` | začiatok sekvencie; jeho výstupný vektor sa často berie ako zhrnutie (viď pooling) |
| `[SEP]` / `</s>` | oddeľovač / koniec sekvencie (dôležité pri cross-encoderi, kde spájame dva texty) |
| `[PAD]` | výplň, aby mali všetky sekvencie v jednom batchi rovnakú dĺžku |
| `[UNK]` | neznámy token (pri byte-level BPE prakticky nevzniká, lebo vždy vieme spadnúť na bajty) |

Po tokenizácii teda reálne do modelu nevchádza `[4521, 892, ...]`, ale napr. `[CLS] 4521 892 ... 5 [SEP]` – s pridanými riadiacimi tokenmi na krajoch.

---

## Krok 2: Token embeddings – obyčajná lookup tabuľka

Prvá "vrstva" modelu vôbec nie je nič inteligentné – je to **embedding matica**, obyčajná tabuľka rozmerov `[vocab_size × hidden_dim]`, napr. `[50 000 × 1024]`. Token ID je index riadku:

```text
token ID 4521 ("Zamest") → riadok 4521 v matici → vektor [0.03, -0.12, 0.44, ..., 0.09]  (1024 čísel)
```

Toto je čisté **vyhľadanie v tabuľke**, žiadny výpočet. Na začiatku trénovania sú tieto čísla náhodné; trénovaním sa postupne "naučia" byť užitočné.

> **Dôležité:** toto ešte **NIE JE** finálny embedding vety, ani len embedding slova v kontexte. Slovo *"banka"* by v tomto kroku malo úplne rovnaký vektor, či ide o vetu o financiách alebo o rieke – model ešte nevidel žiadny kontext.

### Náš bežecký príklad (potiahneme ho cez celý zvyšok dokumentu)

Aby sa dalo počítať ručne, zmenšíme všetko na hračkárske rozmery: **slovník má 6 tokenov** a **hidden_dim = 4**. Spracujeme kratučkú vetu *„nárok na dovolenku"*, ktorá sa (v tomto hračkárskom tokenizéri) rozreže na tri tokeny:

```text
token:    "nárok"   "na"   "dovolenku"
token ID:    2        4         5
```

Embedding matica `E` má teda rozmer `[6 × 4]` (6 riadkov = 6 tokenov v slovníku, 4 stĺpce = dimenzie). Nech vyzerá takto (riadky sú indexované od 0):

```text
        dim0    dim1    dim2    dim3
ID 0 [  0.10   -0.20    0.30    0.00 ]
ID 1 [ -0.50    0.10    0.20    0.40 ]
ID 2 [  0.20    0.90   -0.10    0.30 ]   ← "nárok"
ID 3 [  0.00    0.00    0.50   -0.50 ]
ID 4 [  0.60   -0.30    0.10    0.20 ]   ← "na"
ID 5 [  0.30    0.70    0.40   -0.10 ]   ← "dovolenku"
```

Lookup je len výber riadkov 2, 4, 5. Dostaneme tri vektory (usporiadané do matice `X` rozmeru `[3 × 4]`, riadok = token):

```text
x_nárok      = [ 0.20,  0.90, -0.10,  0.30 ]
x_na         = [ 0.60, -0.30,  0.10,  0.20 ]
x_dovolenku  = [ 0.30,  0.70,  0.40, -0.10 ]
```

Žiadne násobenie – len tri prekopírované riadky. Na týchto troch vektoroch budeme počítať ďalej.

### Pozičné kódovanie – aby model vedel poradie

K tomuto vektoru sa ešte pripočíta **pozičná informácia** (*positional encoding*). Prečo je nutná, vieme z [lekcie 4](01-transformer-siete.md#positional-encoding--kde-je-informácia-o-poradí) – tu si ju len dopočítame.

Klasická (sínusová) verzia definuje pre pozíciu `pos` a dimenziu `i` hodnotu:

```text
PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )
```

kde `d` je hidden_dim (u nás 4). Vypočítajme pozičný vektor pre **pozíciu 0** a **pozíciu 1** (menovatele: pre `i=0` je `10000^(0/4)=1`, pre `i=1` je `10000^(2/4)=100`):

```text
pos=0:  [ sin(0/1),   cos(0/1),   sin(0/100),   cos(0/100)   ] = [ 0.000, 1.000, 0.000, 1.000 ]
pos=1:  [ sin(1/1),   cos(1/1),   sin(1/100),   cos(1/100)   ] = [ 0.841, 0.540, 0.010, 1.000 ]
pos=2:  [ sin(2/1),   cos(2/1),   sin(2/100),   cos(2/100)   ] = [ 0.909,-0.416, 0.020, 1.000 ]
```

Pripočítame ku každému tokenovému vektoru ten pozičný, ktorý zodpovedá jeho poradiu vo vete (nárok=poz.0, na=poz.1, dovolenku=poz.2):

```text
h_nárok      = x_nárok      + PE(0) = [0.20+0.000, 0.90+1.000, -0.10+0.000, 0.30+1.000] = [ 0.200, 1.900, -0.100, 1.300 ]
h_na         = x_na         + PE(1) = [0.60+0.841,-0.30+0.540,  0.10+0.010, 0.20+1.000] = [ 1.441, 0.240,  0.110, 1.200 ]
h_dovolenku  = x_dovolenku  + PE(2) = [0.30+0.909, 0.70-0.416,  0.40+0.020,-0.10+1.000] = [ 1.209, 0.284,  0.420, 0.900 ]
```

Tieto tri vektory `h_*` sú **vstup do prvej transformer vrstvy**.

---

## Krok 3: Transformer vrstvy – tu sa deje "pochopenie kontextu"

Toto je **jadro celého modelu** a zároveň to najdrahšie na výpočet. Máme teraz `n` vektorov (u nás 3, v reálnej vete napr. 10), a tie prechádzajú cez `N` vrstiev (napr. `12–24`, podľa veľkosti modelu). V každej vrstve sa deje self-attention + feed-forward.

Mechaniku poznáme z [lekcie 4](01-transformer-siete.md#attention-krok-po-kroku-query-key-value) – teraz ju **rozpíšeme s číslami**, aby bolo vidno, že za „pochopením kontextu" nie je nič iné než niekoľko násobení a jeden softmax. Toto je zároveň ten ručný prepočet, ktorý bol v lekcii 4 avizovaný.

### 3a) Projekcia na Query, Key, Value

Pre každý token sa z jeho vektora `h` spočítajú **lineárnou transformáciou** (násobenie naučenou maticou váh) tri nové vektory: **Query (Q)** = „čo hľadám", **Key (K)** = „čo ponúkam", **Value (V)** = „aký obsah nesiem". Každá z troch matíc `W_Q, W_K, W_V` má u nás rozmer `[4 × 4]` (z 4-rozmerného vstupu na 4-rozmerný výstup).

Aby sa to dalo počítať v hlave, zvolíme extrémne jednoduché váhové matice. Nech `W_Q`, `W_K`, `W_V` len **vyberajú a škálujú niektoré dimenzie** (v realite sú husté a naučené, ale princíp je ten istý – lineárna kombinácia vstupných čísel):

```text
        (každý riadok W hovorí, ako namiešať vstupné dim0..dim3 do jedného výstupného čísla)
W_Q =  [ 1 0 0 0 ]      W_K = [ 1 0 0 0 ]      W_V = [ 1 0 0 0 ]
       [ 0 1 0 0 ]            [ 0 1 0 0 ]            [ 0 1 0 0 ]
       [ 0 0 0 0 ]            [ 0 0 0 0 ]            [ 0 0 1 0 ]
       [ 0 0 0 0 ]            [ 0 0 0 0 ]            [ 0 0 0 1 ]
```

`W_Q` aj `W_K` tu jednoducho **zoberú prvé dve dimenzie** (dim0, dim1) a zvyšok vynulujú; `W_V` prenesie dim0, dim1, dim2, dim3 (identita). Aplikujeme `q_i = h_i · W_Q` atď. na naše tri tokeny (pripomeňme `h`):

```text
h_nárok      = [ 0.200, 1.900, -0.100, 1.300 ]
h_na         = [ 1.441, 0.240,  0.110, 1.200 ]
h_dovolenku  = [ 1.209, 0.284,  0.420, 0.900 ]
```

Keďže `W_Q`/`W_K` berú len prvé dve súradnice:

```text
q_nárok     = k_nárok     = [ 0.200, 1.900, 0, 0 ]   → pracujeme s dvojicou (0.200, 1.900)
q_na        = k_na        = [ 1.441, 0.240, 0, 0 ]   → (1.441, 0.240)
q_dovolenku = k_dovolenku = [ 1.209, 0.284, 0, 0 ]   → (1.209, 0.284)
```

A `V` (identita, plné 4 dimenzie):

```text
v_nárok     = [ 0.200, 1.900, -0.100, 1.300 ]
v_na        = [ 1.441, 0.240,  0.110, 1.200 ]
v_dovolenku = [ 1.209, 0.284,  0.420, 0.900 ]
```

### 3b) Attention skóre = dot product Q · K

Relevancia tokenu `i` voči tokenu `j` je **dot product** `q_i · k_j`. Spočítame ho pre **všetky páry** – to je matica `n × n`, u nás `3 × 3`. Zoberme si za `i` token **„nárok"** (`q = (0.200, 1.900)`) a počítajme skóre voči každému tokenu:

```text
score(nárok, nárok)     = 0.200·0.200 + 1.900·1.900 = 0.040 + 3.610 = 3.650
score(nárok, na)        = 0.200·1.441 + 1.900·0.240 = 0.288 + 0.456 = 0.744
score(nárok, dovolenku) = 0.200·1.209 + 1.900·0.284 = 0.242 + 0.540 = 0.782
```

### 3c) Škálovanie /√d_k

Skóre sa vydelí odmocninou z rozmeru kľúča `d_k` (aby pri veľkých dimenziách skóre neexplodovali a softmax nespadol do extrémov). U nás sú aktívne 2 dimenzie, `√2 ≈ 1.414`:

```text
3.650 / 1.414 = 2.581
0.744 / 1.414 = 0.526
0.782 / 1.414 = 0.553
```

### 3d) Softmax → attention weights

Škálované skóre prevedieme na váhy so súčtom 1 pomocou **softmax**: `softmax(z)_j = e^{z_j} / Σ_k e^{z_k}`.

```text
e^2.581 = 13.21
e^0.526 =  1.692
e^0.553 =  1.739
súčet   = 13.21 + 1.692 + 1.739 = 16.64

váha(nárok→nárok)     = 13.21  / 16.64 = 0.794
váha(nárok→na)        =  1.692 / 16.64 = 0.102
váha(nárok→dovolenku) =  1.739 / 16.64 = 0.105
                                         (súčet = 1.000 ✓)
```

Čítame to takto: token „nárok" pri prepočte venuje **79 %** pozornosti sám sebe a zvyšok rozdelí medzi „na" a „dovolenku". (V natrénovanom modeli by váhy neboli také samo-centrické – tu ich diktujú naše umelé váhové matice.)

### 3e) Výstup = vážený súčet Value vektorov

Nový vektor tokenu „nárok" je vážený priemer `V` vektorov všetkých tokenov, kde váhy sú práve tie attention weights:

```text
out_nárok = 0.794·v_nárok + 0.102·v_na + 0.105·v_dovolenku

dim0: 0.794·0.200 + 0.102·1.441 + 0.105·1.209 = 0.1588 + 0.1470 + 0.1269 = 0.433
dim1: 0.794·1.900 + 0.102·0.240 + 0.105·0.284 = 1.5086 + 0.0245 + 0.0298 = 1.563
dim2: 0.794·(-0.100)+0.102·0.110 + 0.105·0.420 = -0.0794 + 0.0112 + 0.0441 = -0.024
dim3: 0.794·1.300 + 0.102·1.200 + 0.105·0.900 = 1.0322 + 0.1224 + 0.0945 = 1.249

out_nárok = [ 0.433, 1.563, -0.024, 1.249 ]
```

**Toto je pointa celého transformera:** pôvodný vektor tokenu „nárok" bol `[0.200, 1.900, -0.100, 1.300]`; po attention je `[0.433, 1.563, -0.024, 1.249]` – **primiešali sa doň hodnoty z „na" a „dovolenku"**. Vektor pre „dovolenku" by po analogickom výpočte niesol stopu „nároku". Kontext sa doslova „vmiešal" do čísel. (Rovnaký postup sa spraví pre `q_na` a `q_dovolenku` – dostaneme `out_na` a `out_dovolenku`; vynechávame, aby sa text nezahltil, ale je to identická aritmetika.)

### 3f) Multi-head attention

Práve prepočítaný výpočet je **jedna hlava**. V realite ich beží viac paralelne, každá na svojom výseku dimenzií a s vlastnými `W_Q/W_K/W_V` – detaily v [lekcii 4](01-transformer-siete.md#multi-head-attention--viac-pohľadov-naraz). Pre náš prepočet je podstatné len to, že aritmetika je v každej hlave presne tá istá, len na kratších vektoroch.

### 3g) Reziduálne spojenie + LayerNorm

Výstup attention sa **nepoužije priamo**. Najprv sa **pripočíta späť vstup** (*residual / skip connection*) a výsledok sa **normalizuje** (*LayerNorm*):

```text
r_nárok = h_nárok + out_nárok = [0.200+0.433, 1.900+1.563, -0.100-0.024, 1.300+1.249]
        = [ 0.633, 3.463, -0.124, 2.549 ]
```

Reziduálne spojenie zabezpečí, že sa pôvodná informácia „nestratí" a že gradient má pri trénovaní kadiaľ tiecť aj cez desiatky vrstiev. LayerNorm potom prečísluje vektor tak, aby mal (naprieč svojimi 4 súradnicami) priemer 0 a rozptyl 1, a ešte ho preškáluje dvomi naučenými parametrami `γ, β`.

Ukážka samotnej normalizácie na `r_nárok`. Priemer je `(0.633 + 3.463 − 0.124 + 2.549) / 4 = 1.630`, odchýlky od priemeru sú `[−0.997, 1.833, −1.754, 0.919]`, ich druhé mocniny `[0.994, 3.359, 3.077, 0.844]`, takže rozptyl je `8.275 / 4 = 2.069` a smerodajná odchýlka `√2.069 = 1.438`:

```text
LN(r)_i = (r_i - priemer) / odchýlka
        = [(0.633-1.630)/1.438, (3.463-1.630)/1.438, (-0.124-1.630)/1.438, (2.549-1.630)/1.438]
        = [ -0.693, 1.275, -1.220, 0.639 ]        (priemer ≈ 0, rozptyl ≈ 1)
```

### 3h) Feed-forward sieť (FFN)

Po attention nasleduje ešte **feed-forward sieť** – dve lineárne vrstvy s nelinearitou medzi nimi, aplikované na **každý token nezávisle**:

```text
FFN(x) = W_2 · aktivácia(W_1 · x + b_1) + b_2
```

Typicky prvá vrstva dimenziu **zväčší** (napr. 1024 → 4096), aplikuje sa nelinearita (ReLU/GELU) a druhá vrstva ju vráti späť (4096 → 1024). Malá ukážka s jednou vstupnou súradnicou cez ReLU (`ReLU(z)=max(0,z)`): ak `W_1·x = [2.0, -0.5]`, tak `ReLU → [2.0, 0.0]` – záporná zložka sa „vypne". Práve táto nelinearita dáva modelu schopnosť reprezentovať zložité, nie len lineárne vzťahy. Aj tu je opäť reziduálne spojenie a LayerNorm.

### Zhrnutie jednej vrstvy a opakovanie

Jedna transformer vrstva teda je: `attention → +residual → LayerNorm → FFN → +residual → LayerNorm`. Toto sa opakuje cez všetky vrstvy (`N`-krát) – vektory sa vrstvu po vrstve stávajú čoraz „abstraktnejšími" a kontextovo bohatšími. Po prejdení celého modelu máme stále `n` vektorov (jeden na token), len teraz každý z nich odzrkadľuje aj zvyšok vety.

> **Prečo je práve tento krok výpočtovo náročný:** self-attention je `O(n²)` v počte tokenov (počíta sa každý pár – matica `n × n`), a plus je tu množstvo maticových násobení (Q/K/V projekcie, FFN so zväčšenou dimenziou) naprieč všetkými vrstvami a hlavami. Práve toto z "malého" modelu robí na CPU citeľnú záťaž a na GPU to letí rádovo rýchlejšie. Viac v [Časti 2](#výpočtové-nároky-kde-to-tlačí-na-cpugpu).

---

## Krok 4: Pooling – z n vektorov na 1 vektor

Toto je krok **špecifický práve pre embedding modely** (generatívne/chatovacie modely ho nepotrebujú, lebo tie len predpovedajú ďalší token). Potrebujeme **jeden vektor** pre celý chunk/vetu, nie `n`. Bežné spôsoby:

| Metóda | Ako funguje | Typicky pri |
|---|---|---|
| **Mean pooling** | spriemeruje všetky tokenové vektory | sentence-embedding modely (najčastejšie) |
| **CLS token pooling** | zoberie vektor špeciálneho tokenu `[CLS]` na začiatku, ktorý sa model naučil používať ako "zhrnutie" vety | BERT-style modely |
| **Last-token pooling** | zoberie vektor posledného tokenu, ktorý v kauzálnom attention "videl" všetky predchádzajúce | novšie dekodérové/kauzálne modely |

**Výsledok:** jeden vektor s **pevnou dĺžkou** (napr. 1024 čísel) – či mal vstup 5 slov alebo 500 slov, výstup má vždy rovnaký rozmer.

### Mean pooling s číslami

V kroku 3 sme prepočítali **jednu attention hlavu v jednej vrstve**; reálny model má takých vrstiev tucty, takže výsledné čísla by boli iné. Ďalej preto pokračujeme s týmito troma výstupnými (už kontextualizovanými) vektormi – sú to hodnoty toho istého príkladu po prejdení celého modelu, zaokrúhlené na pekné čísla:

```text
o_nárok      = [ 0.40, 1.50, 0.00, 1.20 ]
o_na         = [ 1.30, 0.30, 0.10, 1.10 ]
o_dovolenku  = [ 1.10, 0.30, 0.40, 0.90 ]
```

Mean pooling = spriemerujeme po jednotlivých dimenziách (súčet troch čísel v stĺpci, delené 3):

```text
dim0: (0.40 + 1.30 + 1.10) / 3 = 2.80 / 3 = 0.933
dim1: (1.50 + 0.30 + 0.30) / 3 = 2.10 / 3 = 0.700
dim2: (0.00 + 0.10 + 0.40) / 3 = 0.50 / 3 = 0.167
dim3: (1.20 + 1.10 + 0.90) / 3 = 3.20 / 3 = 1.067

pooled = [ 0.933, 0.700, 0.167, 1.067 ]
```

> **Poznámka k padding maske:** ak sme v batchi doplnili `[PAD]` tokeny, ich vektory sa do priemeru **nezapočítavajú** (tzv. *masked mean pooling*) – inak by výplň skreslila výsledok. V praxi sa priemer počíta len cez reálne tokeny.

> **Prečo je pooling to, čím sa embedding model líši od chatovacieho.** Obe rodiny majú rovnaké transformer vrstvy; generatívny model berie posledný vektor a predpovedá z neho ďalší token (viď [lekcia 4](01-transformer-siete.md)), embedding model všetky vektory zlúči do jedného a ten uloží. Rovnaká architektúra, iný výstupný krok a iná loss – presne ten princíp „dáta a loss určujú, čo sa model naučí" z [lekcie 5](02-llm-trening.md).

---

## Krok 5: Normalizácia

Výsledný vektor sa zvyčajne ešte **L2-normalizuje** – vydelí sa svojou dĺžkou (normou), takže má veľkosť presne `1`.

Norma (dĺžka) vektora je `‖v‖ = √(v₀² + v₁² + … )`. Pre náš `pooled`:

```text
‖pooled‖ = √(0.933² + 0.700² + 0.167² + 1.067²)
         = √(0.870 + 0.490 + 0.028 + 1.138)
         = √2.526
         = 1.589
```

Každú súradnicu vydelíme touto normou:

```text
e = pooled / 1.589
  = [ 0.933/1.589, 0.700/1.589, 0.167/1.589, 1.067/1.589 ]
  = [ 0.587, 0.440, 0.105, 0.671 ]
```

Overenie, že norma je teraz 1: `0.587² + 0.440² + 0.105² + 0.671² = 0.345 + 0.194 + 0.011 + 0.450 = 1.000 ✓`

**Prečo je to dôležité:** po normalizácii je **cosine similarity = dot product**, čo sa počíta rýchlejšie (odpadne delenie normami – viď [sekcia o podobnosti](#čo-tie-čísla-vlastne-znamenajú)). Presne preto platí, že `IndexFlatIP` vo FAISS funguje ako cosine podobnosť **len ak sú vektory normalizované** – teraz vidíš aj mechanicky prečo.

Finálny výsledok pre našu vetu je teda vektor s normou 1 (v realite 1024 čísel, u nás 4):

```text
[ 0.587, 0.440, 0.105, 0.671 ]     ← norma = 1
```

Presne tento vektor (spolu s ID chunku a odkazom na pôvodný text) sa uloží do FAISS.

---

## Čo tie čísla vlastne "znamenajú"

Jednotlivé súradnice vektora **nemajú ľudsky čitateľný význam**. Neexistuje "dimenzia č. 5 = formálnosť textu" alebo "dimenzia č. 12 = téma financie". Sú to **naučené abstraktné smery** vo vysokorozmernom priestore, ktoré vznikli ako vedľajší produkt trénovania na obrovskom množstve textu.

Podobnosť sa neposudzuje podľa jednej súradnice, ale podľa **uhla medzi celými vektormi** – preto sa používa cosine similarity, nie napr. rozdiel jednotlivých čísel.

### Ako sa podobnosť reálne počíta

Máme tri bežné miery. Nech `a` a `b` sú dva vektory dĺžky `d`.

**1. Dot product (skalárny súčin):**

```text
a · b = a₀·b₀ + a₁·b₁ + … + a_{d-1}·b_{d-1}
```

**2. Cosine similarity** – dot product znormovaný dĺžkami, čiže **kosínus uhla** medzi vektormi. Nezávisí od dĺžky vektorov, len od ich smeru:

```text
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

Nadobúda hodnoty od `-1` (opačný smer) cez `0` (kolmé, nesúvisiace) po `1` (identický smer, maximálna podobnosť).

**3. Euklidovská vzdialenosť** (L2) – „vzdušná" vzdialenosť medzi hrotmi vektorov; čím **menšia**, tým podobnejšie:

```text
d(a, b) = √( (a₀-b₀)² + (a₁-b₁)² + … )
```

### Dopočítaný príklad

Zoberme query vektor `q` a dva chunk-vektory `c₁`, `c₂` (zámerne **neznormované**, aby bol vidno rozdiel medzi dot a cosine):

```text
q  = [ 0.9, 0.3, 0.1 ]
c₁ = [ 0.8, 0.4, 0.2 ]     (významovo blízky – podobný smer)
c₂ = [ 0.2, 0.9, 0.3 ]     (iný smer)
```

**Dot product:**

```text
q · c₁ = 0.9·0.8 + 0.3·0.4 + 0.1·0.2 = 0.72 + 0.12 + 0.02 = 0.86
q · c₂ = 0.9·0.2 + 0.3·0.9 + 0.1·0.3 = 0.18 + 0.27 + 0.03 = 0.48
```

**Normy:**

```text
‖q‖  = √(0.81+0.09+0.01) = √0.91 = 0.954
‖c₁‖ = √(0.64+0.16+0.04) = √0.84 = 0.917
‖c₂‖ = √(0.04+0.81+0.09) = √0.94 = 0.970
```

**Cosine:**

```text
cos(q, c₁) = 0.86 / (0.954·0.917) = 0.86 / 0.875 = 0.983   ← veľmi podobné
cos(q, c₂) = 0.48 / (0.954·0.970) = 0.48 / 0.925 = 0.519   ← málo podobné
```

**Euklidovská vzdialenosť:**

```text
d(q, c₁) = √((0.9-0.8)²+(0.3-0.4)²+(0.1-0.2)²) = √(0.01+0.01+0.01) = √0.03 = 0.173   ← blízko
d(q, c₂) = √((0.9-0.2)²+(0.3-0.9)²+(0.1-0.3)²) = √(0.49+0.36+0.04) = √0.89 = 0.943   ← ďaleko
```

Všetky tri miery sa zhodnú: `c₁` je bližšie/podobnejšie než `c₂`. Retrieval by teda vybral `c₁`.

### Kľúčový vzťah: prečo normalizujeme

Ak sú vektory **L2-normalizované** (`‖a‖ = ‖b‖ = 1`), tak menovateľ v cosine je `1·1 = 1`, a teda:

```text
cos(a, b) = a · b        (cosine sa rovná obyčajnému dot productu)
```

Navyše platí `d(a,b)² = ‖a‖² + ‖b‖² − 2(a·b) = 2 − 2·cos(a,b)`, takže na normalizovaných vektoroch je **poradie podľa euklidovskej vzdialenosti totožné s poradím podľa cosine** – len otočené (väčší cosine = menšia vzdialenosť). Preto je jedno, či vo FAISS použijeme `IndexFlatIP` (dot/cosine, väčšie = lepšie) alebo `IndexFlatL2` (vzdialenosť, menšie = lepšie) – **na normalizovaných dátach dajú rovnaké top-k**. A práve preto je krok 5 (normalizácia) dôležitý: zjednoduší a zrýchli všetko, čo príde po ňom.

---

## Prečo to vôbec funguje – ako sa model naučí

Toto je asi najdôležitejšia časť, lebo vysvetľuje aj to, **prečo sú rôzne modely vzájomne nekompatibilné.**

Embedding model sa netrénuje náhodne – trénuje sa metódou **kontrastívneho učenia** (*contrastive learning*, typicky s **InfoNCE** loss funkciou):

1. Zoberú sa **trojice**: *anchor* (napr. otázka *"Koľko dní dovolenky mám?"*), *positive* (chunk, ktorý na ňu naozaj odpovedá – naša veta o 25 dňoch), a *negatives* (náhodné iné chunky, o niečom úplne inom).
2. Model spočíta embedding pre všetky tri texty (presne postupom vyššie: tokenizácia → embedding matica → transformer vrstvy → pooling → normalizácia).
3. Loss funkcia model "tlačí" k tomu, aby `cosine similarity(anchor, positive)` bola **vysoká**, a `cosine similarity(anchor, negatives)` bola **nízka**.
4. Cez milióny takýchto trojíc sa **gradient descentom** postupne upravujú všetky váhy siete – embedding matica z kroku 2, aj Q/K/V matice z kroku 3, aj feed-forward váhy.

### InfoNCE loss – vzorec a dopočítaný príklad

Formálne sa najčastejšie používa **InfoNCE** (*Noise-Contrastive Estimation*). Pre anchor `a`, jeho pozitív `p` a množinu negatívov `n₁…n_K` je loss:

```text
              exp( sim(a, p) / τ )
L = − ln ────────────────────────────────────────────
          exp( sim(a, p)/τ ) + Σ_j exp( sim(a, n_j)/τ )
```

kde `sim` je cosine similarity a `τ` (*teplota*, typicky `0.05–0.1`) riadi „ostrosť" – nižšia teplota tvrdšie trestá blízke negatívy. Všimnite si, že vnútro logaritmu je presne **softmax** cez pozitív a negatívy: loss je nízka práve vtedy, keď pozitív dostane drvivú väčšinu „pravdepodobnostnej hmoty".

**Príklad.** Nech model po forward-passe dá tieto cosine podobnosti a `τ = 0.1`:

```text
sim(a, p)  = 0.9   → 0.9/0.1 = 9.0    → e^9.0  = 8103.1
sim(a, n₁) = 0.2   → 0.2/0.1 = 2.0    → e^2.0  = 7.389
sim(a, n₂) = 0.1   → 0.1/0.1 = 1.0    → e^1.0  = 2.718
```

```text
menovateľ = 8103.1 + 7.389 + 2.718 = 8113.2
zlomok    = 8103.1 / 8113.2 = 0.99876
L = − ln(0.99876) = 0.00124        ← malá loss: model už dobre rozlišuje
```

Keby bol model **zlý** a dal by `sim(a,p)=0.3`, `sim(a,n₁)=0.8`, `sim(a,n₂)=0.7` (negatívy podobnejšie než pozitív):

```text
e^3.0 = 20.09 (p),  e^8.0 = 2981.0 (n₁),  e^7.0 = 1096.6 (n₂)
menovateľ = 20.09 + 2981.0 + 1096.6 = 4097.7
zlomok = 20.09 / 4097.7 = 0.00490
L = − ln(0.00490) = 5.32           ← veľká loss → veľký gradient → veľká korekcia váh
```

Gradient tejto veľkej straty sa spätne prešíri (*backpropagation*) cez pooling, všetky transformer vrstvy aj embedding maticu a **pošťuchne** váhy tak, aby nabudúce vyšlo `sim(a,p)` vyššie a `sim(a,n)` nižšie. Kľúčový trik moderného tréningu sú **in-batch negatives**: pozitívy iných príkladov v tom istom batchi sa použijú ako negatívy „zadarmo", takže z batchu veľkosti `B` dostaneme `B−1` negatívov na každý anchor bez extra výpočtu.

**Výsledok:** sémanticky súvisiace texty "vygravitujú" v priestore blízko seba, aj keď použili úplne iné slová (napr. *"dovolenka"* a *"voľno"* alebo *"PTO"* skončia blízko seba, ak to tak model videl v trénovacích dátach).

A tu je odpoveď na to, **prečo sú rôzne modely nekompatibilné:** každý model má inú trénovaciu inicializáciu váh, iné trénovacie dáta, možno inú architektúru/veľkosť. Výsledné "smery" v jeho vektorovom priestore sú teda úplne iné geometrické usporiadanie – aj keby dva modely riešili identickú úlohu s rovnakou dimenziou výstupu, ich súradnicové sústavy si vzájomne nič nehovoria.

> **Praktický dôsledok:** ak preindexujete databázu jedným modelom a otázku zaembeddujete iným, vyhľadávanie vráti nezmysly. **Embedding model sa nedá „za behu" vymeniť** bez preindexovania celej databázy.

---

---

## TL;DR

- **Embedding** = text → tokeny → lookup vektorov (embedding matica) → + pozičné kódovanie → transformer vrstvy (Q/K/V → dot product → /√d → softmax → vážený súčet V; +residual, LayerNorm, FFN) → pooling na 1 vektor → L2 normalizácia. Uloží sa do FAISS.
- **Tokenizácia** (BPE) sa učí štatisticky mergovaním najčastejších párov; slovenčina = viac tokenov na tú istú vetu.
- **Attention** je jadro aj úzke hrdlo: skóre `q_i·k_j` pre všetky páry (`O(n²)`), softmax na váhy, výstup = vážený priemer `V` → kontext sa „vmieša" do každého tokenu.
- Súradnice vektora **nemajú** ľudský význam; podobnosť = **uhol** medzi vektormi (cosine). Na **normalizovaných** vektoroch platí `cosine = dot` a poradie podľa cosine = poradie podľa L2 vzdialenosti.
- Model sa učí **kontrastívne** (InfoNCE): tlačí `sim(anchor, positive)` hore a `sim(anchor, negatives)` dole. Preto sú modely **vzájomne nekompatibilné** → čím sa indexuje, tým sa musí aj dotazovať; zmena modelu = preindexovanie.

---

## Kontrolné otázky

1. Prečo sa text delí na **tokeny** a nie na slová alebo znaky? Čo sa stane s neznámym slovom?
2. Aký je rozdiel medzi **token embeddingom** z lookup tabuľky a výstupom po transformer vrstvách? Prečo je slovo „koruna" v oboch prípadoch iné?
3. Čo robí **pooling** a prečo je mean pooling niečo iné než zobrať vektor prvého tokenu?
4. Načo sa vektor na konci **normalizuje** na jednotkovú dĺžku? Ako to súvisí s kosínusovou podobnosťou?
5. Model má rozmer vektora 768. Čo tých 768 čísel „znamená" a prečo sa jednotlivé zložky nedajú pomenovať?
6. Prečo dva rôzne embedding modely nemajú porovnateľné vektory a čo z toho plynie pri výmene modelu v produkcii?

---

### Súvisiace dokumenty

- [05-rag.md](05-rag.md) — **nasleduje**: čo sa s vektormi robí ďalej (indexovanie, retrieval, reranking)
- [01-transformer-siete.md](01-transformer-siete.md) — architektúra, ktorá vektory vyrába
- [03-llm-modely.md](03-llm-modely.md) — výber embedding modelu
- [zadania/RAG_Fine_tunning.md](../../zadania/RAG_Fine_tunning.md) — zadanie 2A

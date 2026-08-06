# Čo sa pri učení pokazí — diagnostika a riešenia

> **Poradie čítania:** ← [Adam — optimalizátor](01-adam-optimalizator.md) · **lekcia 3** · [Transformery a attention](../04-llm/01-transformer-siete.md) →

> Táto stránka je **katalóg porúch tréningu**: čo sa najčastejšie pokazí, ako to rozoznať
> a ako to opraviť. Nadväzuje na [01-adam-optimalizator.md](01-adam-optimalizator.md), ale dá sa
> používať aj samostatne ako referenčná príručka pri ladení.

Doteraz sme opisovali, ako má tréning vyzerať, keď funguje. V praxi však väčšinu času strávite tým, že **nefunguje**: loss stojí na mieste, skáče, alebo sa zmení na `NaN`. Nasleduje katalóg najčastejších porúch — čo ich spôsobuje, ako ich rozoznať a čo s nimi robiť.

Prvá a najdôležitejšia vec: **nehádajte, merajte.** Sieť je čierna skrinka len dovtedy, kým sa do nej nepozriete. Tri veci, ktoré si treba logovať pri každom tréningu:

1. **loss na trénovacej aj validačnej množine** (dve krivky v jednom grafe),
2. **normu gradientu** — globálne aj po vrstvách,
3. **normu parametrov** a pomer `‖update‖ / ‖parameter‖` (zdravá hodnota je rádovo `1e-3`).

## Rýchla diagnostická tabuľka

| Symptóm | Najpravdepodobnejšia príčina | Kam skočiť |
|---|---|---|
| Loss klesne na náhodnú úroveň a **stojí** | miznúce gradienty, príliš malý `lr`, mŕtve neuróny | §1, §3, §5 |
| Loss **osciluje** alebo rastie | príliš veľký `lr`, nenormalizované vstupy | §5, §6 |
| Loss zrazu **`NaN` / `Inf`** | explodujúce gradienty, `log(0)`, chybné dáta, fp16 pretečenie | §2, §8, §9 |
| Trénovací loss klesá, **validačný rastie** | preučenie | §7 |
| **Oba** lossy vysoké a ploché | nedoučenie, príliš malý model, chyba v backprope | §7, §8 |
| Prvé vrstvy sa **nehýbu**, posledné áno | miznúce gradienty | §1 |
| Loss **skvelý na tréningu, katastrofa v produkcii** | leakage, posun rozdelenia dát | §8 |
| Každý beh dá **iný výsledok** | chýbajúci seed, nedeterministické kernely | §10 |
| Pád po hodinách behu, **zakaždým inde** | hardvér (pamäť, teplota, ECC) | §11 |

---

## 1. Miznúce gradienty (*vanishing gradients*)

**Mechanizmus.** Backpropagation počíta gradient reťazovým pravidlom — gradient pre prvú vrstvu je **súčinom** derivácií všetkých vrstiev nad ňou:

```text
   ∂L/∂W₁  =  ∂L/∂aₙ · σ'(zₙ) · Wₙ · … · σ'(z₂) · W₂ · σ'(z₁) · x
              └────────────── súčin n členov ──────────────┘
```

Ak je typický člen súčinu menší než 1, celý súčin klesá **exponenciálne s hĺbkou**. Sigmoid je učebnicový vinník: jeho derivácia má maximum `0,25` (a to len presne v bode `z = 0`, inak je menšia). Pri desiatich vrstvách teda:

```text
   0,25¹⁰  ≈  0,00000095      →  gradient prvej vrstvy je ~milión-krát menší než poslednej
```

**Ako to spoznáte:** loss rýchlo klesne na úroveň „hádania" a zastaví sa. Keď si vypíšete normy gradientov po vrstvách, uvidíte niečo ako `1e-2, 1e-3, 1e-5, 1e-8` — posledné vrstvy sa učia, prvé stoja. Váhy prvých vrstiev zostávajú takmer identické so svojou inicializáciou.

```python
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name:30s} ‖g‖ = {p.grad.norm().item():.3e}")
```

**Riešenia, v poradí podľa účinnosti:**

| Riešenie | Prečo pomáha |
|---|---|
| **ReLU / GELU namiesto sigmoid a tanh** v skrytých vrstvách | derivácia ReLU je pre kladné `z` presne `1` — súčin sa nezmenšuje |
| **Reziduálne (skip) spojenia**: `y = x + F(x)` | derivácia je `1 + F'(x)`; tá jednotka je „diaľnica", po ktorej gradient prejde do hĺbky nezoslabený. Toto je dôvod, prečo sa dajú trénovať siete so stovkami vrstiev, a je to jadro architektúry ResNet aj transformerov |
| **Normalizačné vrstvy** (BatchNorm, LayerNorm) | držia `z` v oblasti, kde derivácia nie je zanedbateľná (viď §6) |
| **Správna inicializácia** (He / Xavier) | nastaví rozptyl signálu tak, aby nerástol ani neklesal cez vrstvy (viď §4) |
| **Plytšia sieť** | najlacnejšia odpoveď, ak hĺbku nepotrebujete |

> Pri rekurentných sieťach (RNN) je problém ešte ostrejší — súčin sa netvorí cez vrstvy, ale cez **časové kroky**, takže sieť „zabúda" staršie vstupy. Historickým riešením boli hradlované bunky **LSTM / GRU**, dnešným je attention v transformeroch, ktorý siaha na ľubovoľnú predchádzajúcu pozíciu v jednom kroku (viď [01-transformer-siete.md](../04-llm/01-transformer-siete.md)).

---

## 2. Explodujúce gradienty a `NaN`

Presný opak: ak je typický člen súčinu väčší než 1, gradient rastie exponenciálne. Jeden krok s obrovským gradientom vystrelí váhy do nezmyselných hodnôt, z nich vyjde `Inf`, z `Inf − Inf` alebo `0 · Inf` vyjde `NaN` — a `NaN` sa už cez celú sieť **rozšíri a nikdy nezmizne** (`NaN` v akejkoľvek operácii dáva `NaN`).

**Ako to spoznáte:** loss chvíľu klesá, potom skokovo vyletí (napr. `0,8 → 14,2 → NaN`). Norma gradientu pred pádom narastie o rády.

**Riešenia:**

- **Orezávanie gradientu (*gradient clipping*)** — najúčinnejšia poistka. Pred update krokom sa spočíta globálna norma všetkých gradientov a ak prekročí prah, celý vektor sa preškáluje:

  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # pred opt.step()
  ```

  Kľúčové je, že sa škáluje **celý gradient naraz** — tým sa zachová jeho *smer* a obmedzí len *dĺžka* kroku. Pri trénovaní LLM je `max_norm=1.0` prakticky štandard.
- **Znížiť learning rate** a pridať **warmup** (viď §5) — najviac explózií sa deje v prvých stovkách krokov.
- **Skontrolovať loss funkciu** — `log(p)` pre `p = 0` je `−Inf`. Preto sa nikdy nepočíta `log(softmax(z))` v dvoch krokoch, ale numericky stabilným `log_softmax(z)`; v PyTorchi to za vás rieši `CrossEntropyLoss`, ktorá berie **logity**, nie pravdepodobnosti. Rovnako `sqrt(v)` v Adame potrebuje `+ ε`.
- **Nájsť prvý `NaN`** — nehľadajte ho ručne, nechajte si ho ukázať:

  ```python
  torch.autograd.set_detect_anomaly(True)   # spomalí tréning, ale ukáže operáciu, ktorá NaN vyrobila
  ```

---

## 3. Mŕtve ReLU neuróny

ReLU rieši miznúce gradienty, ale prináša vlastnú chorobu. Ak sa neurón dostane do stavu, keď je jeho `z` **záporné pre všetky vstupy** v dátach, jeho výstup je vždy 0, derivácia ReLU je tam tiež 0 — a teda **jeho váhy dostanú nulový gradient a už sa nikdy nepohnú**. Neurón je natrvalo mŕtvy. Najčastejšia príčina: jeden príliš veľký update (vysoký `lr`) stlačí bias hlboko do zápornej hodnoty.

**Ako to spoznáte:** zmerajte podiel aktivácií, ktoré sú presne nula, a hlavne podiel neurónov, ktoré sú nulové pre **celý batch**:

```python
a = relu(z)                                    # (batch, neurons)
dead = (a == 0).all(axis=0).mean()             # podiel trvale mŕtvych neurónov
print(f"mŕtve neuróny: {dead:.1%}")            # nad ~20 % je to problém
```

**Riešenia:** **Leaky ReLU** (`max(0.01·z, z)` — záporná časť má malý sklon, takže gradient nikdy nie je presne nula), **GELU** alebo **ELU** (hladké, dnešný štandard v transformeroch), **nižší `lr`**, **He inicializácia** a **nezačínať s výrazne zápornými biasmi** (inicializujte ich na nulu, prípadne na malé kladné číslo).

---

## 4. Zlá inicializácia váh

Inicializácia je jediná vec, ktorú spravíte **raz**, a napriek tomu rozhodne, či sa sieť vôbec rozbehne.

**Nuly nikdy.** Ak sú všetky váhy vrstvy rovnaké, všetky neuróny v nej počítajú to isté, dostanú ten istý gradient a zostanú navždy identické — vrstva so 100 neurónmi sa správa ako jeden. Toto sa volá **problém symetrie** a jediný liek je náhoda. (Biasy na nulu inicializovať možno a je to bežné — symetriu lámu už náhodné váhy.)

**Ani príliš veľké, ani príliš malé.** Ak je rozptyl váh veľký, `z` rastie cez vrstvy a aktivácie sa dostanú do saturácie (sigmoid) alebo explodujú (ReLU). Ak je malý, signál cez vrstvy vyhasne. Chceme, aby **rozptyl aktivácií zostal cez vrstvy zhruba rovnaký** — a presne to zabezpečujú štandardné schémy:

| Schéma | Rozptyl | Pre ktoré aktivácie |
|---|---|---|
| **He (Kaiming)** | `std = √(2 / n_vstupov)` | **ReLU a príbuzné** — tá dvojka kompenzuje, že ReLU zahodí polovicu signálu |
| **Xavier (Glorot)** | `std = √(2 / (n_vstupov + n_výstupov))` | tanh, sigmoid |

V PyTorchi sú vrstvy rozumne inicializované automaticky; vo vlastnej NumPy implementácii zo [zadania 1](../../zadania/rozpoznavanie-obrazkov.md) to musíte spraviť sami:

```python
W1 = np.random.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)    # He, pre ReLU
b1 = np.zeros(n_hidden)
```

---

## 5. Zle zvolený learning rate

Najčastejšia príčina neúspechu vôbec — a našťastie sa diagnostikuje pohľadom na tvar krivky lossu:

```text
  loss                    loss                    loss
   │╲                      │  ╱╲  ╱╲               │╲
   │ ╲___________          │ ╱  ╲╱  ╲              │ ╲___
   │              ‾‾       │╱                      │     ‾‾‾───___
   └──────────────► krok   └──────────────► krok   └──────────────► krok
   lr je akurát            lr je priveľký          lr je primalý
   (rýchly pokles,         (osciluje, skáče        (klesá, ale
    potom plató)            cez minimum)            beh trvá večnosť)
```

**Ako nájsť ten správny:** urobte krátky beh, v ktorom `lr` **exponenciálne rastie** (napr. z `1e-6` na `1`) a vykreslite loss proti `lr`. Vhodná hodnota je rádovo tam, kde loss klesá najstrmšie — teda zhruba **desatina hodnoty, pri ktorej začne divergovať**. Pre Adama je `1e-3` dobrý štart, pre fine-tuning predtrénovaného modelu skôr `1e-5` až `5e-5` (model je už blízko dobrého riešenia a veľký krok by ho z neho vykopol).

**Warmup a decay.** Na začiatku tréningu sú `m` a `v` v Adame ešte nespoľahlivé odhady z pár vzoriek, takže plný `lr` môže model rozhodiť. Preto sa `lr` prvých niekoľko sto krokov lineárne dvíha z nuly (*warmup*) a potom pomaly klesá (kosínusovo) — viď [hyperparametre Adama](01-adam-optimalizator.md#4-hyperparametre).

---

## 6. Nenormalizované vstupy a normalizačné vrstvy

Ak jeden vstupný stĺpec nadobúda hodnoty v tisícoch (suma v eurách) a iný v desatinách (podiel), krajina chybovej funkcie je **dlhá úzka roklina**: v jednom smere extrémne strmá, v druhom plochá. Jediný `lr` nevyhovuje obom naraz — buď v strmom smere osciluje, alebo sa v plochom vlečie.

**Vstupy preto vždy normalizujte** — odčítajte priemer a vydeľte smerodajnou odchýlkou (**štandardizácia**), pričom oba parametre počítajte **len z trénovacej množiny** a tie isté hodnoty použite na validačnú aj testovaciu (inak ide o data leakage).

Rovnaký problém vzniká aj **vnútri** siete, keď sa rozdelenie aktivácií medzi vrstvami rozbieha. Riešia ho normalizačné vrstvy:

| Vrstva | Normalizuje cez | Kde sa používa | Prečo |
|---|---|---|---|
| **BatchNorm** | cez **batch** (pre každý kanál zvlášť) | CNN | veľmi účinná, navyše mierne regularizuje |
| **LayerNorm** | cez **príznaky jedného príkladu** | transformery, RNN | nezávisí od veľkosti batchu ani od dĺžky sekvencie, správa sa rovnako pri tréningu aj inferencii |

Praktická poznámka pre BatchNorm: pri inferencii sa nepoužíva štatistika aktuálneho batchu, ale kĺzavý priemer z tréningu. Ak zabudnete prepnúť model do `model.eval()`, dostanete pri predikcii nezmyselné a od veľkosti batchu závislé výsledky — je to jedna z najčastejších „záhad" v PyTorchi (spolu s aktívnym dropoutom, viď §7).

> V transformeroch sa navyše rozlišuje **pre-LN** (normalizácia pred blokom) a **post-LN** (za ním). Pre-LN sa trénuje výrazne stabilnejšie a je dnes prakticky vždy voľbou — post-LN pri hlbokých modeloch vyžaduje opatrný warmup, inak sa tréning rozpadne.

---

## 7. Preučenie a nedoučenie

Rozdiel medzi nimi je rozpísaný v [03-generalizacia-a-preucenie.md](../01-prehlad/03-generalizacia-a-preucenie.md); tu je praktický postup, ako ich rozlíšiť a čo s tým. Rozhodujúci je **odstup medzi trénovacím a validačným lossom**:

| Trénovací loss | Validačný loss | Diagnóza | Čo robiť |
|---|---|---|---|
| nízky | **výrazne vyšší a rastie** | preučenie | dropout, weight decay, augmentácia dát, early stopping, **viac dát**, menší model |
| vysoký | vysoký, podobný | nedoučenie | väčší model, dlhší tréning, vyšší `lr`, menej regularizácie, lepšie príznaky |
| nízky | nízky | v poriadku | nechajte to tak |

**Early stopping** je zadarmo a mal by byť predvolený: sledujte validačný loss, zapamätajte si váhy z jeho minima, a ak sa `N` epôch (napr. 10) nezlepší, tréning ukončite a vráťte sa k zapamätaným váham. Bez toho posledného kroku early stopping stráca polovicu zmyslu.

**Dropout** počas tréningu náhodne vynuluje časť neurónov, takže sieť sa nemôže spoľahnúť na žiadny jediný z nich. Pri inferencii sa **musí vypnúť** — v PyTorchi to robí `model.eval()`. Zabudnutý `model.eval()` je najčastejšia príčina toho, že „model má horšie výsledky pri testovaní než pri tréningu, hoci by nemal".

---

## 8. Keď je chyba v dátach alebo v kóde, nie v modeli

Väčšina hodín premárnených ladením hyperparametrov mala byť venovaná dátam. Než začnete ladiť `lr`, urobte tieto dva testy:

**1. Test „preuč jeden batch".** Vezmite **8 príkladov** a trénujte na nich bez akejkoľvek regularizácie. Sieť s dostatočnou kapacitou ich musí zvládnuť naspamäť — loss má klesnúť prakticky na nulu a presnosť na 100 %. **Ak to nedokáže, nemáte problém s učením, ale chybu v kóde** — v backprope, v tvare tenzorov, v priradení labelov, v poradí `zero_grad()` a `step()`. Tento jediný test odhalí drvivú väčšinu implementačných chýb za dve minúty.

**2. Pozrite sa na dáta očami.** Vypíšte si 20 náhodných príkladov aj s ich labelmi. Prekvapivo často sa tam nájde odpoveď.

Typické dátové poruchy:

- **Únik informácie (*data leakage*)** — vo vstupoch je stĺpec, ktorý sa v realite dozviete až po tom, čo poznáte odpoveď (napr. „počet upomienok" pri predikcii nesplácania). Poznávacie znamenie: podozrivo vysoká presnosť a jeden dominantný príznak v SHAP. Ďalšia forma: normalizačné štatistiky alebo výber príznakov počítané z celých dát pred rozdelením na train/test.
- **Duplicity medzi trénovacou a testovacou množinou** — testovacia presnosť je potom fikcia.
- **Nesprávne poradie príkladov a labelov** — po zamiešaní (`shuffle`) jednej z dvoch polí sa model učí šum. Loss klesá podozrivo pomaly na úroveň náhody.
- **Nevyvážené triedy** — pri 99 : 1 model konverguje k „vždy väčšinová trieda". Riešenie: váhy tried v loss funkcii, prevzorkovanie, a hlavne **nesledovať accuracy** (viď príklad detekcie podvodov v [04-metriky.md](../01-prehlad/04-metriky.md)).
- **`NaN` alebo `Inf` priamo vo vstupných dátach** — jeden chýbajúci údaj zmení celý gradient na `NaN` v prvom kroku. `assert np.isfinite(X).all()` na začiatku tréningu stojí jeden riadok.
- **Posun rozdelenia (*distribution shift*)** — model natrénovaný na dátach z minulého roka narazí na inú realitu. Prejaví sa až v produkcii; jediná obrana je monitoring vstupov a pravidelné pretrénovanie.

---

## 9. Numerická presnosť: fp32, fp16 a bf16

Moderný tréning nebeží v plnej presnosti — polovičná presnosť je dvakrát rýchlejšia a zaberá polovicu pamäte. Prináša však vlastnú triedu chýb:

| Formát | Rozsah | Presnosť (mantisa) | Poznámka |
|---|---|---|---|
| **fp32** | ±3,4·10³⁸ | 23 bitov | bezpečný štandard |
| **fp16** | **±65 504** | 10 bitov | **preteká aj podteká**; potrebuje loss scaling |
| **bf16** | ±3,4·10³⁸ | 7 bitov | rovnaký rozsah ako fp32, len hrubšia presnosť — **v praxi bezpečnejšia voľba** |

Pri **fp16** sú dve pasce. Zhora: aktivácie alebo gradienty nad `65 504` sa stanú `Inf`. Zdola: malé gradienty (bežne rádovo `1e-8`) sa **podtečú na nulu** a príslušné váhy sa prestanú učiť — tiché zlyhanie, ktoré na krivke lossu vyzerá ako plató. Preto sa používa **loss scaling**: loss sa pred backpropom vynásobí veľkým číslom (napr. 2¹⁶), gradienty sa tým posunú do reprezentovateľného pásma a pred update krokom sa vydelia späť. `torch.amp.GradScaler` to robí automaticky vrátane dynamického prispôsobovania mierky.

**bf16** obetuje presnosť namiesto rozsahu, čím obe pasce odpadajú — preto sa dnes tréning veľkých modelov robí v bf16 (podporujú ho Ampere a novšie, teda RTX 30xx/40xx, A100, H100). **Optimalizátorový stav (`m`, `v`) a hlavná kópia váh sa aj tak držia v fp32** — to je podstata *mixed precision*: rýchle operácie v polovičnej presnosti, akumulácia v plnej.

---

## 10. Nedeterminizmus a nereprodukovateľnosť

Dva behy s rovnakým kódom dajú rôzne výsledky. Bežné a väčšinou neškodné — kým nehľadáte chybu, vtedy je to zabijak. Zdroje: inicializácia váh, miešanie dát, dropout, a na GPU aj **poradie sčítavania v paralelných redukciách** (sčítanie v pohyblivej rádovej čiarke nie je asociatívne, takže iné poradie dá iný posledný bit) a nedeterministické cuDNN kernely.

```python
torch.manual_seed(42); np.random.seed(42); random.seed(42)
torch.use_deterministic_algorithms(True)      # vynúti deterministické kernely (pomalšie)
```

Ak sa beh **nedá zreprodukovať ani so zafixovaným seedom a determinizmom**, je to silná indícia, že problém nie je v kóde — pokračujte §11.

---

## 11. Hardvér: ECC, tichá korupcia pamäte a dlhé behy

Drvivá väčšina problémov z predchádzajúcich sekcií je softvérová. Hardvérové sa však stávajú tiež, a majú veľmi charakteristický rukopis: **nereprodukovateľnosť**. Softvérová chyba padne pri rovnakom seede vždy na tom istom kroku; hardvérová zakaždým inde.

**Prečo je jeden preklopený bit katastrofa.** Pamäť DRAM aj GPU HBM sú fyzikálne zariadenia — nabitý kondenzátor môže stratiť náboj vplyvom kozmického žiarenia, vadnej bunky alebo prehriatia. V pohyblivej rádovej čiarke pritom nie sú všetky bity rovnako dôležité: preklopenie bitu v mantise zmení hodnotu o zanedbateľný zlomok, ale **preklopenie horného bitu exponentu** zmení `1,0` na `1,7·10³⁸`. Takáto hodnota sa v ďalšom kroku znásobí, vyletí do `Inf`, z toho vznikne `NaN` a niekoľkodňový tréning je na odpis.

**ECC (*Error-Correcting Code*)** je hardvérová obrana: pamäť ku každému slovu ukladá kontrolné bity, ktoré umožnia **jednobitovú chybu automaticky opraviť** a dvojbitovú aspoň **spoľahlivo detegovať** (schéma SECDED — *Single Error Correct, Double Error Detect*).

- **Serverové GPU (A100, H100) a serverová RAM majú ECC.** Opraviteľné chyby sa ticho opravia a len sa započítajú do štatistiky; neopraviteľná chyba zhodí proces s **Xid** chybou.
- **Herné karty (GeForce RTX) ECC na VRAM nemajú.** Pre tréning zo zadania to nevadí — beh trvá minúty. Pre viactýždňový beh je to reálne riziko.
- Stav a históriu chýb si viete pozrieť:

  ```bash
  nvidia-smi -q -d ECC,ROW_REMAPPER   # počty opravených/neopravených chýb, remapované riadky
  nvidia-smi -q -d TEMPERATURE,CLOCK  # throttling: nie je to chyba, ale spomalenie
  dmesg | grep -i -E "xid|ecc|mce"    # hardvérové chyby v systémovom logu
  ```

**Ďalšie hardvérové príčiny, ktoré vyzerajú ako chyba v modeli:**

- **Nestabilná pretaktovaná pamäť** (undervolting, agresívne OC profily) — sporadické `NaN`, ktoré zmiznú po vrátení na sériové takty. Otestujte pamäť záťažovým testom, nie tréningom.
- **Teplotný throttling a nedostatočný zdroj** — model sa neučí horšie, len pomalšie; prejaví sa poklesom taktov v `nvidia-smi`.
- **Viac GPU:** rozsynchronizovaná komunikácia (NCCL timeout / zaseknutie na `all_reduce`), alebo jedna pomalšia karta, ktorá brzdí všetky ostatné.

**Ako často sa to naozaj stáva — čísla z reálneho behu.** Meta zverejnila v technickej správe k modelu **Llama 3** štatistiku z tréningu 405B modelu na klastri **16 384 GPU NVIDIA H100**. Za **54-dňový úsek** predtrénovania došlo k **466 prerušeniam** behu, z toho 47 plánovaných (údržba, aktualizácie firmvéru) a **419 neočakávaných** — teda v priemere **jedno zlyhanie každé ~3 hodiny**:

| Príčina neočakávaného prerušenia | Počet | Podiel |
|---|---|---|
| Chybná GPU (najväčšia jednotlivá kategória) | 148 | ~30 % |
| **Pamäť GPU HBM3** | 72 | 17,2 % |
| Softvérová chyba | 54 | 12,9 % |
| Sieťový prvok (switch, kábel) | 35 | 8,4 % |
| Neplánovaná údržba hosta | 32 | 7,6 % |
| Pamäť GPU SRAM | 19 | 4,5 % |
| **Tichá korupcia dát (*silent data corruption*)** | 6 | 1,4 % |
| ostatné (NIC, SSD, zdroj, CPU, systémová pamäť…) | zvyšok | — |

Zhrnutie správy: **~78 % neočakávaných prerušení malo potvrdenú alebo predpokladanú hardvérovú príčinu** a **58,7 % pripadalo na GPU**. Napriek tomu klaster dosiahol **vyše 90 % efektívneho tréningového času** a manuálny zásah bol potrebný len **trikrát** — všetko ostatné vyriešila automatika: detekcia chybného uzla, jeho vyradenie a reštart z posledného checkpointu.

> Zdroj: Grattafiori et al., **[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)** (arXiv:2407.21783), sekcia 3.3.4 *Reliability and Operational Challenges*, Table 5. *(Poznámka: percentá v Table 5 nie sú úplne konzistentné s počtami — prvý riadok je v origináli uvedený ako 30,1 %, hoci 148 zo 419 je 35,3 %. Ostatné riadky sedia.)*

**Poučenie pre vás nie je „kúpte si lepší hardvér", ale „počítajte so zlyhaním".** Ukladajte checkpoint každých `N` krokov a ukladajte **kompletný stav**, nie len váhy: parametre, **`m` a `v` z Adama** (bez nich sa optimalizátor po reštarte rozbieha odznova a loss vyskočí), číslo kroku, stav rozvrhu learning rate, stav `GradScaler`-a a stav generátorov náhodných čísel. Tréning musí vedieť pokračovať tak, aby na krivke lossu nebolo vidno, kde bol prerušený. Všimnite si aj riadok **silent data corruption**: to je presne ten prípad, keď hardvér nespadne ani nenahlási chybu, len ticho vráti nesprávne číslo — a jediné, čo ho odhalí, je nedôvera k nereprodukovateľným výsledkom.

> Pravidlo pri delení vinníkov: **deterministická chyba = softvér, náhodná = hardvér.** Než začnete podozrievať železo, zafixujte seed a overte, že chyba nastáva zakaždým na rovnakom kroku.

---

## 12. Postup pri ladení — v tomto poradí

1. **Preuč 8 príkladov.** Nejde to? Chyba je v kóde, nie v učení. Ďalej nepokračujte.
2. **Skontroluj dáta.** Tvary, `NaN`, rozsahy, párovanie labelov, duplicity, rozloženie tried.
3. **Znormalizuj vstupy.**
4. **Rozbehni najjednoduchší možný model**, ktorý má šancu fungovať, a až potom pridávaj zložitosť.
5. **Nalaď `lr`** — jeden hyperparameter má väčší dopad než všetky ostatné dohromady.
6. **Loguj normy gradientov po vrstvách.** Miznú? → §1. Explodujú? → §2.
7. **Až keď to konverguje, rieš preučenie** — dropout, weight decay, augmentácia, early stopping.
8. **Meň jednu vec naraz** a zapisuj si výsledky. Bez záznamu nemáte experiment, len dojem.

---

## Kontrolné otázky

1. Vysvetlite mechanizmus miznúcich gradientov cez reťazové pravidlo. Prečo je sigmoid v skrytých vrstvách horší než ReLU a prečo reziduálne spojenie `y = x + F(x)` problém odstraňuje?
2. Loss počas tréningu osciluje hore-dole a nekonverguje. Ktorý hyperparameter podozrievate ako prvý a ako overíte, že máte pravdu?
3. Prečo sa gradient pri clippingu škáluje ako celok, a nie každá zložka zvlášť?
4. Načo je „test preučenia jedného batchu" a čo presne vám hovorí jeho zlyhanie?
5. Sieť s ReLU sa po pár epochách prestane učiť. Ako zmeriate, či za to môžu mŕtve neuróny, a čo s tým spravíte?
6. V čom je bf16 bezpečnejší než fp16, hoci má menej bitov mantisy? Načo slúži loss scaling?
7. Tréning padá po niekoľkých hodinách, zakaždým na inom mieste. Ako rozlíšite softvérovú chybu od hardvérovej a čo je ECC?
8. Prečo pri checkpointe nestačí uložiť len váhy modelu?

---

### Súvisiace dokumenty

- [01-adam-optimalizator.md](01-adam-optimalizator.md) — optimalizátor, ktorého správanie tu ladíme
- [04-feed-forward-siete.md](../02-typy-modelov/04-feed-forward-siete.md) — aktivačné funkcie a prečo na nich záleží
- [03-generalizacia-a-preucenie.md](../01-prehlad/03-generalizacia-a-preucenie.md) — preučenie, regularizácia, delenie dát
- [04-metriky.md](../01-prehlad/04-metriky.md) — prečo accuracy pri nevyvážených triedach klame
- [zadania/rozpoznavanie-obrazkov.md](../../zadania/rozpoznavanie-obrazkov.md) — zadanie, pri ktorom to budete potrebovať

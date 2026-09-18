# Ako sa trénuje LLM — od surového textu po Instruct model

> **Poradie čítania:** ← [Tréning transformera](../03-trening-modelov/04-trening-transformera.md) · **lekcia 5** · [Prehľad súčasných modelov](04-llm-modely.md) →

> **Cieľ dokumentu:** vysvetliť celú tréningovú pipeline veľkého jazykového modelu — čo sa deje od stiahnutia surového internetu až po model s príponou `-Instruct`, ktorý si viete stiahnuť z Hugging Face a ktorý odpovedá na otázky. Po prečítaní budete rozumieť, prečo *base* model „nevie odpovedať", čo presne pridáva inštrukčné ladenie, odkiaľ sa berie povaha hotového asistenta, a kam do tejto pipeline zapadá váš vlastný fine-tuning (LoRA).

Predpokladá znalosť [transformerov](01-transformer-siete.md) (architektúra, ktorá sa trénuje) a [tréningovej slučky](../03-trening-modelov/01-adam-optimalizator.md) (backprop + Adam — presne tá istá mechanika, len v obrovskej mierke); optimalizačné detaily jedného kroku (warmup, clipping, pamäť, čo do toho vnáša attention) rozoberá [04-trening-transformera.md](../03-trening-modelov/04-trening-transformera.md). Tokenizáciu a BPE detailne rozoberá [05-embeddings.md](05-embeddings.md).

### Mapa dokumentu

| Časť | Čo v nej je | Otázka, na ktorú odpovedá |
|---|---|---|
| [Pipeline](#celková-pipeline) · [Fáza 0: dáta](#fáza-0-dáta--surovina-ktorá-rozhoduje-o-všetkom) | tri fázy vedľa seba, príprava korpusu | Čo všetko sa s modelom stane, než ho stiahnem z Hugging Face? |
| [Fáza 1: pretraining](#fáza-1-pretraining--predikcia-ďalšieho-tokenu) | predikcia ďalšieho tokenu, [jeden prechod = veľa príkladov](#jeden-prechod-veľa-tréningových-príkladov-teacher-forcing), perplexita, scaling laws, klaster | Ako sa dá učiť na biliónoch tokenov bez labelov a čo z toho vyjde? |
| [Base model v akcii](#base-model-v-akcii-iterácie-generovania) | **generovanie krok po kroku** | Prečo vznikne len jeden token a ako sa výstup vracia na vstup? |
| [Fáza 2: SFT](#fáza-2-sft--instruction-tuning--z-dokončovača-asistent) | chat šablóna, [maskovanie loss](#tréning-loss-len-na-odpovedi), [dve masky](#dve-masky-ktoré-sa-ľahko-zamieňajú), [Instruct model v akcii](#instruct-model-v-akcii-tá-istá-otázka-iný-priebeh) | Ako sa z dokončovača textu stane asistent, ktorý odpovedá a vie prestať? |
| [Fáza 3: preferenčné ladenie](#fáza-3-preferenčné-ladenie--rlhf-dpo-a-reasoning) | RLHF, DPO, RLVR a povaha hotového asistenta | Odkiaľ sa berie tón, odmietanie a „reasoning"? |

---

## Celková pipeline

```text
  surový text (web, knihy, kód)          ~bilióny tokenov
        │
        ▼  filtrovanie, deduplikácia, mixovanie
  ČISTÝ KORPUS
        │
        ▼  FÁZA 1: PRETRAINING (predikcia ďalšieho tokenu)     mesiace, tisíce GPU
  BASE MODEL            („dokončovač textu" — napr. Llama-3.1-8B)
        │
        ▼  FÁZA 2: SFT / INSTRUCTION TUNING (dvojice inštrukcia → odpoveď)   dni
  INSTRUCT MODEL        („asistent" — napr. Llama-3.1-8B-Instruct)
        │
        ▼  FÁZA 3: preferenčné ladenie (RLHF / DPO / RLVR)     dni až týždne
  CHATOVACÍ MODEL, ktorý reálne používate
```

Kľúčová intuícia: **všetky fázy používajú tú istú sieť a tú istú tréningovú slučku** (forward → loss → backprop → Adam update). Líšia sa len **dátami a loss funkciou** — a práve dáta určujú, čo sa model naučí.

| | **Fáza 1 — pretraining** | **Fáza 2 — SFT** | **Fáza 3 — preferenčné ladenie** |
|---|---|---|---|
| Dáta | surový text (10¹² tokenov) | dvojice inštrukcia → odpoveď (10⁴–10⁶) | dvojice lepšia / horšia odpoveď |
| Loss | cross-entropy na **každom** tokene | cross-entropy **len na tokenoch odpovede** | preferenčná (DPO) alebo odmena z reward modelu (PPO) |
| Čo sa mení | znalosti a schopnosti | správanie a formát dialógu | odtiene: tón, opatrnosť, odmietanie |
| Cena | mesiace, tisíce GPU, desiatky M$ | hodiny až dni, jedna až pár GPU | dni |
| Výsledok | base model | model s príponou `-Instruct` | model, ktorý reálne používate |

Posledný riadok je zároveň mapa toho, čo si z celého dokumentu odniesť: **schopnosti → asistent → vycibrený asistent.**

---

## Fáza 0: Dáta — surovina, ktorá rozhoduje o všetkom

Pretraining potrebuje **bilióny (10¹²) tokenov** textu. Typický mix:

| Zdroj | Podiel (orientačne) | Prečo |
|---|---|---|
| web (Common Crawl a pod.) | najväčší | šírka tém a jazykov |
| kód (GitHub) | výrazný | učí štruktúrované myslenie — zlepšuje aj *ne*programovacie schopnosti |
| knihy, články, Wikipedia | menší, ale kvalitný | dlhé súvislé texty, fakty |
| matematika, veda | cielený | reasoning |

Surový web je však plný spamu, duplikátov a nekvalitného obsahu, preto sa robí:

1. **Filtrovanie kvality** — heuristiky aj klasifikátory vyhodia spam, generovaný balast, toxický obsah.
2. **Deduplikácia** — ten istý text miliónkrát by model naučil memorovať, nie generalizovať.
3. **Mixovanie** — pomery zdrojov sú starostlivo ladené; „dáta sú nový hyperparameter".

> **Prečo je to dôležité pochopiť:** kvalita a zloženie dát vysvetľuje väčšinu rozdielov medzi modelmi. Preto je taký veľký rozdiel medzi *open-weight* (dáta tajné) a *plne open-source* modelmi (dáta verejné) — viď [04-llm-modely.md](04-llm-modely.md). A preto malé modely horšie zvládajú slovenčinu: v mixe jej je málo.

Text sa nakoniec **tokenizuje** (BPE — detailne v [05-embeddings.md](05-embeddings.md)) a rozdelí na bloky dĺžky kontextového okna.

---

## Fáza 1: Pretraining — predikcia ďalšieho tokenu

### Úloha

Model dostane začiatok textu a má predpovedať **ďalší token**. Nič viac. Na výstupe transformera je softmax cez celý slovník — pravdepodobnosť pre každý z ~100 000 tokenov:

```text
vstup:  "Hlavné mesto Slovenska je"
cieľ:   "Bratislava"

model:  P(" Bratislava") = 0.62   ← správny token, chceme čo najvyššie
        P(" Praha")      = 0.05
        P(" krásne")     = 0.03
        ...
```

Loss je **cross-entropy**: `L = −ln P(správny token)`. V príklade `L = −ln(0.62) = 0.48`. Keby model dal správnemu tokenu len 0,01, loss je `−ln(0.01) = 4.6` → veľký gradient → veľká korekcia váh. Presne tá istá mechanika ako pri malej sieti v [01-adam-optimalizator.md](../03-trening-modelov/01-adam-optimalizator.md), len parametrov sú miliardy.

Dve vlastnosti robia z tejto jednoduchej úlohy mimoriadne silný nástroj:

1. **Self-supervised** — labely netreba vyrábať, sú to ďalšie slová samotného textu. Preto sa dá trénovať na biliónoch tokenov: každá pozícia v každom texte je jeden tréningový príklad.
2. **Predpovedať ďalší token dobre = rozumieť** — aby model vedel dokončiť „Násobenie 23 × 17 = ", musí vedieť násobiť. Aby dokončil detektívku vetou „Vrahom je …", musí sledovať dej. Kompresia textu si vynúti model sveta.

### Jeden prechod, veľa tréningových príkladov (teacher forcing)

Keby jeden prechod 8 miliardami váh vyrobil jedinú predikciu, pretraining by bol beznádejne
drahý. Únosný je preto, že **jeden prechod nad blokom textu dá toľko tréningových príkladov,
koľko má blok tokenov**. Ukážme si to na jednej vete:

```text
text:      "Hlavné mesto Slovenska je Bratislava."

vstup  X:  <bos>   Hlavné   ␣mesto      ␣Slovenska  ␣je          ␣Bratislava
cieľ   Y:  Hlavné  ␣mesto   ␣Slovenska  ␣je         ␣Bratislava  .
           └────────── Y je to isté ako X, len posunuté o jednu pozíciu ──────────┘
```

Labely sa teda nevyrábajú — sú to tie isté tokeny, len o krok posunuté (*shifted labels*). Jeden
forward prechod vráti `n` výstupných vektorov a **každý z nich je samostatná predikcia s vlastnou
loss**:

| poz. | čo model na tejto pozícii vidí | má predpovedať | P(správny) | loss `−ln P` |
|---|---|---|---|---|
| 1 | `<bos>` | `Hlavné` | 0,004 | 5,52 |
| 2 | `<bos> Hlavné` | `␣mesto` | 0,21 | 1,56 |
| 3 | `<bos> Hlavné ␣mesto` | `␣Slovenska` | 0,05 | 3,00 |
| 4 | `… ␣mesto ␣Slovenska` | `␣je` | 0,88 | 0,13 |
| 5 | `… ␣Slovenska ␣je` | `␣Bratislava` | 0,62 | 0,48 |
| 6 | `… ␣je ␣Bratislava` | `.` | 0,73 | 0,31 |

Loss celej sekvencie je **priemer** týchto čísel (tu `1,83` nat/token), z neho ide jeden backward
prechod a jeden Adam krok ([lekcia 3](../03-trening-modelov/01-adam-optimalizator.md)). Pri bloku 8192
tokenov je to 8192 predikcií za jeden prechod modelom — odtiaľ tá efektivita. Všimnite si aj to,
čo tabuľka hovorí o pozícii 1: predpovedať prvé slovo textu z ničoho je skoro nemožné (loss 5,52),
kým s kontextom je to ľahké (0,13). Priemerná loss (a teda aj perplexita) je vždy zmes ľahkých
a takmer nemožných pozícií.

**Prečo model pri tom nepodvádza.** Cieľ pozície 4 je `␣je` — a ten token je doslova o riadok
nižšie na vstupe. Jediné, čo bráni modelu prečítať si odpoveď dopredu, je **kauzálna maska
v attention**: skóre voči všetkým pozíciám napravo sa nastaví na `−∞`, takže vektor na pozícii `i`
vidí len tokeny `1…i`. Bez nej by loss spadla na nulu okamžite a model by sa nenaučil nič.

Tomuto postupu — *na vstupe je vždy správna história, nie to, čo model sám vygeneroval* — sa hovorí
**teacher forcing**. Práve preto sa tréning dá robiť nad celým blokom paralelne, kým generovanie
musí ísť token po tokene: pri tréningu pokračovanie poznáme dopredu, pri inferencii nie.

> **A tu je odpoveď na „prečo vznikne len jeden token".** Aj pri generovaní model vyrobí `n`
> výstupných vektorov — lenže predikcie na pozíciách `1…n−1` hovoria, čo malo nasledovať po
> tokenoch, ktoré **už poznáme** (sú to tokeny promptu), takže sa zahodia. Nové je jedine to, čo
> predpovedá **posledná** pozícia. Tá istá sieť, ten istý prechod; líši sa len to, koľko výstupov
> z neho použijeme — pri tréningu všetky, pri inferencii jeden.
> (Rozpísané v [02-transformer-vnutro.md](02-transformer-vnutro.md#6-ako-vznikne-výstupný-token--a-prečo-len-jeden).)

### Ako sa meria pokrok: loss a perplexita

Cross-entropy loss je číslo v *natoch na token* a samo osebe nič nehovorí. Preto sa
prepočítava na **perplexitu**:

```text
PPL = e^loss           (loss je priemerná cross-entropy na token)
```

Perplexita má veľmi konkrétny význam: **medzi koľkými rovnako pravdepodobnými tokenmi model
efektívne váha**. Loss `2.30` → `PPL = 10`, čiže model je na tom tak, ako keby pri každom
tokene hádzal desaťstennou kockou. Loss `0` → `PPL = 1` → dokonalá istota.

| Loss (nat/token) | Perplexita | Ako si to predstaviť |
|---|---|---|
| 6,9 | 1000 | náhodný model nad slovníkom (ešte sa nič nenaučil) |
| 3,0 | 20 | vie gramatiku a bežné frázy |
| 2,0 | 7,4 | slušný model na bežnom webovom texte |
| 1,5 | 4,5 | dnešné veľké modely na dobrých dátach |

Počas pretrainingu sa sleduje **perplexita na oddelenej (held-out) vzorke**, ktorú model
nikdy nevidel — presne ako validačná krivka z [lekcie 3](../03-trening-modelov/02-problemy-pri-uceni.md).
Klesajúca trénovacia a stagnujúca validačná perplexita znamená to isté ako inde: preučenie.

**Dve pasce, na ktoré sa v praxi naráža:**

- **Perplexity sa nedá porovnávať medzi modelmi s rôznym tokenizérom.** Je to
  „prekvapenie na token" — a token je pri každom tokenizéri iný. Model s väčším slovníkom
  vyjadrí tú istú vetu menším počtom ťažších tokenov a jeho PPL bude vyššia, hoci text
  predpovedá presne rovnako dobre. Porovnateľné je až **bits-per-byte** (`loss / ln2`
  prepočítané na bajty textu), ktoré tokenizér vykráti.
- **Nízka perplexita ≠ užitočný asistent.** PPL meria predikciu ďalšieho tokenu, nie
  schopnosť vyriešiť úlohu. Preto sa hotové modely vyhodnocujú **benchmarkmi** (MMLU —
  vedomostné otázky, GSM8K — slovné úlohy z matematiky, HumanEval — programovanie,
  MT-Bench — kvalita dialógu). Tie majú zas vlastnú slabinu: **kontamináciu** — ak sa
  testovacie otázky ocitli v tréningovom korpuse (a web ich obsahuje), model ich má
  zapamätané a skóre nemeria schopnosť, ale memorovanie. Preto je vlastná testovacia sada
  na vlastných dátach cennejšia než tabuľka na leaderboarde ([ako ju zostaviť](07-fine-tuning-lora.md#5-ako-zmerať-či-to-pomohlo)).

### Koľko parametrov a koľko dát: scaling laws a Chinchilla

Empiricky platí, že loss klesá **predvídateľne** s veľkosťou modelu `N`, množstvom tréningových
tokenov `D` a vynaloženým výpočtom `C`. Nie skokovo, ale ako hladká mocninová krivka — a to je
dôvod, prečo sa dá pred spustením tréningu za desiatky miliónov dolárov **dopredu odhadnúť**,
aký dobrý model z neho vyjde.

Výpočet sa pritom dá spočítať jedným vzorcom. Na jeden parameter a jeden token padne pri
tréningu zhruba 6 operácií (2 na forward, 4 na backward):

```text
C ≈ 6 · N · D        [FLOPs]      N = počet parametrov, D = počet tréningových tokenov
```

Kľúčová otázka znie: **keď mám rozpočet `C`, mám radšej väčší model, alebo viac dát?**
Odpoveď dal článok *Chinchilla* (DeepMind, 2022): pri pevnom rozpočte treba `N` aj `D` zväčšovať
**rovnakým tempom**, čo prakticky znamená pomer

```text
D ≈ 20 · N           (~20 tréningových tokenov na každý parameter)
```

Predchádzajúca generácia modelov bola podľa tohto meradla **výrazne pod-trénovaná**:

| Model | Parametre `N` | Tokeny `D` | `D / N` | Poznámka |
|---|---|---|---|---|
| GPT-3 | 175 B | 300 B | 1,7 | pod-trénovaný — za tie peniaze mal byť menší a vidieť viac dát |
| Chinchilla | 70 B | 1,4 T | 20 | compute-optimálny bod |
| Llama 3 8B | 8 B | 15 T | **1875** | zámerne ďaleko za optimom |
| Llama 3.1 405B | 405 B | 15,6 T | 38 | blízko optima |

Prečo idú dnešné malé modely tak ďaleko za „optimum"? Lebo Chinchilla optimalizuje **cenu
tréningu**, ale model sa trénuje raz a **inferuje miliardkrát**. Menší model, ktorý sa učil
oveľa dlhšie, dosiahne rovnakú kvalitu ako väčší compute-optimálny — a potom navždy stojí menej
pamäte a menej času na token. Llama 3 8B je presne tento obchod: drahší tréning výmenou za
model, ktorý sa vojde na jednu kartu. Prínos každého ďalšieho biliónu tokenov pritom klesá,
takže to nie je zadarmo — je to vedomé posunutie nákladov z tréningu do prevádzky.

> **Ako to súvisí s tvarom modelu:** scaling laws hovoria, **koľko** parametrov;
> [pomer šírky a hĺbky](02-transformer-vnutro.md#ako-model-rastie-šírka-vs-hĺbka) hovorí,
> **ako ich usporiadať**. Kvalita je na tvare prekvapivo málo citlivá, preto sa tvar vyberá
> podľa toho, ako dobre sa paralelizuje.

### Ako to beží na tisíckach GPU

Model s 405 miliardami parametrov sa do jednej GPU nezmestí ani náhodou — tréning preto beží
na klastri a delí sa **tromi nezávislými spôsobmi naraz** (v praxi sa kombinujú, tzv. *3D
parallelism*):

| Spôsob | Čo sa delí | Čo sa prenáša medzi GPU |
|---|---|---|
| **Data parallel** | dáta — každá GPU dostane iné mikro-dávky, model má celý | gradienty (all-reduce raz za krok); pri **FSDP/ZeRO** sa delia aj váhy a stav optimalizátora |
| **Tensor parallel** | šírka vrstvy — matice rozrezané medzi GPU ([sekcia 3](02-transformer-vnutro.md#prečo-sa-oplatí-ísť-skôr-do-šírky-problém-s-paralelizáciou)) | aktivácie, 2× za vrstvu (drží sa vnútri jedného uzla, kde je NVLink) |
| **Pipeline parallel** | hĺbka — vrstvy rozdelené medzi GPU | aktivácie na hraniciach stupňov; bublinu prekryjú mikro-dávky |

K tomu tri techniky, bez ktorých by sa to do pamäte nevošlo a ktoré poznáte z lekcie 3:

- **Mixed precision** — počíta sa v `bf16`, ale master kópia váh a stav Adamu ostávajú v `fp32`
  (rozpočet pamäte je v [07-fine-tuning-lora.md](07-fine-tuning-lora.md#1-prečo-sa-celý-model-dotrénovať-nedá)).
- **Gradient accumulation** — efektívna dávka miliónov tokenov sa poskladá z mnohých malých
  krokov bez updatu; až potom sa urobí jeden Adam krok. Veľká dávka je pri LLM nutnosť,
  lebo gradient z pár viet je príliš hlučný.
- **Activation checkpointing** — medzivýsledky sa neukladajú, ale pri backprope prepočítajú:
  ušetrí pamäť za ~30 % výpočtu navyše.

Tréning navyše **padá** — pri tisíckach GPU je výpadok karty bežný jav, preto sa priebežne
ukladajú checkpointy a z posledného sa pokračuje. Jeden beh Llama 3 8B = rádovo
`6 · 8e9 · 15e12 ≈ 7 · 10²³` operácií, čo je rádovo milión GPU-hodín — teda tisíce GPU bežiacich
týždne až mesiace a desiatky miliónov dolárov. Preto pretraining robí pár firiem a všetci ostatní
**stavajú na hotových base/Instruct modeloch**.

### Výsledok: base model

Base model je **dokončovač textu**, nie asistent. Toto treba naozaj pochopiť, lebo vysvetľuje existenciu Fázy 2:

```text
Prompt:  "Napíš báseň o mori."

BASE model (zlé, ale logické):
  "Napíš báseň o jeseni. Napíš báseň o láske. Toto sú typické
   maturitné zadania zo slovenčiny..."
   → nedokončil ÚLOHU, dokončil TEXT — takto podobný text na webe pokračuje
     (zoznamy zadaní), model robí presne to, na čo bol trénovaný.

INSTRUCT model:
  "More šumí do diaľky, vlny spievajú..."
   → pochopil, že prompt je inštrukcia a má ju splniť.
```

Base model má v sebe všetky znalosti a schopnosti — len ich „nepodáva" formou dialógu. (Trik z čias GPT-3: sformulovať úlohu ako text na dokončenie, napr. few-shot príklady. Dnes to za nás rieši Fáza 2.)

### Base model v akcii: iterácie generovania

Teraz tú istú sieť pustíme opačným smerom — bez cieľových tokenov, len s promptom. Každý riadok
tabuľky je **jeden celý prechod** všetkými 32 vrstvami; token, ktorý z neho vzíde, sa pripojí na
koniec sekvencie a stane sa súčasťou vstupu nasledujúceho prechodu:

```text
prompt:  "Napíš báseň o mori."        (base model, žiadna chat šablóna)

iterácia   vstup = celá sekvencia doteraz                        → nový token
───────────────────────────────────────────────────────────────────────────────
    1      Napíš báseň o mori.                                   → "␣Napíš"
    2      Napíš báseň o mori. Napíš                             → "␣báseň"
    3      Napíš báseň o mori. Napíš báseň                       → "␣o"
    4      Napíš báseň o mori. Napíš báseň o                     → "␣jeseni"
    5      Napíš báseň o mori. Napíš báseň o jeseni              → "."
    6      Napíš báseň o mori. Napíš báseň o jeseni.             → "␣Napíš"
    …
   41      … Toto sú typické maturitné zadania zo                → "␣slovenčiny"
    …      (a pokračuje, kým ho nezastaví max_new_tokens)
```

Tri veci, ktoré sa z tabuľky dajú prečítať:

1. **Výstup ide späť na vstup.** Iné spojenie medzi iteráciami neexistuje — model je bezstavový a
   jeho jedinou pamäťou je sekvencia tokenov. (KV cache nie je výnimka: je to len predpočítaná
   podoba tej istej sekvencie, aby sa tokeny `1…n−1` nemuseli rátať znova —
   [02-transformer-vnutro.md](02-transformer-vnutro.md#7-ďalší-prechod-autoregresia-a-kv-cache).)
2. **Model nikdy „nevidí úlohu".** V každej iterácii odpovedá na jedinú otázku: *aký token je
   najpravdepodobnejší ďalej?* A po vete „Napíš báseň o mori." na webe najčastejšie nasleduje
   ďalšie podobné zadanie. Model teda robí presne to, na čo bol trénovaný, a robí to dobre —
   chyba je v tom, čo od neho čakáme.
3. **Nevie prestať.** Pretraining mu nikdy nedal vzor „tu sa odpoveď končí", lebo v surovom texte
   také miesto nie je; token EOS má preto všade nízku pravdepodobnosť. Base model generuje, kým mu
   nedôjde `max_new_tokens` alebo kontextové okno.

Obe posledné vlastnosti — *pokračuj v texte* a *neprestávaj* — treba prepísať. To je presne náplň
Fázy 2.

---

## Fáza 2: SFT / Instruction tuning — z dokončovača asistent

**SFT** (*Supervised Fine-Tuning*), tiež *instruction tuning*, doučí base model na dátach v tvare **inštrukcia → odpoveď**. Výsledok sú modely s príponou `-Instruct` / `-it` / `-chat` na Hugging Face.

### Dáta

Desaťtisíce až milióny ukážkových dialógov. Kde sa berú:

- **ručne písané** ľuďmi (drahé, kvalitné) — otázky, úlohy, ideálne odpovede,
- **syntetické** — generované silnejším modelom a filtrované (dnes prevažujúce; tzv. distillation, viď [05-llm-trendy.md](../05-prakticke/05-llm-trendy.md)),
- reálne konverzácie s asistentom (so súhlasom, filtrované).

Oproti biliónom tokenov pretrainingu je to **maličký dataset** — SFT nemá modelu dodať nové znalosti, len **zmeniť správanie**: „keď vidíš otázku, odpovedz na ňu; odpovedaj v tomto tóne; odmietni škodlivé požiadavky".

### Chat šablóna a špeciálne tokeny

Dialóg sa serializuje do jedného textu pomocou **chat šablóny** so špeciálnymi tokenmi, ktoré oddeľujú role (formát sa líši podľa modelu — preto pri fine-tuningu vždy `tokenizer.apply_chat_template`):

```text
<|system|>Si užitočný asistent.<|end|>
<|user|>Koľko nôh má pavúk?<|end|>
<|assistant|>Pavúk má osem nôh.<|end|>
```

Model sa počas SFT naučí význam týchto tokenov: po `<|assistant|>` nasleduje moja odpoveď, `<|end|>` znamená „dohovoril som". Aj „ukončenie odpovede" je teda naučené správanie — base model by pokračoval donekonečna. (Ako presne taký serializovaný dialóg vyzerá v reálnom modeli a prečo sa rola nedá „podstrčiť" v texte, je v [02-transformer-vnutro.md](02-transformer-vnutro.md#špeciálne-a-chat-tokeny).)

### Tréning: loss len na odpovedi

Mechanika je identická s pretrainingom: celý serializovaný dialóg vojde do modelu ako **jedna
sekvencia**, teacher forcing, shifted labels, cross-entropy, Adam. Jeden prechod opäť dá `n`
predikcií. Rozdiel je jediný, zato zásadný — **loss sa priemeruje len cez tokeny odpovede
asistenta**, ostatné pozície sa vynásobia nulou:

```text
tokeny:   <|user|> Koľko nôh má pavúk ? <|end|> <|assistant|> Pavúk má osem nôh . <|end|>
loss:        ✗      ✗    ✗   ✗    ✗   ✗    ✗          ✗         ✓    ✓    ✓   ✓  ✓    ✓
```

Rozpísané ako pri pretrainingu, len s pridaným stĺpcom „počíta sa loss?":

| poz. | vstupný token | cieľ (nasl. token) | loss? | čo by sa model z tejto pozície naučil |
|---|---|---|---|---|
| 1 | <code>&lt;&#124;user&#124;&gt;</code> | `Koľko` | ✗ | vymýšľať otázky |
| 2 | `Koľko` | `␣nôh` | ✗ | pokračovať v otázke používateľa |
| 3 | `␣nôh` | `␣má` | ✗ | to isté |
| 4 | `␣má` | `␣pavúk` | ✗ | to isté |
| 5 | `␣pavúk` | `?` | ✗ | to isté |
| 6 | `?` | <code>&lt;&#124;end&#124;&gt;</code> | ✗ | ukončovať cudziu repliku |
| 7 | <code>&lt;&#124;end&#124;&gt;</code> | <code>&lt;&#124;assistant&#124;&gt;</code> | ✗ | formát (časť receptov tu loss necháva zapnutú) |
| 8 | <code>&lt;&#124;assistant&#124;&gt;</code> | `Pavúk` | ✓ | **po hlavičke asistenta začni odpovedať** |
| 9 | `Pavúk` | `␣má` | ✓ | obsah a štýl odpovede |
| 10 | `␣má` | `␣osem` | ✓ | obsah a štýl odpovede |
| 11 | `␣osem` | `␣nôh` | ✓ | obsah a štýl odpovede |
| 12 | `␣nôh` | `.` | ✓ | obsah a štýl odpovede |
| 13 | `.` | <code>&lt;&#124;end&#124;&gt;</code> | ✓ | **tu skonči** |

V kóde je to obyčajné pole `labels`, v ktorom sa nepočítané pozície nahradia hodnotou `-100`
(PyTorch `CrossEntropyLoss` ju ignoruje):

```python
input_ids = tok.apply_chat_template(dialog)          # celý dialóg vrátane odpovede
prompt_len = len(tok.apply_chat_template(dialog[:-1], add_generation_prompt=True))

labels = input_ids.clone()
labels[:prompt_len] = -100        # všetko po hlavičku <|assistant|> vrátane sa neučí
```

**Riadok 8 je celá pointa inštrukčného ladenia.** Sekvencia sa tam končí tokenmi
„`… pavúk ? <|end|> <|assistant|>`" a gradient hovorí: *po hlavičke asistenta nasleduje začiatok odpovede*. Presne
tento vzor base modelu chýbal — ten po otázke pokračoval ďalšou otázkou, lebo taký text videl na
webe. Po desaťtisícoch takýchto príkladov je pravdepodobnosť „pokračovať v otázke" na tomto mieste
zatlačená k nule a pravdepodobnosť začiatku odpovede k jednotke.

**Riadok 13 je ten druhý naučený vzor:** po dopovedanej odpovedi nasleduje EOS. Bez neho by model
odpovedal správne — a potom by si sám položil ďalšiu otázku a odpovedal aj na ňu. (Zabudnutý EOS
na konci odpovede je najčastejšia chyba pri vlastnom fine-tuningu a prejaví sa presne takto.)

Prečo sa prompt vôbec maskuje: model sa má naučiť **generovať odpovede**, nie generovať otázky
používateľa. Keby loss bežala všade, minul by časť kapacity na modelovanie toho, ako píšu ľudia —
a pri generovaní by sa to prejavilo sklonom pokračovať v používateľovej replike. (Maskovanie nie je
dogma: niektoré recepty loss na prompte nechávajú so zníženou váhou, lebo pri veľmi malých
datasetoch pôsobí ako regularizácia. Štandard je ale maskovať.)

#### Dve masky, ktoré sa ľahko zamieňajú

V tomto texte vystupujú dve úplne odlišné masky a ich zamieňanie je zdroj zmätku:

| | **Kauzálna maska** (v attention) | **Loss maska** (maskovanie promptu) |
|---|---|---|
| Kde pôsobí | vnútri každej attention vrstvy | až na výstupe, pri výpočte loss |
| Čo robí | zakáže pozerať sa na tokeny **napravo** | vypne učenie na vybraných pozíciách |
| Kedy je aktívna | vždy — pri tréningu aj pri inferencii | len pri tréningu (SFT) |
| Keby chýbala | model by pri tréningu videl odpoveď dopredu a nenaučil sa nič | model by sa učil generovať aj otázky používateľa |

Podstatné: loss maska **nezakrýva prompt pred modelom**. Model sa naň attentionom normálne pozerá
a pozerať sa musí — inak by nemal na čo odpovedať. Maska hovorí len toľko: *za svoju predpoveď na
týchto pozíciách nedostávaš ani odmenu, ani trest.*

### Instruct model v akcii: tá istá otázka, iný priebeh

Pri inferencii sa dialóg zabalí do tej istej šablóny, akú model videl pri tréningu — ale **končí
sa hlavičkou asistenta**, ktorá ostáva otvorená:

```text
vstup (prompt):  <|user|>Koľko nôh má pavúk?<|end|><|assistant|>
                                                   ↑ tu sa text končí, model pokračuje
```

Odtiaľ beží úplne tá istá slučka ako pri base modeli — jeden prechod, jeden token, výstup späť na
vstup:

```text
iterácia   sekvencia na vstupe (skrátene)                       → nový token
─────────────────────────────────────────────────────────────────────────────
    1      <|user|>Koľko nôh má pavúk?<|end|><|assistant|>      → "Pavúk"
    2      …<|assistant|>Pavúk                                  → "␣má"
    3      …<|assistant|>Pavúk má                               → "␣osem"
    4      …<|assistant|>Pavúk má osem                          → "␣nôh"
    5      …<|assistant|>Pavúk má osem nôh                      → "."
    6      …<|assistant|>Pavúk má osem nôh.                     → <|end|>   ← STOP
```

**Iterácia 1 je celý rozdiel medzi base a Instruct modelom.** Vstup sa končí tokenom
`<|assistant|>` a model rieši tú istú úlohu ako vždy — „čo nasleduje?". Po pretrainingu na ňu
nemal jasnú odpoveď; po SFT je to jeden z najčastejších vzorov, aké videl, a to vďaka riadku 8
z tabuľky vyššie. **Iterácia 6** je ten druhý naučený vzor (riadok 13): odpoveď je hotová,
prichádza `<|end|>`, samplovacia slučka ho rozpozná a generovanie zastaví
([kedy sa generovanie zastaví](02-transformer-vnutro.md#kedy-sa-generovanie-zastaví)).

Tie isté dva modely na tom istom prompte vedľa seba:

| | Base model | Instruct model |
|---|---|---|
| Čo dostane na vstup | holý text otázky | otázku v chat šablóne + otvorenú hlavičku <code>&lt;&#124;assistant&#124;&gt;</code> |
| Prvý vygenerovaný token | pokračovanie textu (typicky ďalšia otázka) | prvé slovo odpovede |
| Kedy prestane | keď dôjde `max_new_tokens` | sám, vygenerovaním EOS |
| Prečo | v pretrainingu videl zoznamy zadaní | v SFT videl desaťtisíce vzorov „hlavička → odpoveď → EOS" |

> **Praktický dôsledok:** ak Instruct model odrazu odpovie a potom si sám položí ďalšiu otázku,
> príčina je takmer vždy v **šablóne, nie v modeli** — prompt bol poskladaný ručne, inak než pri
> tréningu (iné tokeny rolí, chýbajúca hlavička asistenta, zdvojený `<|begin_of_text|>`). Preto sa
> vstup nikdy nelepí ako reťazec, ale skladá cez
> `tokenizer.apply_chat_template(..., add_generation_prompt=True)` — ten posledný parameter je
> práve tá otvorená hlavička.

### Výsledok: Instruct model

Po SFT model:

- interpretuje prompt ako úlohu a plní ju,
- drží formát dialógu (role, ukončovanie),
- má natrénovaný štýl a základné odmietanie škodlivých požiadaviek,
- **znalosti má stále z pretrainingu** — SFT ich len sprístupnil formou dialógu.

> **Dôležitý dôsledok pre prax:** fine-tuning je dobrý na **štýl a správanie**, zlý na **vkladanie nových faktov** — fakty sú uložené vo váhach z pretrainingu a malý SFT dataset ich spoľahlivo neprepíše. Na nové/aktuálne fakty použite RAG. (Detailne v [05-llm-trendy.md](../05-prakticke/05-llm-trendy.md) a v [zadaní](../../zadania/RAG_Fine_tunning.md).)

### Váš vlastný fine-tuning = tá istá Fáza 2 v malom

Keď v [zadaní](../../zadania/RAG_Fine_tunning.md) robíte **LoRA/QLoRA** fine-tuning, robíte presne SFT — dvojice otázka → odpoveď, chat šablóna, [loss len na odpovedi](#tréning-loss-len-na-odpovedi) a EOS na jej konci. Rozdiel je len v úspornosti: namiesto všetkých miliárd váh trénujete malé **adaptérové matice** (LoRA) pripojené k zamrznutému modelu, takže to zvládne jedno GPU. Ako presne tie adaptéry vyzerajú a prečo stačia, rozoberá [07-fine-tuning-lora.md](07-fine-tuning-lora.md) (lekcia 7).

---

## Fáza 3: Preferenčné ladenie — RLHF, DPO a reasoning

SFT má jednu zabudovanú slabinu: učí model **napodobniť jednu ukážkovú odpoveď**. Lenže „dobrá
odpoveď" nie je jedna — je ich veľa a líšia sa v odtieňoch, ktoré sa ťažko píšu do zadania
(primeraná dĺžka, opatrnosť pri neistote, odmietnutie škodlivej požiadavky, tón). Navyše je pre
človeka **oveľa jednoduchšie povedať „táto z dvoch je lepšia"** než napísať ideálnu odpoveď.
Na tom stojí celá tretia fáza.

### Dáta: porovnania namiesto vzorov

Model vygeneruje na tú istú otázku dve (alebo viac) odpovedí a anotátor — človek alebo iný
model (*RLAIF*, *Constitutional AI*) — určí, ktorá je lepšia:

```text
prompt:      "Vysvetli, prečo je obloha modrá."
odpoveď A:   dlhý fyzikálny výklad s rozptylom svetla        ← anotátor označí ako lepšiu
odpoveď B:   "Pretože odráža more."                          ← zamietnutá
                          ↓
              dvojica (chosen, rejected)
```

### Cesta A: RLHF (reward model + posilňované učenie)

Klasický postup z ChatGPT má dva kroky:

1. **Reward model.** Kópia modelu, ktorej sa výstupná vrstva vymení za jedno číslo — skóre
   odpovede. Trénuje sa na dvojiciach tak, aby `r(chosen) > r(rejected)`. Vznikne tým
   **naučená náhrada za ľudského hodnotiteľa**, ktorá vie oskórovať ľubovoľnú novú odpoveď.
2. **Optimalizácia politiky (PPO).** Model generuje odpovede, reward model ich hodnotí a
   posilňované učenie posúva váhy k vyššiemu skóre. Kľúčová poistka je **KL-penalizácia**:
   k odmene sa pripočíta trest za to, ako ďaleko sa model vzdialil od SFT verzie.

Bez tej poistky nastáva **reward hacking** — model nájde odpoveď, ktorú reward model hodnotí najvyššie,
hoci je pre človeka nezmyselná (typicky zdvorilé, dlhé a prázdne texty). Reward model je totiž
len model: má svoje slepé miesta a optimalizovať naplno proti nemu znamená nájsť presne tie
miesta.

### Cesta B: DPO (bez reward modelu)

**DPO** (*Direct Preference Optimization*, 2023) si všimol, že sa ten istý cieľ dá dosiahnuť
**obyčajnou loss funkciou priamo na dvojiciach** — bez reward modelu a bez RL slučky. Loss
jednoducho tlačí nahor pravdepodobnosť zvolenej odpovede a nadol pravdepodobnosť zamietnutej,
pričom sa obe merajú **relatívne voči referenčnému (SFT) modelu**, čo plní tú istú úlohu ako
KL-penalizácia v PPO.

| | RLHF (PPO) | DPO |
|---|---|---|
| Koľko modelov treba | politika + reward + referenčný (+ kritik) | politika + zamrznutý referenčný |
| Generovanie počas tréningu | áno — drahé, pomalé | nie, dvojice sú pripravené dopredu |
| Stabilita ladenia | citlivé, veľa hyperparametrov | podstatne jednoduchšie |
| Kde sa dnes používa | veľké laboratóriá | **de facto štandard pre open-weight modely** |

Praktický dôsledok pre vás: ak v `trl` narazíte na `DPOTrainer`, je to presne toto — a spustiť
sa to dá nad LoRA adaptérom na jednej karte, rovnako ako SFT.

### Cesta C: RL na overiteľných úlohách (reasoning modely)

Pri matematike a programovaní netreba ľudské preferencie — **odmena sa dá overiť strojom**:
výsledok sedí / testy prejdú. To je *RLVR* (*RL with verifiable rewards*, algoritmy PPO alebo
GRPO) a je to recept za dnešnými **reasoning modelmi** (o-séria, DeepSeek R1, Claude
s extended thinking). Model je odmeňovaný za správny výsledok a sám si nájde, že sa oplatí
najprv **dlho uvažovať nahlas** a až potom odpovedať — presne to je ten „reťazec uvažovania",
ktorý vidíte ako státisíce tokenov navyše ([prečo premýšľanie = viac tokenov](02-transformer-vnutro.md#prečo-premýšľanie-znamená-viac-tokenov)).

### Čo z toho študent vidí každý deň

Preferenčné ladenie vysvetľuje väčšinu „povahových čŕt" dnešných asistentov:

- **Odmietanie.** Model neodmieta preto, že by mu to zakazoval filter — odmietnutie je
  **naučená odpoveď** s vysokou odmenou. Preto sa dá niekedy „obísť" preformulovaním: nie je to
  pravidlo, je to naučená tendencia.
- **Prispôsobivosť (*sycophancy*).** Ľudia (a teda aj reward model) hodnotia lepšie odpovede,
  ktoré im dávajú za pravdu. Model sa preto učí súhlasiť — a pri „určite? nie je to inak?"
  cúvne aj zo správnej odpovede. Je to priamy dôsledok toho, čím sa odmeňoval.
- **Rozvláčny štýl a „ako jazykový model…"** — tie isté preferencie: dlhšie a opatrnejšie
  odpovede dostávali lepšie hodnotenie.
- **Alignment tax.** Po zarovnaní model na čistých vedomostných testoch typicky mierne
  **stráca** oproti base modelu. Platí sa tým za použiteľnosť a bezpečnosť.

> **Prečo je toto dôležité aj pri vlastnom fine-tuningu:** keď doladíte Instruct model na
> vlastných dátach, prepisujete aj toto zarovnanie. Malý SFT dataset vie modelu „odnaučiť"
> odmietanie aj formát — preto sa po fine-tuningu vždy testuje nielen úloha, ale aj to, či sa
> model nezačal správať inak, než má.

Zhrnutie celej cesty jednou vetou: **pretraining dá modelu schopnosti, SFT z neho urobí
asistenta, preferenčné ladenie ho vycibrí.**

---

## Kontrolné otázky

1. Prečo sa pretraining dá robiť na biliónoch tokenov, hoci nikto tie dáta „nelabeloval"?
2. Jeden forward prechod nad blokom 4096 tokenov — koľko tréningových príkladov z neho vznikne? A koľko predikcií sa použije, keď tým istým modelom nad tým istým blokom **generujete**? Prečo ten rozdiel?
3. Čo je teacher forcing a prečo sa vďaka nemu dá tréning počítať nad celým blokom naraz, kým generovanie musí ísť token po tokene?
4. Base model na prompt „Prelož do angličtiny: pes" odpovie „Prelož do angličtiny: mačka. Prelož do angličtiny: dom." — vysvetlite, prečo je to z pohľadu jeho tréningu *správne* správanie. Čo presne sa musí zmeniť, aby namiesto toho odpovedal „dog"?
5. Čím sa líši SFT od pretrainingu (a) v dátach, (b) v loss funkcii, (c) v cieli? Čo majú mechanicky spoločné?
6. Vysvetlite rozdiel medzi **kauzálnou maskou** a **loss maskou**. Čo by sa pokazilo, keby chýbala prvá? A čo, keby chýbala druhá?
7. Prompt pre Instruct model sa končí tokenom `<|assistant|>`, za ktorým už nič nie je. Prečo je práve toto miesto rozhodujúce a ktorá pozícia v tréningovej sekvencii ho naučila?
8. Váš dolaďovaný model odpovie správne, ale potom si sám položí ďalšiu otázku a odpovie aj na ňu. Uveďte dve možné príčiny — jednu v tréningových dátach, jednu v tom, ako skladáte prompt pri inferencii.
9. Prečo je fine-tuning nevhodný na naučenie modelu nových faktov a čo použiť namiesto neho?
10. Kolega stiahol z Hugging Face `Llama-3.1-8B` (bez prípony) do chatbota a sťažuje sa, že „model odpovedá nezmysly". Čo mu poradíte?
11. Model má na held-out dátach loss `2.0`. Aká je jeho perplexita a čo to číslo znamená? Prečo ju nemôžete porovnať s modelom, ktorý má iný tokenizér?
12. Model dosiahol na MMLU 85 %. Aké dve nezávislé otázky si musíte položiť, než tomu číslu uveríte?
13. Máte rozpočet na `10²³` FLOPov. Podľa Chinchilla pomeru — aký veľký model a koľko tokenov? Prečo sa Llama 3 8B napriek tomu trénovala na 15 biliónoch tokenov?
14. Čo presne je reward model a prečo sa pri PPO pridáva KL-penalizácia voči SFT modelu?
15. Čím sa DPO líši od RLHF v tom, čo všetko musí bežať počas tréningu — a prečo je preto obľúbenejšie pri open-weight modeloch?
16. Prečo model niekedy ustúpi zo správnej odpovede, keď mu používateľ oponuje? Z ktorej tréningovej fázy to pochádza?
17. Prečo sa reasoning modely dajú trénovať posilňovaným učením bez ľudských anotátorov, kým na „buď zdvorilý asistent" to nejde?

---

### Súvisiace dokumenty

- [prehlad-predmetu.md](../../prehlad-predmetu.md) — prehľad celého predmetu (8 lekcií)
- [01-transformer-siete.md](01-transformer-siete.md) — architektúra, ktorá sa tu trénuje (lekcia 4)
- [01-adam-optimalizator.md](../03-trening-modelov/01-adam-optimalizator.md) — tréningová slučka a optimalizátor (rovnaké aj pre LLM)
- [04-llm-modely.md](04-llm-modely.md) — **druhá polovica lekcie 5**: prehľad dnešných modelov
- [05-embeddings.md](05-embeddings.md) — tokenizácia (BPE), z ktorej pretraining vychádza (lekcia 6)
- [07-fine-tuning-lora.md](07-fine-tuning-lora.md) — SFT v malom: LoRA/QLoRA, kedy fine-tuning áno/nie (lekcia 7)
- [zadania/RAG_Fine_tunning.md](../../zadania/RAG_Fine_tunning.md) — vlastný SFT cez LoRA/QLoRA

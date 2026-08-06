# RAG — vyhľadávanie ako pamäť pre LLM

> **Poradie čítania:** ← [Embeddingy](04-embeddings.md) · **lekcia 6** · [Fine-tuning: LoRA a QLoRA](06-fine-tuning-lora.md) →

**RAG** (*Retrieval-Augmented Generation*) rieši jednoduchý problém: jazykový model nepozná vaše dokumenty a doučiť mu ich je drahé a nepružné. Namiesto toho mu ich **podsunieme do promptu** — ale len tie kúsky, ktoré sa práve na otázku hodia. Celé to stojí na vektoroch z [predchádzajúceho dokumentu](04-embeddings.md): keď je otázka aj text uložený ako vektor, „nájdi relevantné" sa zmení na „nájdi najbližšie".

Celá pipeline má dve polovice — jednu, ktorá beží raz dopredu, a druhú, ktorá beží pri každej otázke:

```text
   ┌──────────────────── PRÍPRAVA DÁT (offline, raz / pri zmene) ────────────────────┐
   │                                                                                  │
   │   dokumenty ──► chunking ──► [embedding model] ──► vektory ──► FAISS index        │
   │                              (malý LLM, CPU/GPU)                                  │
   └──────────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────── DOTAZ (online, pri každej otázke) ─────────────────────────┐
   │                                                                                 │
   │   otázka ──► [embedding model] ──► query vektor ──► FAISS top-k                  │
   │                (malý LLM, CPU/GPU)                       │                       │
   │                                                          ▼                       │
   │                                            [reranker / cross-encoder]            │
   │                                             (malý LLM, DRAHÝ – GPU rád)          │
   │                                                          │                       │
   │                                                          ▼                       │
   │                                        top-3 chunky ──► [veľký generatívny LLM]  │
   │                                                          ──► odpoveď             │
   └─────────────────────────────────────────────────────────────────────────────────┘
```

V hranatých zátvorkách sú **modely, ktoré reálne počítajú** (a teda spotrebujú CPU/GPU). Všimnite si, že „malých" modelov je viac a bežia na rôznych miestach – nižšie rozoberieme každý z nich.

Kľúčové je uvedomiť si, že **v RAG bežia typicky až tri modely**, a dva z nich sú „malé" LLM-ká, ktoré napriek tomu **nie sú zadarmo** na výpočet.

---

## 1. Príprava dát (offline fáza)

Toto sa robí **raz** (alebo pri zmene dokumentov) a je to *dávkové* spracovanie:

1. **Extrakcia textu** – z PDF, DOCX, HTML, wiki... získame surový text.
2. **Chunking** – text sa nareže na kúsky (viac nižšie).
3. **Embedding** – každý chunk prejde embedding modelom ([04-embeddings.md](04-embeddings.md)) → vektor.
4. **Indexovanie** – vektory + metadáta (ID chunku, `parent_id`, zdroj, odkaz na text) sa uložia do vektorovej DB (napr. FAISS).

Keďže je to offline a dávkové, dá sa to nechať bežať aj dlhšie na CPU, alebo to výrazne zrýchliť na GPU pri veľkom objeme dokumentov. **Latencia tu nie je kritická**, dôležitý je throughput.

### Chunking – prečo naň záleží

Každý embedding model má **maximálne kontextové okno** (napr. `bge-m3` zvláda do `8192` tokenov, staršie modely len `512`). Ak je chunk dlhší než toto okno, text sa buď **oreže**, alebo to model interne rieši kompromisom (skreslením a stratou informácie z konca).

Chunk nesmie byť ani príliš **veľký**, ani príliš **malý**:

- **Príliš veľký chunk** = mean pooling spriemeruje priveľa rôznych myšlienok do jedného vektora → vektor je „rozmazaný", nesie priemer viacerých tém a nezhoduje sa presne so žiadnou otázkou.
- **Príliš malý chunk** = stratí sa kontext (napr. veta „Je to 25 dní." bez okolia nevie, o čom je reč).

V praxi sa cieľová veľkosť volí niekde v pásme **200–500 tokenov** na chunk (podľa modelu a typu dokumentov).

### Chunking na konkrétnom texte

Zoberme si odsek (predpokladajme, že **1 token ≈ 0,7 slova** pre slovenčinu – teda slovo v priemere ~1,4 tokenu):

```text
"Zamestnanec má nárok na 25 dní platenej dovolenky za kalendárny rok.
Nárok vzniká po odpracovaní 60 dní. Nevyčerpanú dovolenku možno preniesť
do nasledujúceho roka len po dohode so zamestnávateľom. Preplácanie
dovolenky je možné iba pri skončení pracovného pomeru."
```

Nech je cieľ **chunk = 30 tokenov** s **overlapom = 8 tokenov** (prekryv, aby sa neroztrhla myšlienka na hranici). Rozdelenie *s prekryvom* vyzerá takto (čísla sú pozície tokenov):

```text
chunk A: tokeny  0–29    "Zamestnanec … po odpracovaní 60 dní."
chunk B: tokeny 22–51    "… Nevyčerpanú dovolenku možno preniesť … so zamestnávateľom."
chunk C: tokeny 44–70    "… Preplácanie dovolenky je možné iba pri skončení pracovného pomeru."
```

Všimnite si, že tokeny `22–29` sú **v chunku A aj B** – to je tých 8 tokenov overlapu. Prečo? Predstavme si otázku *„Kedy vzniká nárok na dovolenku?"* – odpoveď („po odpracovaní 60 dní") leží presne na hranici. Bez overlapu by sa mohla rozseknúť medzi dva chunky a ani jeden by ju neobsahoval celú. Overlap túto stratu na hraniciach zmierňuje. Cena je **redundancia**: prekrývajúci text sa embedduje a ukladá viackrát (pri overlape 8 z 30 tokenov je to ~27 % dát navyše).

### Stratégie delenia (od najhoršej po najlepšiu)

| Stratégia | Ako reže | Riziko |
|---|---|---|
| **Fixed-size** | naslepo po N tokenov | reže uprostred vety/slova |
| **Sentence-aware** | na hraniciach viet (podľa `.`, `?`, `!`) | vety rôznej dĺžky |
| **Recursive** | skúša deliť po odsekoch → vetách → slovách, kým sa nezmestí | najbežnejší kompromis |
| **Semantic** | reže tam, kde sa mení téma (podľa poklesu podobnosti susedných viet) | drahšie, ale najčistejšie hranice |

> **Preto:** veľkosť chunku treba prispôsobiť **konkrétnemu** embedding modelu, ktorý sa použije. Je nutné **vopred vedieť presnú špecifikáciu modelu** od toho, kto vektorovú DB pripravuje. (A nezabudnite na postreh z [tokenizácie](04-embeddings.md#krok-1-tokenizácia) – slovenský text zaberie viac tokenov, takže reálne sa doň zmestí menej textu, než by sa zdalo.)

### Metadáta – čo sa ukladá popri vektore

K vektoru sa **nikdy** neukladá len samotné pole čísel. Vektor je „adresa" v priestore, ale na zostavenie odpovede treba vedieť, **z čoho pochádza**. Typický záznam v indexe:

| pole | príklad hodnoty | načo |
|---|---|---|
| `id` | `doc42_chunk_03` | jednoznačný identifikátor chunku |
| `vector` | `[0.587, 0.440, …]` | to, v čom sa vyhľadáva |
| `text` | `"Nevyčerpanú dovolenku možno…"` | pôvodný text – vloží sa do promptu LLM |
| `source` | `zakonnik_prace.pdf` | odkiaľ to je (citácia pre používateľa) |
| `page` | `12` | číslo strany / sekcie |
| `parent_id` | `doc42_sec_dovolenka` | odkaz na väčší nadradený blok (viď nižšie) |
| `token_count` | `27` | kontrola, či sa chunk zmestil do okna |

Vyhľadávanie beží nad `vector`, ale používateľovi sa vracia `text` + `source` + `page`. **Bez metadát by RAG vedel nájsť relevantný vektor, ale nevedel by, aký text ani odkiaľ ho ukázať.**

### Parent-child chunking

Šikovná technika, ktorá spája výhody malých aj veľkých chunkov:

1. **Vyhľadáva sa** cez **malé** child-chunky (presné, ostrý vektor).
2. **Do promptu LLM sa ale vloží** ich **veľký** parent-chunk (širší kontext okolo nájdeného miesta).

Príklad: child-chunk „nárok vzniká po odpracovaní 60 dní" sa vo vyhľadávaní trafí presne, ale cez `parent_id` sa do promptu dotiahne celý odsek o dovolenke, aby mal generatívny LLM dosť kontextu na plnú odpoveď. V indexe teda `parent_id` prepája child záznam s uloženým textom rodiča.

## 2. Dotaz (online fáza)

Toto sa deje **pri každej otázke používateľa** a tu už **latencia záleží** – používateľ čaká na odpoveď:

1. **Embedding otázky** – tá istá cesta ako pri chunkoch ([04-embeddings.md](04-embeddings.md)), ale len pre jednu krátku vetu → *query vektor*. Keďže je to bi-encoder, chunky boli zaembeddované vopred, teraz sa počíta iba embedding query.
2. **Vyhľadanie top-k** – vo FAISS sa nájde napr. `top-20–50` najbližších vektorov (rýchle, čistá lineárna algebra / ANN index).
3. **Reranking (voliteľné, ale veľmi účinné)** – užší set kandidátov prejde cross-encoderom, ktorý vyberie skutočný `top-3–5`.
4. **Generovanie odpovede** – vybrané chunky sa vložia do promptu a **veľký generatívny LLM** vygeneruje odpoveď.

### Ako vyzerá index a ako sa v ňom hľadá

FAISS index je zjednodušene **matica uložených vektorov** `[počet_chunkov × dim]` plus mapovanie riadok → `id` chunku. Predstavme si index so 4 chunkami (dim = 3, vektory sú L2-normalizované, aby dot = cosine):

```text
riadok | id            | vektor
   0   | chunk_dovolenka | [ 0.80, 0.55, 0.20 ]
   1   | chunk_mzda      | [ 0.10, 0.30, 0.95 ]
   2   | chunk_nadcas    | [ 0.60, 0.70, 0.35 ]
   3   | chunk_vypoved   | [ 0.20, 0.10, 0.97 ]
```

Príde otázka *„Koľko dní dovolenky mám?"*, zaembedduje sa ([rovnakým modelom](04-embeddings.md)) a znormuje na query vektor:

```text
q = [ 0.78, 0.60, 0.18 ]
```

**Vyhľadanie top-k = spočítaj podobnosť q voči každému riadku a zoraď.** S dot productom (= cosine, lebo je všetko normované):

```text
q · chunk_dovolenka = 0.78·0.80 + 0.60·0.55 + 0.18·0.20 = 0.624 + 0.330 + 0.036 = 0.990
q · chunk_mzda      = 0.78·0.10 + 0.60·0.30 + 0.18·0.95 = 0.078 + 0.180 + 0.171 = 0.429
q · chunk_nadcas    = 0.78·0.60 + 0.60·0.70 + 0.18·0.35 = 0.468 + 0.420 + 0.063 = 0.951
q · chunk_vypoved   = 0.78·0.20 + 0.60·0.10 + 0.18·0.97 = 0.156 + 0.060 + 0.175 = 0.391
```

Zoradené zostupne: `dovolenka (0.990) > nadcas (0.951) > mzda (0.429) > vypoved (0.391)`. Pri **top-2** vráti index `chunk_dovolenka` a `chunk_nadcas`. Presne toto je celé „vyhľadávanie" – žiadne kúzlo, len `N` dot productov a zoradenie.

### Flat vs. ANN – prečo nie vždy počítame všetkých N

To, čo sme práve spravili (porovnať query so **všetkými** vektormi), je **brute-force / flat** index (`IndexFlatIP`, `IndexFlatL2`). Je **presný**, ale je `O(N·dim)` na dotaz – pri miliónoch chunkov to je pri každej otázke priveľa.

Preto existujú **ANN** indexy (*Approximate Nearest Neighbor*), ktoré obetujú štipku presnosti za obrovské zrýchlenie:

| Index | Princíp | Kompromis |
|---|---|---|
| **IndexFlat** | porovná všetkých N | presný, pomalý pri veľkom N |
| **IVF** (inverted file) | vektory sa rozdelia do `nlist` zhlukov (k-means); pri dotaze sa prehľadá len `nprobe` najbližších zhlukov | rýchly; môže minúť suseda za hranicou zhluku |
| **HNSW** (graf) | vektory sú uzly grafu, hľadá sa „skákaním" po najbližších susedoch | veľmi rýchly, vyššia pamäť |
| **PQ** (product quantization) | vektory sa komprimujú na pár bajtov | úspora pamäte, mierna strata presnosti |

Príklad IVF: pri `nlist = 100` zhlukoch a `nprobe = 5` sa namiesto všetkých N vektorov porovná len ~5 % z nich → ~20× rýchlejšie, za cenu drobnej pravdepodobnosti, že sa najbližší sused mimo prehľadaných zhlukov prehliadne. Práve preto sa robí **reranking** (ďalší krok) – ANN vytiahne širší, trochu „hrubý" `top-k`, a presný cross-encoder ho dočistí.

---

## 3. Bi-encoder vs. cross-encoder (reranker) – dva rôzne „malé" modely

Toto je kľúčové rozlíšenie, lebo vysvetľuje, prečo je jeden malý model lacný a druhý drahý.

| | **Bi-encoder** (embedding model) | **Cross-encoder** (reranker) |
|---|---|---|
| Ako spracuje vstup | otázku a chunk kóduje **nezávisle**, každý sám prejde celým procesom | otázku aj chunk **spojí do jedného vstupu** `[CLS] otázka [SEP] chunk [SEP]` a prejdú self-attention **spolu** |
| Vidí interakciu slov? | nie – až na konci porovná dva hotové vektory | áno – priama interakcia slov otázky a chunku už v attention |
| Dá sa predpočítať? | **áno** – embeddingy chunkov sa spočítajú vopred a uložia do FAISS | **nie** – musí sa počítať znova pre **každý pár** (otázka, chunk) |
| Presnosť | dobrá | **vyššia** |
| Cena pri dotaze | lacná (1× embedding query) | **drahá** (`k`× priebeh modelu) |
| Kde sa použije | na celú databázu (retrieval) | len na užší `top-k` z retrievalu |

**Praktický dôsledok:** reranker sa **nikdy** nepúšťa na celú databázu – bežal by pri každom dotaze `N`-krát (raz za každý chunk v DB). Preto sa najprv lacným bi-encoderom vytiahne širší set a **až ten** sa preženie drahým rerankerom. Reranking býva jedno z najlacnejších a najúčinnejších vylepšení kvality RAG – ale „lacné" je myslené na *implementáciu*, nie na *výpočet*.

---

## 4. Reranking – čo to je a kedy sa oplatí

**Čo je reranking.** Vyhľadanie cez bi-encoder (`top-k` z FAISS) je **rýchle, ale hrubé** – zoraďuje podľa podobnosti dvoch *nezávisle* spočítaných vektorov, takže niekedy vytiahne chunk, ktorý je len povrchovo podobný (spoločné slová), no na otázku vlastne neodpovedá. **Reranking je druhý, presnejší priechod**, ktorý tento zoznam kandidátov **preusporiada** podľa skutočnej relevancie k otázke. Robí ho **cross-encoder** (viď sekcia 3 vyššie): každú dvojicu *(otázka, kandidát)* prečíta **spolu** a dá jej skóre relevancie; podľa tých skóre sa kandidáti zoradia nanovo a do promptu ide finálny `top-3–5`.

Kľúčové je poradie krokov – **dvojfázový retrieval**:

```text
otázka
  │
  ├─(1) bi-encoder + FAISS ──►  top-k kandidátov (napr. 20–50)   ← rýchle, hrubé
  │
  └─(2) cross-encoder rerank ─►  preusporiadať, vziať top-3–5    ← pomalé, presné
                                  │
                                  └──►  do promptu pre generatívny LLM
```

Fáza 1 zúži milióny chunkov na desiatky (lacno). Fáza 2 tých pár desiatok **dôkladne prehodnotí** (draho, ale už len `k`-krát). Bez fázy 1 by bol reranker neúnosne drahý (bežal by `N`×), bez fázy 2 zas do promptu prepadnú „falošne podobné" chunky.

**Prečo to zvyšuje kvalitu.** Do generatívneho LLM sa zmestí len pár chunkov. Ak je medzi nimi ten správny, ale až na 8. mieste, a vy berete `top-5`, **odpoveď v kontexte vôbec nie je** a model buď mlží, alebo povie „neviem". Reranker ten správny chunk posunie z 8. na 1.–2. miesto → **recall v rámci malého okna sa zásadne zlepší** (metrika *nDCG* / *recall@k*).

### Kedy má reranking zmysel

- **Otázky vyžadujú porozumenie, nie len zhodu slov** – parafrázy, súvislosti, „prečo/ako". Tu čistý bi-encoder najviac chybuje.
- **Veľká alebo šumivá databáza** – veľa chunkov, ktoré sú si navzájom podobné; treba jemne rozlíšiť, ktorý *naozaj* odpovedá.
- **Malé okno kontextu / drahý generatívny LLM** – keď si môžete dovoliť poslať len 3–5 chunkov, kvalita tých pár rozhoduje o všetkom.
- **Používate hybridný alebo ANN retrieval** – kombinujete BM25 + vektory alebo ANN index (IVF/HNSW), ktorý vracia širší, „hrubší" set; reranker ho dočistí.
- **Kvalita odpovede je dôležitejšia než pár desiatok ms latencie** – interné vyhľadávanie, právo, medicína, podpora.

### Kedy sa (zatiaľ) neoplatí

- **Malá databáza a jasné, kľúčovkové otázky** – ak `top-5` z bi-encodera už spoľahlivo obsahuje odpoveď, reranker nič nepridá, len pridá latenciu.
- **Prísny latency rozpočet bez GPU** – cross-encoder na CPU vie pridať stovky ms až sekundy na dotaz (viď nižšie); v real-time chate to môže byť neúnosné.
- **Málo kandidátov** – rerankovať `top-3` nemá zmysel, keď aj tak všetky tri idú do promptu.
- **Skôr riešte základy** – ak je slabý **chunking** alebo nevhodný **embedding model**, reranker to nezachráni; najprv opravte fázu 1.

> **Pravidlo palca:** začnite **bez** rerankera (bi-encoder + `top-5`) a zmerajte kvalitu. Ak sa ukáže, že správny chunk *sa vyhľadá, ale je príliš nízko* (je v `top-20`, ale nie v `top-5`), pridajte reranker – vytiahnite `top-20–50` a nechajte ho vybrať finálnych 3–5. To je presne situácia, keď reranking dáva najväčší zisk za najmenšiu prácu.

**Voľba modelu.** Bežné rerankery: `cross-encoder/ms-marco-MiniLM-L-6-v2` (rýchly, anglický), `BAAI/bge-reranker-v2-m3` (viacjazyčný, aj slovenčina), `jina-reranker`. Používajú sa cez `sentence-transformers` (`CrossEncoder`) – dostanú zoznam dvojíc *(otázka, chunk)* a vrátia skóre.

---

## 5. Výpočtové nároky: kde to tlačí na CPU/GPU

Zhrnutie, prečo aj „malé" modely reálne potrebujú výkon:

- **Kde je záťaž:** drvivá väčšina výpočtu je v **transformer vrstvách** – maticové násobenia Q/K/V, self-attention `O(n²)` a feed-forward vrstvy. Tokenizácia a lookup v embedding matici sú zanedbateľné, pooling a normalizácia tiež (rozpísané krok po kroku v [04-embeddings.md](04-embeddings.md)).

- **Embedding model (bi-encoder) – CPU zvládne, GPU zrýchli:**
  - *Offline indexovanie* je dávkové → CPU stačí, GPU sa oplatí len pri veľkých objemoch (throughput).
  - *Query embedding pri dotaze* je jedna krátka veta → na CPU rádovo desiatky ms, čo býva OK.

- **Reranker (cross-encoder) – tu GPU dáva najväčší zmysel:**
  - Beží `k`-krát pri **každom** dotaze (napr. 20–50× priebeh modelu na jednu otázku).
  - Vstup je dlhší (otázka **+** celý chunk spolu), takže `n` je väčšie a `O(n²)` attention bolí.
  - Na CPU to vie pridať stovky ms až sekundy na dotaz; na GPU je to prijateľné.
  - **Toto je typicky prvý kandidát na GPU** v RAG systéme.

- **Batchovanie:** modely bežia efektívnejšie, keď spracúvajú viac vstupov naraz (jeden veľký maticový výpočet). Pri indexovaní sa to využíva prirodzene; pri online dotaze menej (jedna otázka), preto tam pomáha práve GPU alebo aspoň dobre nastavené vlákna na CPU.

- **Kvantizácia (INT8/FP16):** malé modely sa dajú kvantizovať, čím klesne pamäť aj výpočet a na CPU to beží citeľne rýchlejšie – za cenu malej straty presnosti. Bežný kompromis pri lokálnom nasadení.

> **Zhrnutie pre nasadenie:** embedding model rád beží aj na CPU (najmä query pri dotaze), reranker si o GPU priam pýta, a veľký generatívny LLM je úplne iná váhová kategória (rieši sa samostatne – lokálne GPU alebo API). Pri plánovaní hardvéru pre RAG počítajte s tým, že **„malé modely" sú malé len v porovnaní s generatívnym LLM** – na CPU sú stále citeľnou záťažou, hlavne reranker pri každom dotaze.

---

## 6. Pokročilý retrieval — kam sa RAG posunul

Sekcie 1 a 2 opisujú **základnú pipeline**, ktorá stačí na zadanie aj na väčšinu firemných nasadení: chunkovať → embeddovať → hľadať top-k → prípadne rerankovať → generovať. Nasledujúce techniky riešia jej konkrétne slabiny. Nasadzujte ich **až keď zmeriate, že základ nestačí** – každá pridáva latenciu aj kód.

### 1. Hybridné vyhľadávanie (vektor + BM25)

Vektorové vyhľadávanie chytá **význam**, ale zlyháva na presných reťazcoch: kódy dielov (`XR-4420`), skratky, priezviská, čísla zmlúv. Tie sú pre embedding model takmer nerozlíšiteľné – v natrénovanom priestore ležia všetky „nezmyselné" reťazce blízko seba. Klasické lexikálne vyhľadávanie **BM25** (počíta zhodu slov s váhou podľa ich zriedkavosti) ich naopak trafí presne, ale nepozná synonymá.

**Hybrid search** pustí obe a výsledky zlúči – najčastejšie cez **RRF** (*Reciprocal Rank Fusion*): dokument dostane skóre `Σ 1/(k + poradie)` z každého zoznamu, takže sa nemusia porovnávať navzájom nekompatibilné skóre. Natívne to podporujú Elasticsearch/OpenSearch, Qdrant, Weaviate aj `pgvector` v kombinácii s full-textom Postgresu.

> Praktické pravidlo: ak sa v dokumentoch vyskytujú **identifikátory, ktoré musí používateľ nájsť doslova**, hybrid je prvá vec, ktorú pridáte – prináša väčší zisk než ladenie chunkov.

### 2. Prepis dotazu (query transformation)

Používateľská otázka často nevyzerá ako text, ktorý hľadáme:

- **Rozšírenie / prepis** – LLM otázku preformuluje do podoby bližšej dokumentom (doplní synonymá, odborný termín).
- **HyDE** (*Hypothetical Document Embeddings*) – LLM najprv **vymyslí hypotetickú odpoveď**, tá sa zaembedduje a hľadá sa podľa nej. Hľadáme tak odpoveď podobnú odpovedi, nie odpoveď podobnú otázke – čo je geometricky bližšie.
- **Rozklad na podotázky** – zložená otázka („Ako sa líši nárok na dovolenku u nás a v zmluve X?") sa rozbije na samostatné dotazy a výsledky sa spoja (*multi-hop*).

Cena je vždy jedno LLM volanie navyše pred vyhľadávaním.

### 3. Filtrovanie podľa metadát

Vektorové hľadanie sa dá skombinovať so **štruktúrovaným filtrom** nad metadátami zo sekcie 1 (`source`, `page`, dátum, oddelenie, prístupové práva). Dva typické dôvody: zúženie na relevantnú podmnožinu („len smernice platné v roku 2026") a **oprávnenia** – používateľ nesmie dostať do odpovede chunk z dokumentu, na ktorý nemá prístup. Toto je bezpečnostná, nie kvalitatívna vlastnosť, a rieši sa vo vektorovej DB, nie v prompte.

### 4. Small-to-big: varianty parent-child

Princíp poznáme zo sekcie 1 (hľadaj malým, vkladaj veľký). V praxi sa objavuje v troch podobách:

| Technika | Ako funguje |
|---|---|
| **Parent-child** | child chunk má v metadátach `parent_id`; po nájdení sa dotiahne rodič |
| **Sentence-window** | indexujú sa jednotlivé vety, vracia sa okno ±N viet okolo nájdenej |
| **Auto-merging** | hierarchia chunkov; ak sa trafí dosť súrodencov pod jedným rodičom, vráti sa rovno rodič |

Hotové implementácie: `ParentDocumentRetriever` v LangChain, `AutoMergingRetriever` v LlamaIndex. Vo všetkých prípadoch ide o vzor na úrovni aplikácie (vektor → `id` → dotiahnutie plného textu), nie o vlastnosť databázy: malé chunky idú do vektorovej DB, plné texty do obyčajného úložiska.

### 5. Agentický RAG

Doteraz bol retrieval **pevný**: vyhľadaj raz, vlož do promptu, generuj. Agentický RAG necháva rozhodovanie na modeli – **či** vôbec hľadať, **čo** hľadať, **koľkokrát** (nájde niečo, zistí, že to nestačí, hľadá znova inak), **kde** (vektorová DB / SQL / web) a **kedy má dosť** informácií. Príbuzný vzor **self-check**: model si navrhnutú odpoveď spätne overí voči zdrojom a pri nezhode hľadá znova.

Zaplatí sa za to viacerými LLM volaniami na jednu otázku – teda latenciou, cenou a podstatne ťažším ladením. Oplatí sa pri komplexných otázkach cez viacero zdrojov. Samotná agentová slučka, ktorá to poháňa, je témou [lekcie 8](../05-prakticke/01-agenti-a-nastroje.md).

---

## TL;DR

- **Chunking**: cieľ ~200–500 tokenov, **overlap** proti roztrhnutiu myšlienky na hranici, **metadáta** (`text`, `source`, `parent_id`…) sa ukladajú popri vektore, **parent-child** = hľadaj malým, vkladaj veľký.
- **Vyhľadávanie** = `N` dot productov + zoradenie (flat index), alebo **ANN** (IVF/HNSW/PQ) pre veľké `N` – rýchlejšie za cenu drobnej straty presnosti; preto sa dočisťuje rerankerom.
- V RAG bežia **tri modely**: embedding (lacný, bi-encoder), reranker (drahý, cross-encoder, beží `k`× na dotaz), generatívny LLM (samostatná kategória).
- Výpočet drží **transformer vrstvy** (`O(n²)` attention + maticové násobenia). Embedding zvládne **CPU**, reranker si pýta **GPU**.
- Keď základ nestačí: **hybrid** (vektor + BM25) na presné kódy a mená, **prepis dotazu / HyDE** na zle formulované otázky, **filtre nad metadátami** na oprávnenia, **agentický RAG** na zložené otázky — všetko za cenu latencie.

---

## Kontrolné otázky

1. Kolega navrhuje chunky po 5 000 tokenov, „aby sa nič nestratilo". Vysvetlite mu dva problémy, ktoré tým vzniknú.
2. Otázka „Kedy vzniká nárok na dovolenku?" má odpoveď presne na hranici dvoch chunkov. Ktorý mechanizmus z tohto dokumentu problém rieši a ako?
3. Prečo sa cross-encoder (reranker) nikdy nepúšťa na celú databázu, ale bi-encoder áno? (Kľúč: čo sa dá predpočítať.)
4. Zaindexovali ste databázu modelom A a otázky embeddujete modelom B (rovnaká dimenzia výstupu). Prečo vyhľadávanie vráti nezmysly?
5. Kedy sa oplatí ANN index (IVF/HNSW) namiesto flat indexu a čím za to platíte?
6. Používatelia sa sťažujú, že RAG nenájde dokument, keď zadajú presné číslo zmluvy `ZML-2024/118`. Prečo na tom vektorové vyhľadávanie zlyháva a čím to opravíte?
7. Čo robí HyDE a prečo môže hľadanie podľa vymyslenej odpovede fungovať lepšie než hľadanie podľa otázky?
8. Ktoré tri modely v RAG pipeline bežia a ktorý z nich si pýta GPU?

---

### Súvisiace dokumenty

- [04-embeddings.md](04-embeddings.md) — ako vzniká vektor, ktorý sa tu indexuje
- [01-transformer-siete.md](01-transformer-siete.md) — attention mechanika, ktorá beží vo vnútri
- [02-llm-trening.md](02-llm-trening.md) — ako sa trénuje generatívny LLM na konci pipeline
- [06-fine-tuning-lora.md](06-fine-tuning-lora.md) — **nasleduje**: druhá cesta k tomu istému cieľu
- [01-agenti-a-nastroje.md](../05-prakticke/01-agenti-a-nastroje.md) — agentová slučka za agentickým RAG
- [zadania/RAG_Fine_tunning.md](../../zadania/RAG_Fine_tunning.md) — **zadanie 2A**: postaviť túto pipeline

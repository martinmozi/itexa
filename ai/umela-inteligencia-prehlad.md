# Umelá inteligencia — prehľad prístupov a modelov

> **Cieľ dokumentu:** dať ucelený, ale prakticky ladený prehľad toho, čo je umelá inteligencia, ako sa delí, a aké hlavné rodiny modelov sa dnes používajú — od rozhodovacích stromov a XGBoostu cez feed-forward a konvolučné neurónové siete až po genetické algoritmy. Ku každému modelu je obrázok, typické použitie (tabuľkové dáta, spracovanie obrazu…) a stručné výhody/nevýhody.

Transformery a mechanizmus attention majú vlastný, detailnejší dokument: [transformer-siete.md](transformer-siete.md).

---

## Čo je umelá inteligencia

**Umelá inteligencia (AI)** je široký pojem pre systémy, ktoré riešia úlohy, na aké by sme u človeka povedali, že vyžadujú „inteligenciu": rozpoznať objekt na fotke, preložiť vetu, naplánovať trasu, odporučiť film alebo hrať šach. AI nie je jedna konkrétna technológia — je to skôr **cieľ** (napodobniť rozumné správanie), ku ktorému vedie viacero rôznych ciest.

Historicky sa vyvinuli dva veľké prúdy:

1. **Symbolická AI** (staršia, „Good Old-Fashioned AI"). Znalosti a pravidlá do systému **vloží človek** vo forme explicitných pravidiel typu „ak–tak", logických výrokov, rozhodovacích tabuliek alebo prehľadávania stavového priestoru. Príklady: expertné systémy pre diagnostiku, šachové enginy so stromom ťahov, plánovače, pravidlové chatboty. Výhoda: je to **vysvetliteľné** a predvídateľné. Nevýhoda: pravidiel je pri reálnych problémoch priveľa a niektoré veci (napr. „čo je na obrázku mačka") sa pravidlami napísať prakticky nedajú.

2. **Strojové učenie (Machine Learning, ML)** (dominantné dnes). Systém sa **naučí vzory priamo z dát**, namiesto toho, aby mu ich niekto naprogramoval. Ukážeme mu tisíce príkladov a on si sám nastaví vnútorné parametre tak, aby dobre predpovedal. Sem patria stromy, XGBoost aj celé hlboké učenie.

### Taxonómia — ako do seba veci zapadajú

![Taxonómia umelej inteligencie: AI obsahuje symbolickú AI a strojové učenie; strojové učenie obsahuje klasické metódy a neurónové siete; neurónové siete obsahujú hlboké učenie](images/ai-taxonomia.svg)

Kľúčové je pochopiť **vzťah vnorenia**: hlboké učenie je podmnožinou neurónových sietí, tie sú podmnožinou strojového učenia a to je podmnožinou AI. Bežná chyba je používať „AI" a „neurónové siete" ako synonymá — v skutočnosti je neurónová sieť len jeden (dnes veľmi úspešný) nástroj vo veľkej škatuli AI.

**Genetické algoritmy** stoja trochu bokom: nie sú to klasifikátory ani siete, ale **optimalizačná metóda** inšpirovaná evolúciou. Dajú sa použiť aj na trénovanie/ladenie iných modelov, preto ich spomíname v tomto prehľade samostatne na konci.

---

## Strojové učenie — tri základné režimy

Podľa toho, aké dáta máme k dispozícii a čo od modelu chceme, rozlišujeme tri hlavné režimy učenia:

| Režim | Čo máme | Čo sa učí | Typický príklad |
|---|---|---|---|
| **Učenie s učiteľom** (*supervised*) | vstupy **aj správne odpovede** (labely) | mapovanie vstup → výstup | „táto fotka = mačka", predikcia ceny bytu |
| **Učenie bez učiteľa** (*unsupervised*) | len vstupy, **bez labelov** | štruktúra, zhluky, podobnosti | segmentácia zákazníkov, [embeddingy](embeddings.md) |
| **Posilňované učenie** (*reinforcement*) | prostredie + **odmena** za akcie | stratégia (politika) maximalizujúca odmenu | hra Go, riadenie robota, [demo s tankom](../demo/Readme.md) |

Väčšina modelov v tomto dokumente (stromy, XGBoost, klasifikačné siete) sú príklady **učenia s učiteľom**. Spoločná schéma je vždy rovnaká:

```text
  trénovacie dáta ──►  MODEL  ──► predpoveď
                         ▲            │
                         │            ▼
                    úprava parametrov ◄── porovnaj s pravdou (loss)
```

Model urobí predpoveď, porovná ju so správnou odpoveďou (chyba = *loss*), a upraví svoje parametre tak, aby chyba klesala. Toto sa opakuje na tisícoch príkladov. Detailne je tréningová slučka a optimalizátor rozpísaný v [adam-optimalizator.md](adam-optimalizator.md).

Ešte jedno praktické rozdelenie dát, ktoré sa ťahá celým ML:

- **Tabuľkové dáta** — riadky a stĺpce (Excel, databáza): vek, príjem, počet klikov… Tu dnes **kraľujú stromové metódy a XGBoost**.
- **Neštruktúrované dáta** — obraz, zvuk, text, video. Tu **kraľuje hlboké učenie** (CNN pre obraz, transformery pre text).

Toto rozlíšenie je najdôležitejšia intuícia pri výbere modelu, preto sa k nemu budeme vracať pri každej rodine.

---

## 1. Rozhodovacie stromy

**Rozhodovací strom** rozdeľuje dáta sériou jednoduchých otázok typu „je príjem väčší ako 1500 €?". Každá otázka rozdelí dáta na dve vetvy; postupným vetvením sa dopracujeme k listu, ktorý obsahuje predpoveď.

![Rozhodovací strom pre schválenie úveru: koreň sa pýta na príjem, vnútorné uzly na vek a ručiteľa, listy hovoria schváliť alebo zamietnuť](images/rozhodovaci-strom.svg)

**Ako sa učí:** algoritmus v každom uzle vyskúša možné otázky (splity) a vyberie tú, ktorá dáta najlepšie „vyčistí" — teda po rozdelení sú skupiny čo najviac homogénne (jedna vetva prevažne „schváliť", druhá prevažne „zamietnuť"). Miera nečistoty sa meria napr. **Gini indexom** alebo **entropiou**. Vetvenie pokračuje, kým nie sú listy dostatočne čisté alebo kým sa nedosiahne maximálna hĺbka.

**Typické použitie:** tabuľkové dáta — schvaľovanie úverov, medicínska triáž, jednoduché pravidlové rozhodovanie, kde chceme, aby sa výsledok dal ukázať a obhájiť.

| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Veľmi **vysvetliteľné** — cestu k rozhodnutiu vie prečítať aj laik | Jeden strom ľahko **preučí** (overfitting) — zapamätá si šum v dátach |
| Netreba škálovať ani normalizovať vstupy | **Nestabilný** — malá zmena dát môže dať úplne iný strom |
| Zvláda číselné aj kategorické atribúty | Sám osebe má **nižšiu presnosť** ako ansámble |
| Rýchle trénovanie aj predikcia | Nevie dobre modelovať plynulé, „šikmé" hranice |

> Práve nestabilita a náchylnosť na preučenie viedli k tomu, že sa jednotlivé stromy skladajú do **ansámblov** — random forest a boosting.

---

## 2. Random Forest a XGBoost (ansámble stromov)

Namiesto jedného stromu sa použije **veľa stromov naraz** a ich predpovede sa skombinujú. Existujú dve hlavné stratégie, ako to spraviť — a je dobré vidieť ich vedľa seba:

![Porovnanie random forest a XGBoost: random forest učí stromy nezávisle a paralelne a spriemeruje ich, XGBoost učí stromy postupne, pričom každý opravuje chyby predchádzajúcich](images/ensemble-forest-boosting.svg)

### Random Forest (bagging)

Natrénuje sa mnoho stromov **nezávisle a paralelne**, každý na inom náhodnom podvzorku dát a stĺpcov. Finálna predpoveď je **priemer** (regresia) alebo **hlasovanie** (klasifikácia). Keďže sa chyby jednotlivých stromov navzájom „vyrušia", výsledok je oveľa stabilnejší než jeden strom. Random forest hlavne **znižuje rozptyl** (variance).

### XGBoost (gradient boosting)

**Boosting** ide na to opačne: stromy sa učia **postupne**, jeden po druhom. Prvý strom dá hrubý odhad, ďalší strom sa učí **opravovať chyby (rezíduá)** predchádzajúcich, a tak ďalej. Finálna predpoveď je **súčet** všetkých stromov. Boosting hlavne **znižuje skreslenie** (bias) a spravidla dosahuje vyššiu presnosť.

**XGBoost** (*eXtreme Gradient Boosting*) je najznámejšia, vysoko optimalizovaná implementácia gradient boostingu. Pridáva regularizáciu, prácu s chýbajúcimi hodnotami a efektívne paralelné budovanie stromov. Spolu s príbuznými (LightGBM, CatBoost) je to **dlhodobo najúspešnejší model na tabuľkové dáta** a takmer štandardný víťaz Kaggle súťaží mimo obrazu a textu.

**Typické použitie:** predikcia na tabuľkových dátach — riziko úveru, predikcia dopytu/predaja, detekcia podvodov, ranking, scoring zákazníkov. Tam, kde máte stĺpce a riadky, začnite XGBoostom.

| ✅ Výhody | ❌ Nevýhody |
|---|---|
| **Špičková presnosť na tabuľkových dátach**, často lepšia než neurónky | Viac **hyperparametrov** na ladenie (počet stromov, hĺbka, learning rate) |
| Robustný, zvláda chýbajúce hodnoty a rôzne škály | Menej vysvetliteľný než jeden strom (ale existuje SHAP) |
| Random forest sa ťažko preučí a beží paralelne | Boosting je **sekvenčný** → pomalšie trénovanie na obrích dátach |
| Netreba veľa dát ani GPU | **Nehodí sa** na obraz/zvuk/text (surové pixely či slová) |

---

## 3. Feed-forward neurónové siete (MLP)

**Feed-forward neurónová sieť** (viacvrstvový perceptrón, *MLP*) je najzákladnejší typ neurónovej siete. Skladá sa z vrstiev neurónov; informácia tečie **jedným smerom** — od vstupu cez skryté vrstvy k výstupu, bez cyklov.

![Feed-forward sieť: vstupná vrstva, skrytá vrstva a výstupná vrstva, prepojené váhami](images/ff-siet-prehlad.svg)

Každý neurón spočíta vážený súčet svojich vstupov, pripočíta **bias** a prevedie výsledok cez nelineárnu **aktivačnú funkciu** (ReLU, sigmoid…):

![Detail jedného neurónu: vstupy vážené váhami w, pripočítaný bias b, výsledok z prejde aktivačnou funkciou σ na výstup a](images/neuron-detail.svg)

Práve nelineárne aktivácie robia zo siete niečo mocnejšie než len lineárnu regresiu — vrstvením dokáže MLP aproximovať prakticky ľubovoľnú funkciu. Ako sa váhy a biasy ladia tréningom (forward pass → loss → backpropagation → update optimalizátorom Adam), podrobne rozoberá [adam-optimalizator.md](adam-optimalizator.md).

**Typické použitie:** univerzálny „lepiaci" model — klasifikácia a regresia na stredne veľkých dátach, koncové vrstvy v zložitejších sieťach (napr. klasifikačná hlava CNN alebo transformera), aproximácia funkcií v simuláciách.

| ✅ Výhody | ❌ Nevýhody |
|---|---|
| **Univerzálny aproximátor** — teoreticky zvládne ľubovoľný vzťah | Ignoruje štruktúru dát (u obrazu nevie, že susedné pixely spolu súvisia) |
| Základný stavebný blok všetkých hlbokých sietí | Veľa parametrov → **potrebuje veľa dát**, ľahko sa preučí |
| Zvláda nelineárne vzťahy, ktoré strom ťažko | Na tabuľkových dátach ho **XGBoost často predbehne** |
| Beží dobre na GPU | Menej vysvetliteľný — „čierna skrinka" |

---

## 4. Konvolučné neurónové siete (CNN)

**Konvolučná sieť (CNN)** je navrhnutá pre dáta s **priestorovou štruktúrou** — predovšetkým obrázky. Kľúčová myšlienka: namiesto toho, aby každý neurón videl všetky pixely (ako v MLP), použije malý **filter (jadro)**, ktorý kĺže po obrázku a hľadá lokálny vzor — hranu, roh, textúru.

![Detail konvolúcie: filter 3×3 kĺže po vstupnej matici 5×5, pre každú pozíciu spočíta vážený súčet a vznikne mapa príznakov](images/konvolucia-detail.svg)

Ten istý filter má **rovnaké váhy pre celý obrázok** (*weight sharing*), takže detektor hrany funguje rovnako v ľavom hornom aj pravom dolnom rohu. To dramaticky znižuje počet parametrov a dáva sieti **invarianciu voči posunu** — mačka je mačka, nech je kdekoľvek v zábere.

Celá sieť potom **strieda konvolúciu a pooling** (zmenšovanie), čím postupne extrahuje čoraz abstraktnejšie príznaky, a na konci pripojí feed-forward vrstvy na samotné rozhodnutie:

![Architektúra CNN: vstupný obrázok prechádza sériou konvolučných a pooling vrstiev, potom sa sploští a prejde plne prepojenými vrstvami do softmax výstupu](images/cnn-architektura.svg)

Hĺbkou siete rastie abstrakcia: prvé vrstvy detegujú hrany a farby, stredné časti objektov (oko, koleso), posledné celé objekty.

**Typické použitie:** **spracovanie obrazu** — klasifikácia a detekcia objektov, segmentácia, rozpoznávanie tvárí, analýza medicínskych snímok, OCR; funguje aj na spektrogramy zvuku a iné mriežkové dáta. Praktická úloha v tomto repozitári: [rozpoznávanie obrázkov](zadania/rozpoznavanie-obrazkov.md).

| ✅ Výhody | ❌ Nevýhody |
|---|---|
| **Špička na obraz** a priestorové dáta | Vyžaduje **veľa dát a výpočtu** (GPU) |
| Weight sharing → menej parametrov, invariancia voči posunu | Málo vysvetliteľná — ťažko sa zisťuje „prečo" |
| Automaticky sa naučí príznaky (netreba ich ručne navrhovať) | Citlivá na adversariálne zmeny (malý šum ju zmätie) |
| Hierarchia hrany → tvary → objekty | Na **tabuľkových dátach zbytočná** — použite XGBoost |

> Pre postupnosti (text, časové rady) CNN nestačí — tam sa dnes používajú **transformery**, ktorým sa venuje [samostatný dokument](transformer-siete.md).

---

## 5. Genetické algoritmy

**Genetický algoritmus (GA)** nie je klasifikátor — je to **optimalizačná metóda** inšpirovaná prirodzeným výberom. Používa sa tam, kde hľadáme dobré riešenie v obrovskom priestore možností a nemáme (alebo nechceme počítať) gradient — napr. návrh rozvrhu, trasy, tvaru súčiastky, alebo ladenie hyperparametrov iného modelu.

![Cyklus genetického algoritmu: počiatočná populácia, hodnotenie fitness, selekcia rodičov, kríženie, mutácia, nová generácia, opakovanie až po ukončovaciu podmienku](images/geneticky-algoritmus.svg)

Riešenie sa zakóduje ako **„chromozóm"** (napr. reťazec bitov alebo čísel). Algoritmus udržiava celú **populáciu** riešení a opakuje evolučný cyklus:

1. **Fitness** — ohodnotí, aké dobré je každé riešenie.
2. **Selekcia** — vyberie najlepšie jedince ako rodičov.
3. **Kríženie (crossover)** — skombinuje gény dvoch rodičov do potomka.
4. **Mutácia** — náhodne zmení malú časť génov (udržuje rozmanitosť, bráni uviaznutiu).
5. Vznikne **nová generácia** a cyklus sa opakuje, kým nie je riešenie dosť dobré alebo neuplynie daný počet generácií.

**Typické použitie:** kombinatorická optimalizácia (rozvrhy, logistika, packing), návrh a ladenie (architektúry sietí, hyperparametre), evolučné umenie, hry a agenti bez explicitného gradientu.

| ✅ Výhody | ❌ Nevýhody |
|---|---|
| Nepotrebuje gradient ani spojitú funkciu | **Výpočtovo drahé** — veľa vyhodnotení fitness |
| Zvláda členité, nespojité priestory s mnohými lokálnymi optimami | Nezaručuje nájdenie globálneho optima |
| Ľahko sa paralelizuje a prispôsobí problému | Citlivé na nastavenie (veľkosť populácie, miera mutácie) |
| Univerzálne — stačí vedieť ohodnotiť riešenie | Pri problémoch, kde *je* gradient, býva pomalšie než gradientné metódy |

---

## Zhrnutie — ktorý model kedy

| Dáta / úloha | Odporúčaný prvý model | Prečo |
|---|---|---|
| **Tabuľkové dáta** (riadky × stĺpce) | **XGBoost** / random forest | najvyššia presnosť, málo dát, netreba GPU |
| Potrebujem **vysvetliteľnosť** | rozhodovací strom (+ SHAP na XGBoost) | čitateľná cesta k rozhodnutiu |
| **Obraz** (klasifikácia, detekcia) | **CNN** | weight sharing, hierarchia príznakov |
| **Text / postupnosti / jazyk** | **transformer** → [transformer-siete.md](transformer-siete.md) | attention, kontext, dnešné LLM |
| Univerzálny nelineárny vzťah, koncová hlava | **feed-forward (MLP)** | jednoduchý, univerzálny aproximátor |
| **Optimalizácia bez gradientu** | **genetický algoritmus** | členité priestory, kombinatorika |

**Najdôležitejšie pravidlo:** typ dát určuje model viac než čokoľvek iné. Na tabuľky nasadzujte stromy/XGBoost, na obraz CNN, na text transformery — a neurónovú sieť neťahajte tam, kde jednoduchší model spraví rovnakú prácu lacnejšie a vysvetliteľnejšie.

---

### Súvisiace dokumenty v repozitári

- [transformer-siete.md](transformer-siete.md) — detailné vysvetlenie transformerov a attention (s obrázkami)
- [adam-optimalizator.md](adam-optimalizator.md) — ako sa neurónové siete trénujú (backpropagation, Adam)
- [embeddings.md](embeddings.md) — ako sa z textu stane vektor (embeddingy, RAG)
- [llm-trendy.md](llm-trendy.md) — aktuálne trendy vo veľkých jazykových modeloch
- [zadania/rozpoznavanie-obrazkov.md](zadania/rozpoznavanie-obrazkov.md), [zadania/RAG_Fine_tunning.md](zadania/RAG_Fine_tunning.md) — praktické úlohy

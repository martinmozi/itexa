# Ako sa trénuje LLM — od surového textu po Instruct model

> **Cieľ dokumentu:** vysvetliť celú tréningovú pipeline veľkého jazykového modelu — čo sa deje od stiahnutia surového internetu až po model s príponou `-Instruct`, ktorý si viete stiahnuť z Hugging Face a ktorý odpovedá na otázky. Po prečítaní budete rozumieť, prečo *base* model „nevie odpovedať", čo presne pridáva inštrukčné ladenie, a kam do tejto pipeline zapadá váš vlastný fine-tuning (LoRA).

Predpokladá znalosť [transformerov](transformer-siete.md) (architektúra, ktorá sa trénuje) a [tréningovej slučky](adam-optimalizator.md) (backprop + Adam — presne tá istá mechanika, len v obrovskom merítku). Tokenizáciu a BPE detailne rozoberá [embeddings.md](embeddings.md).

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
        ▼  FÁZA 3: preferenčné ladenie (RLHF / DPO)            [nad rámec tohto dokumentu]
  CHATOVACÍ MODEL, ktorý reálne používate
```

Kľúčová intuícia: **všetky fázy používajú tú istú sieť a tú istú tréningovú slučku** (forward → loss → backprop → Adam update). Líšia sa len **dátami a loss funkciou** — a práve dáta určujú, čo sa model naučí.

---

## Fáza 0: Dáta — surovina, ktorá rozhoduje o všetkom

Pretraining potrebuje **bilióny (10¹²) tokenov** textu. Typický mix:

| Zdroj | Podiel (orientačne) | Prečo |
|---|---|---|
| web (Common Crawl a pod.) | najväčší | šírka tém a jazykov |
| kód (GitHub) | výrazný | učí štruktúrované myslenie — zlepšuje aj *ne*programovacie schopnosti |
| knihy, články, Wikipedia | menší, ale kvalitný | dlhé súvislé texty, fakty |
| matematika, veda | cielený | reasoning |

Surový web je ale plný spamu, duplikátov a smetí, preto sa robí:

1. **Filtrovanie kvality** — heuristiky aj klasifikátory vyhodia spam, generovaný balast, toxický obsah.
2. **Deduplikácia** — ten istý text miliónkrát by model naučila memorovať, nie generalizovať.
3. **Mixovanie** — pomery zdrojov sú starostlivo ladené; „dáta sú nový hyperparameter".

> **Prečo je to dôležité pochopiť:** kvalita a zloženie dát vysvetľuje väčšinu rozdielov medzi modelmi. Preto je taký veľký rozdiel medzi *open-weight* (dáta tajné) a *plne open-source* modelmi (dáta verejné) — viď [llm-modely.md](llm-modely.md). A preto malé modely horšie zvládajú slovenčinu: v mixe jej je málo.

Text sa nakoniec **tokenizuje** (BPE — detailne v [embeddings.md](embeddings.md)) a nareže na bloky dĺžky kontextového okna.

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

Loss je **cross-entropy**: `L = −ln P(správny token)`. V príklade `L = −ln(0.62) = 0.48`. Keby model dal správnemu tokenu len 0.01, loss je `−ln(0.01) = 4.6` → veľký gradient → veľká korekcia váh. Presne tá istá mechanika ako pri malej sieti v [adam-optimalizator.md](adam-optimalizator.md), len parametrov sú miliardy.

Dve vlastnosti robia z tejto jednoduchej úlohy zázrak:

1. **Self-supervised** — labely netreba vyrábať, sú to ďalšie slová samotného textu. Preto sa dá trénovať na biliónoch tokenov: každá pozícia v každom texte je jeden tréningový príklad.
2. **Predpovedať ďalší token dobre = rozumieť** — aby model vedel dokončiť „Násobenie 23 × 17 = ", musí vedieť násobiť. Aby dokončil detektívku vetou „Vrahom je …", musí sledovať dej. Kompresia textu si vynúti model sveta.

### Škála a scaling laws

Empiricky platí: loss klesá predvídateľne s **veľkosťou modelu**, **množstvom dát** a **výpočtom** (tzv. *scaling laws*). Odtiaľ preteky v čipoch a dátach. Rádovo: špičkový pretraining = tisíce GPU, týždne až mesiace, desiatky miliónov dolárov. Preto pretraining robí pár firiem a všetci ostatní **stavajú na hotových base/Instruct modeloch**.

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

---

## Fáza 2: SFT / Instruction tuning — z dokončovača asistent

**SFT** (*Supervised Fine-Tuning*), tiež *instruction tuning*, doučí base model na dátach v tvare **inštrukcia → odpoveď**. Výsledok sú modely s príponou `-Instruct` / `-it` / `-chat` na Hugging Face.

### Dáta

Desaťtisíce až milióny ukážkových dialógov. Kde sa berú:

- **ručne písané** ľuďmi (drahé, kvalitné) — otázky, úlohy, ideálne odpovede,
- **syntetické** — generované silnejším modelom a filtrované (dnes prevažujúce; tzv. distillation, viď [llm-trendy.md](llm-trendy.md)),
- reálne konverzácie s asistentom (so súhlasom, filtrované).

Oproti biliónom tokenov pretrainingu je to **maličký dataset** — SFT nemá modelu dodať nové znalosti, len **zmeniť správanie**: „keď vidíš otázku, odpovedz na ňu; odpovedaj v tomto tóne; odmietni škodlivé požiadavky".

### Chat šablóna a špeciálne tokeny

Dialóg sa serializuje do jedného textu pomocou **chat šablóny** so špeciálnymi tokenmi, ktoré oddeľujú role (formát sa líši podľa modelu — preto pri fine-tuningu vždy `tokenizer.apply_chat_template`):

```text
<|system|>Si užitočný asistent.<|end|>
<|user|>Koľko nôh má pavúk?<|end|>
<|assistant|>Pavúk má osem nôh.<|end|>
```

Model sa počas SFT naučí význam týchto tokenov: po `<|assistant|>` nasleduje moja odpoveď, `<|end|>` znamená „dohovoril som". Aj „ukončenie odpovede" je teda naučené správanie — base model by pokračoval donekonečna.

### Tréning: loss len na odpovedi

Mechanika je identická s pretrainingom (predikcia ďalšieho tokenu, cross-entropy, Adam) — s jedným kľúčovým rozdielom: **loss sa počíta len na tokenoch odpovede asistenta** (tokeny promptu sa maskujú):

```text
tokeny:   <|user|> Koľko nôh má pavúk ? <|end|> <|assistant|> Pavúk má osem nôh . <|end|>
loss:        ✗      ✗    ✗   ✗    ✗   ✗    ✗          ✗         ✓    ✓    ✓   ✓  ✓    ✓
```

Prečo: model sa má naučiť **generovať odpovede**, nie generovať otázky používateľa. Keby sme loss počítali všade, učili by sme ho imitovať aj používateľov.

### Výsledok: Instruct model

Po SFT model:

- interpretuje prompt ako úlohu a plní ju,
- drží formát dialógu (role, ukončovanie),
- má natrénovaný štýl a základné odmietanie škodlivých požiadaviek,
- **znalosti má stále z pretrainingu** — SFT ich len sprístupnil formou dialógu.

> **Dôležitý dôsledok pre prax:** fine-tuning je dobrý na **štýl a správanie**, zlý na **vkladanie nových faktov** — fakty „sedia" vo váhach z pretrainingu a malý SFT dataset ich spoľahlivo neprepíše. Na nové/aktuálne fakty použite RAG. (Detailne v [llm-trendy.md](llm-trendy.md) a v [zadaní](zadania/RAG_Fine_tunning.md).)

### Váš vlastný fine-tuning = tá istá Fáza 2 v malom

Keď v [zadaní](zadania/RAG_Fine_tunning.md) robíte **LoRA/QLoRA** fine-tuning, robíte presne SFT — dvojice otázka → odpoveď, chat šablóna, loss na odpovedi. Rozdiel je len v úspornosti: namiesto všetkých miliárd váh trénujete malé **adaptérové matice** (LoRA) pripojené k zamrznutému modelu, takže to zvládne jedno GPU.

---

## Fáza 3 (výhľad): preferenčné ladenie — RLHF / DPO

Za hranicou tohto dokumentu, ale pre úplnosť: po SFT nasleduje ladenie podľa **ľudských preferencií** — ľudia (alebo model) porovnávajú dvojice odpovedí („táto je lepšia") a model sa optimalizuje smerom k preferovaným odpovediam (**RLHF** cez reward model a posilňované učenie, alebo jednoduchšie **DPO** priamo z porovnávacích párov). Toto dolaďuje užitočnosť, neškodnosť a štýl. Moderné *reasoning* modely pridávajú ešte RL na overiteľných úlohách (matematika, kód), kde sa odmeňuje správny výsledok.

Zhrnutie celej cesty jednou vetou: **pretraining dá modelu schopnosti, SFT z neho urobí asistenta, preferenčné ladenie ho vycibrí.**

---

## Kontrolné otázky

1. Prečo sa pretraining dá robiť na biliónoch tokenov, hoci nikto tie dáta „nelabeloval"?
2. Base model na prompt „Preloz do angličtiny: pes" odpovie „Preloz do angličtiny: mačka. Preloz do angličtiny: dom." — vysvetlite, prečo je to z pohľadu jeho tréningu *správne* správanie.
3. Čím sa líši SFT od pretrainingu (a) v dátach, (b) v loss funkcii, (c) v cieli? Čo majú mechanicky spoločné?
4. Prečo sa pri SFT maskuje loss na tokenoch používateľa?
5. Prečo je fine-tuning nevhodný na naučenie modelu nových faktov a čo použiť namiesto neho?
6. Kolega stiahol z Hugging Face `Llama-3.1-8B` (bez prípony) do chatbota a sťažuje sa, že „model odpovedá nezmysly". Čo mu poradíte?

---

### Súvisiace dokumenty

- [prehlad-predmetu.md](prehlad-predmetu.md) — prehľad celého predmetu (8 lekcií)
- [transformer-siete.md](transformer-siete.md) — architektúra, ktorá sa tu trénuje
- [adam-optimalizator.md](adam-optimalizator.md) — tréningová slučka a optimalizátor (rovnaké aj pre LLM)
- [embeddings.md](embeddings.md) — tokenizácia (BPE), z ktorej pretraining vychádza
- [llm-modely.md](llm-modely.md) — prehľad modelov (base vs Instruct nájdete v názvoch na HF)
- [llm-trendy.md](llm-trendy.md) — distillation, kedy fine-tuning áno/nie
- [zadania/RAG_Fine_tunning.md](zadania/RAG_Fine_tunning.md) — vlastný SFT cez LoRA/QLoRA

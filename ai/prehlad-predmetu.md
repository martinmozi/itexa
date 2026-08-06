# Umelá inteligencia — prehľad predmetu (8 lekcií)

> **Cieľ predmetu:** porozumieť, ako moderná AI funguje *zvnútra* — od klasického strojového učenia cez neurónové siete a transformery až po prácu s dnešnými LLM (RAG, fine-tuning, agenti). Dôraz je na **porozumení, nie memorovaní**: každý kľúčový mechanizmus si prepočítame na malom príklade alebo naprogramujeme vlastnými rukami, až potom siahneme po hotovom frameworku.

**Ako predmet funguje:** teória (dokumenty nižšie) + dve väčšie praktické zadania, ktoré sa tiahnu cez viac lekcií. Každý dokument končí kontrolnými otázkami — ak na ne viete odpovedať vlastnými slovami, lekciu ste pochopili.

**Prerekvizity:** znalosť Pythonu na úrovni „viem napísať a odladiť skript" a stredoškolská matematika (funkcie, derivácia, vektory a matice). Skúsenosti s ML sa nepredpokladajú. Pred lekciou 3 si treba pripraviť počítač podľa [01-vyvojove-prostredie.md](tutorials/00-prostredie/01-vyvojove-prostredie.md).

---

## Mapa predmetu

```text
  ČASŤ I — ZÁKLADY                          ČASŤ II — MODERNÉ LLM
  ────────────────                          ─────────────────────
  0. Vývojové prostredie (príručka)
  1. Princípy AI a ML                       5. Ako sa trénuje LLM + krajina modelov
  2. Klasické modely (stromy, XGBoost)      6. Embeddingy a RAG          ── zadanie 2A
  3. Feed-forward siete a učenie ─┐         7. Fine-tuning (LoRA), RAG vs. FT ── zadanie 2B
  4. Transformery a attention     │         8. Agenti, nástroje, Claude Code
                                  └─ zadanie 1
```

Červená niť: **typ dát a úlohy určuje model** (lekcie 1–4) a **dáta + loss určujú, čo sa model naučí** (lekcie 5–7). Kto pochopí tieto dva princípy, vie sa zorientovať v čomkoľvek novom, čo v AI vyjde.

Každý dokument má na začiatku riadok **Poradie čítania** s odkazom na predchádzajúci a nasledujúci — dá sa nimi prejsť celý predmet bez vracania sa sem.

---

## Lekcia 0 — Príprava prostredia

**Materiál:** [01-vyvojove-prostredie.md](tutorials/00-prostredie/01-vyvojove-prostredie.md)

Python a virtuálne prostredia, VS Code, PyTorch, GPU (CUDA na NVIDIA, MPS na Macu), knižnice kurzu, lokálna inferencia (vLLM, Ollama), odporúčaný hardvér a kedy si prenajať GPU v cloude.

Nie je to samostatná prednáška — je to príručka, ktorú treba prejsť **pred lekciou 3**, keď začína zadanie 1. Vracať sa k nej budete pri zadaní 2 (VRAM, kvantizácia, Colab).

---

## Lekcia 1 — Princípy umelej inteligencie

**Materiál:** [tutorials/01-prehlad](tutorials/01-prehlad/README.md) — [co-je-ai](tutorials/01-prehlad/01-co-je-ai.md) · [rezimy-strojoveho-ucenia](tutorials/01-prehlad/02-rezimy-strojoveho-ucenia.md) · [generalizacia-a-preucenie](tutorials/01-prehlad/03-generalizacia-a-preucenie.md) · [metriky](tutorials/01-prehlad/04-metriky.md)

Čo je AI a čo nie je; symbolická AI vs. strojové učenie; taxonómia (AI ⊃ ML ⊃ neurónové siete ⊃ deep learning). Tri režimy učenia: s učiteľom, bez učiteľa, posilňované. Spoločná schéma učenia: predikcia → porovnanie s pravdou (loss) → úprava parametrov. Zovšeobecnenie a preučenie, delenie dát na trénovaciu/validačnú/testovaciu množinu, regularizácia a metriky kvality.

**Po lekcii viete:**
- vysvetliť rozdiel medzi „pravidlá píše človek" a „vzory sa učí z dát" a kedy má ktorý prístup zmysel,
- zaradiť ľubovoľnú úlohu do správneho režimu učenia,
- rozlíšiť tabuľkové vs. neštruktúrované dáta — najdôležitejšia intuícia pri výbere modelu,
- vysvetliť, načo je validačná množina a prečo presnosť sama osebe klame.

---

## Lekcia 2 — Klasické modely: stromy, Random Forest, XGBoost

**Materiál:** [tutorials/02-typy-modelov](tutorials/02-typy-modelov/README.md) — [rozhodovacie-stromy](tutorials/02-typy-modelov/01-rozhodovacie-stromy.md) · [random-forest-a-xgboost](tutorials/02-typy-modelov/02-random-forest-a-xgboost.md) · [xgboost-priklad-iso8583](tutorials/02-typy-modelov/03-xgboost-priklad-iso8583.md)

Rozhodovacie stromy a ako sa učia (Gini/entropia); prečo jeden strom preučí a ako to riešia ansámble — Random Forest (bagging, paralelne, znižuje rozptyl) vs. XGBoost (boosting, sekvenčne opravuje chyby, znižuje skreslenie). **Kľúčové posolstvo: na tabuľkové dáta je XGBoost dodnes prvá voľba — nie neurónová sieť.**

**Po lekcii viete:**
- prečítať a obhájiť rozhodnutie stromu; vysvetliť overfitting na jednom strome,
- vysvetliť rozdiel bagging vs. boosting vlastnými slovami,
- pre danú úlohu (úverové riziko, detekcia podvodov…) vybrať vhodný klasický model.

---

## Lekcia 3 — Feed-forward siete a ich učenie

**Materiál:** [feed-forward-siete](tutorials/02-typy-modelov/04-feed-forward-siete.md) · [konvolucne-siete](tutorials/02-typy-modelov/05-konvolucne-siete.md) · [ktory-model-kedy](tutorials/02-typy-modelov/06-ktory-model-kedy.md) + [tutorials/03-ucenie](tutorials/03-ucenie/README.md) ([adam-optimalizator](tutorials/03-ucenie/01-adam-optimalizator.md), [problemy-pri-uceni](tutorials/03-ucenie/02-problemy-pri-uceni.md)) → **[Zadanie 1: rozpoznávanie obrázkov](zadania/rozpoznavanie-obrazkov.md)**

Neurón (vážený súčet + bias + aktivácia), viacvrstvový perceptrón, prečo nelinearita robí sieť univerzálnym aproximátorom. Tréningová slučka: forward → loss → backpropagation → update. Optimalizátor Adam do detailu (momentum, adaptívny krok, bias correction) — tak, aby ste ho vedeli naprogramovať; lokálne minimá vs. sedlové body. **Čo sa pri učení pokazí a ako to riešiť** ([02-problemy-pri-uceni.md](tutorials/03-ucenie/02-problemy-pri-uceni.md)): miznúce a explodujúce gradienty, mŕtve ReLU neuróny, inicializácia, `NaN`, chyby v dátach, mixed precision a hardvérové poruchy. **Poznámky o iných typoch sietí:** CNN pre obraz (konvolúcia, weight sharing, hierarchia príznakov).

**Po lekcii viete:**
- ručne prepočítať výstup neurónu a jeden Adam update,
- vysvetliť, čo počíta backpropagation a prečo sieť potrebuje nelineárne aktivácie,
- povedať, prečo na obraz CNN a nie MLP (a prečo na tabuľky ani jedno),
- vysvetliť, čo v hlbokej sieti reálne brzdí tréning — a prečo to nie sú lokálne minimá,
- podľa tvaru krivky lossu a noriem gradientov určiť, čo je s tréningom zle, a vedieť to opraviť.

**Zadanie 1** (cez lekcie 3–4): vlastná feed-forward sieť v NumPy vrátane backpropu a Adama, potom to isté v PyTorch, porovnanie. Klasifikácia obrázkov + rozpoznanie vlastného nakresleného vstupu.

---

## Lekcia 4 — Transformery a attention

**Materiál:** [01-transformer-siete.md](tutorials/04-llm/01-transformer-siete.md)

Prečo RNN nestačili (sekvenčnosť, krátka pamäť) a čo priniesol „Attention Is All You Need". Self-attention krok po kroku: Query/Key/Value, skóre, softmax, vážený súčet — každý token sa „pozrie" na všetky ostatné naraz. Multi-head, positional encoding, maskovaná attention. Encoder / decoder / decoder-only, autoregresívne generovanie a **dekódovanie** (greedy, teplota, top-p).

**Po lekcii viete:**
- vysvetliť roly Q, K, V analógiou s vyhľadávaním a opísať postup výpočtu attention,
- povedať, prečo je attention kvadratická v dĺžke vstupu a čo z toho plynie pre dlhý kontext,
- opísať, ako z „predpovedz ďalší token" vzniká generovanie celých odpovedí,
- nastaviť dekódovanie podľa toho, či chcete faktickú alebo kreatívnu odpoveď.

> Ručne prepočítaný príklad tej istej attention (s číslami) je v [embeddings.md, Krok 3](tutorials/04-llm/04-embeddings.md#krok-3-transformer-vrstvy--tu-sa-deje-pochopenie-kontextu). Teraz je nepovinný, v lekcii 6 sa k nemu vrátime.

---

## Lekcia 5 — Ako sa trénuje LLM a krajina dnešných modelov

**Materiál:** [02-llm-trening.md](tutorials/04-llm/02-llm-trening.md) → [03-llm-modely.md](tutorials/04-llm/03-llm-modely.md)

Celá tréningová pipeline: dáta (filtrovanie, deduplikácia, mix) → **pretraining** (predikcia ďalšieho tokenu, self-supervised, scaling laws) → **base model** (dokončovač textu) → **SFT / instruction tuning** (chat šablóna, loss len na odpovedi) → **Instruct model**; výhľad na RLHF/DPO. Potom prehľad trhu: **proprietárne vs. open-weight vs. plne open-source** a výber modelu podľa úlohy. Na záver právne a etické mantinely (GDPR, licencie, AI Act, bias).

**Po lekcii viete:**
- vysvetliť, prečo base model „neposlúcha" a čo presne opraví instruction tuning,
- rozlíšiť tri stupne otvorenosti modelov a ich praktické dôsledky (audit, licencie, nasadenie),
- pre konkrétnu firemnú úlohu vybrať kategóriu aj konkrétny model a rozhodnutie obhájiť,
- pomenovať právne povinnosti, ktoré s voľbou modelu a dát prichádzajú.

---

## Lekcia 6 — Embeddingy a RAG

**Materiál:** [04-embeddings.md](tutorials/04-llm/04-embeddings.md) → [05-rag.md](tutorials/04-llm/05-rag.md) → **[Zadanie 2, úloha A: RAG](zadania/RAG_Fine_tunning.md)**

Cesta textu na vektor: tokenizácia (BPE) → embedding matica → transformer vrstvy → pooling → normalizácia — celé prepočítané ručne na malom príklade. Podobnosť (cosine, dot product), prečo sú modely vzájomne nekompatibilné (kontrastívne učenie). RAG pipeline: chunking, indexovanie (FAISS, flat vs. ANN), retrieval, reranking (bi-encoder vs. cross-encoder), výpočtové nároky. Na záver pokročilý retrieval: hybrid search, prepis dotazu, agentický RAG.

**Po lekcii viete:**
- prepočítať attention aj cosine similarity na papieri a vysvetliť, prečo sa vektory normalizujú,
- navrhnúť chunking stratégiu pre konkrétny typ dokumentov a obhájiť veľkosť chunku,
- vysvetliť, prečo sa reranker púšťa len na top-k a nie na celú databázu,
- postaviť kompletný RAG od dokumentu po odpoveď (= zadanie 2A) a vedieť, čím ho vylepšiť, keď nestačí.

---

## Lekcia 7 — Fine-tuning a rozhodovanie RAG vs. fine-tuning

**Materiál:** [06-fine-tuning-lora.md](tutorials/04-llm/06-fine-tuning-lora.md) → **[Zadanie 2, úloha B: fine-tuning](zadania/RAG_Fine_tunning.md)**

Prečo sa celý model dotrénovať nedá (pamäťová matematika). **LoRA** — rozklad `ΔW = A·B`, čo je rank `r`, prečo malé adaptéry stačia; **QLoRA** ako 4-bitová nadstavba. Kedy sa fine-tuning oplatí (štýl, formát, distillation, edge) a kedy nie (nové fakty → RAG; často sa meniace dáta; potreba citovať zdroj). Halucinácie a ako ich meria testovacia sada s „chytákmi".

**Po lekcii viete:**
- vysvetliť, čo je LoRA a spočítať, koľko parametrov sa pri danom `r` reálne trénuje,
- rozhodnúť RAG vs. fine-tuning vs. dlhý kontext pre konkrétny prípad a rozhodnutie obhájiť,
- navrhnúť vyhodnotenie (baseline surového modelu, testovacie otázky, chytáky).

**Zadanie 2** (cez lekcie 6–7): malý open model z HF + dlhý neznámy dokument; sprístupniť jeho obsah cez RAG **alebo** LoRA fine-tuning, zmerať proti baseline, porovnať prístupy.

---

## Lekcia 8 — Agenti, nástroje a Claude Code

**Materiál:** [01-agenti-a-nastroje.md](tutorials/05-prakticke/01-agenti-a-nastroje.md) + [02-llm-trendy.md](tutorials/05-prakticke/02-llm-trendy.md) (záver) + živé demá na hodine

Čo robí z LLM **agenta**: slučka model → nástroj → výsledok → model (ReAct), ukázaná na dvadsiatich riadkoch kódu. Tool use / function calling, MCP ako štandard pripájania nástrojov. **Claude Code** ako ukážka hotového agenta: práca s repozitárom, spúšťanie príkazov, kedy mu (ne)veriť. **LangChain / LangGraph** — a kedy framework *ne*použiť. Bezpečnosť agentov: prompt injection, least-privilege, sandboxing. Context engineering a evaluácia agentov. Na záver výhľad, čo sledovať po kurze.

**Po lekcii viete:**
- vysvetliť agentovú slučku a rozdiel medzi „chatbot" a „agent",
- napísať jednoduchý agent s jedným-dvomi nástrojmi (bez frameworku aj v LangChaine),
- vymenovať hlavné riziká (prompt injection) a základné obrany,
- efektívne používať Claude Code pri vlastnej práci.

---

## Zhrnutie: dva princípy, ktoré sa oplatí odniesť

1. **Typ dát a úlohy určuje model.** Tabuľky → XGBoost. Obraz → CNN. Text/sekvencie → transformer. Neťahajte LLM tam, kde jednoduchší model spraví lacnejšiu a vysvetliteľnejšiu prácu.
2. **Dáta + loss určujú, čo sa model naučí.** Rovnaká sieť a rovnaká slučka (forward → loss → backprop → Adam) dá dokončovač textu, asistenta aj embedding model — podľa toho, aké dáta a akú loss jej dáte. Kto rozumie tejto mechanike, rozumie celému modernému AI stacku.

---

## Hodnotenie predmetu

| Zložka | Váha | Kedy |
|---|---|---|
| **Zadanie 1** — vlastná sieť + Adam + PyTorch | 40 % | odovzdanie po lekcii 4 |
| **Zadanie 2** — RAG alebo LoRA fine-tuning | 40 % | odovzdanie po lekcii 7 |
| **Záverečná ústna rozprava** nad kontrolnými otázkami | 20 % | skúškové obdobie |

Bodovanie vnútri každého zadania je v jeho dokumente. Na absolvovanie treba odovzdať **obe** zadania a získať aspoň 50 % celkovo. Za bonusové časti zadaní sa dá získať navyše až +10 %, resp. +15 %.

**Kontrolné otázky na konci každého dokumentu sú zároveň okruhmi na skúšku** — nie sú to cvičenia navyše.

---

## Odporúčaná literatúra

Nič z toho nie je povinné; dokumenty predmetu sú sebestačné.

- **Pôvodné články** (všetky voľne na arXiv, každý sa dá prečítať za hodinu): *Attention Is All You Need* (2017) — transformer; *LoRA: Low-Rank Adaptation of Large Language Models* (2021); *Retrieval-Augmented Generation…* (2020); *Adam: A Method for Stochastic Optimization* (2014); *[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)* (2024) — okrem architektúry obsahuje aj vzácne čísla o tom, ako často zlyháva hardvér pri tréningu na 16 000 GPU (sekcia 3.3.4).
- **Knihy:** Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (praktický úvod do časti I); Goodfellow, Bengio, Courville — *Deep Learning* (teoretická referencia, voľne online).
- **Dokumentácia:** [PyTorch](https://pytorch.org/docs), [Hugging Face](https://huggingface.co/docs) — pri zadaniach ju budete otvárať častejšie než čokoľvek iné.

---

### Všetky dokumenty predmetu (v poradí, v akom sa preberajú)

Materiály sú v adresári [`tutorials/`](tutorials/README.md), rozdelené do šiestich tematických oblastí.

| # | Dokument | Obsah | Lekcia |
|---|---|---|---|
| 0 | [00-prostredie/01-vyvojove-prostredie.md](tutorials/00-prostredie/01-vyvojove-prostredie.md) | inštalácia (PyTorch, CUDA/MPS), vLLM, hardvér, cloud | príprava |
| 1 | [01-prehlad/01-co-je-ai.md](tutorials/01-prehlad/01-co-je-ai.md) | symbolická AI vs. ML, taxonómia AI ⊃ ML ⊃ NN ⊃ DL | 1 |
| 2 | [01-prehlad/02-rezimy-strojoveho-ucenia.md](tutorials/01-prehlad/02-rezimy-strojoveho-ucenia.md) | s učiteľom / bez učiteľa / posilňované, typy dát | 1 |
| 3 | [01-prehlad/03-generalizacia-a-preucenie.md](tutorials/01-prehlad/03-generalizacia-a-preucenie.md) | train/val/test, overfitting, bias vs. variance, regularizácia | 1 |
| 4 | [01-prehlad/04-metriky.md](tutorials/01-prehlad/04-metriky.md) | matica zámen, precision, recall, F1, MAE/RMSE | 1 |
| 5 | [02-typy-modelov/01-rozhodovacie-stromy.md](tutorials/02-typy-modelov/01-rozhodovacie-stromy.md) | Gini/entropia, preučenie a nestabilita stromu | 2 |
| 6 | [02-typy-modelov/02-random-forest-a-xgboost.md](tutorials/02-typy-modelov/02-random-forest-a-xgboost.md) | bagging vs. boosting, prečo stromy vyhrávajú na tabuľkách | 2 |
| 7 | [02-typy-modelov/03-xgboost-priklad-iso8583.md](tutorials/02-typy-modelov/03-xgboost-priklad-iso8583.md) | rezíduá krok za krokom na kartových transakciách | 2 |
| 8 | [02-typy-modelov/04-feed-forward-siete.md](tutorials/02-typy-modelov/04-feed-forward-siete.md) | neurón, MLP, nelineárne aktivácie | 3 |
| 9 | [02-typy-modelov/05-konvolucne-siete.md](tutorials/02-typy-modelov/05-konvolucne-siete.md) | konvolúcia, weight sharing, pooling | 3 |
| 10 | [02-typy-modelov/06-ktory-model-kedy.md](tutorials/02-typy-modelov/06-ktory-model-kedy.md) | rozhodovacia tabuľka dáta → model | 1–3 |
| 11 | [03-ucenie/01-adam-optimalizator.md](tutorials/03-ucenie/01-adam-optimalizator.md) | tréningová slučka, backprop, Adam do detailu | 3 |
| 12 | [03-ucenie/02-problemy-pri-uceni.md](tutorials/03-ucenie/02-problemy-pri-uceni.md) | miznúce/explodujúce gradienty, `NaN`, dáta, fp16, hardvér | 3 |
| — | [zadania/rozpoznavanie-obrazkov.md](zadania/rozpoznavanie-obrazkov.md) | **zadanie 1** — vlastná sieť + Adam + PyTorch | 3–4 |
| 13 | [04-llm/01-transformer-siete.md](tutorials/04-llm/01-transformer-siete.md) | attention, multi-head, positional encoding, dekódovanie | 4 |
| 14 | [04-llm/02-llm-trening.md](tutorials/04-llm/02-llm-trening.md) | pretraining → base → SFT → Instruct | 5 |
| 15 | [04-llm/03-llm-modely.md](tutorials/04-llm/03-llm-modely.md) | proprietárne / open-weight / open-source, právo a etika | 5 |
| 16 | [04-llm/04-embeddings.md](tutorials/04-llm/04-embeddings.md) | tokenizácia, embeddingy, pooling, normalizácia | 6 |
| 17 | [04-llm/05-rag.md](tutorials/04-llm/05-rag.md) | chunking, index, retrieval, reranking, pokročilý RAG | 6 |
| — | [zadania/RAG_Fine_tunning.md](zadania/RAG_Fine_tunning.md) | **zadanie 2** — RAG alebo LoRA fine-tuning | 6–7 |
| 18 | [04-llm/06-fine-tuning-lora.md](tutorials/04-llm/06-fine-tuning-lora.md) | LoRA/QLoRA, RAG vs. fine-tuning, halucinácie | 7 |
| 19 | [05-prakticke/01-agenti-a-nastroje.md](tutorials/05-prakticke/01-agenti-a-nastroje.md) | agentová slučka, tool use, MCP, Claude Code, bezpečnosť | 8 |
| 20 | [05-prakticke/02-llm-trendy.md](tutorials/05-prakticke/02-llm-trendy.md) | trendy a čo sledovať po kurze | 8 (záver) |

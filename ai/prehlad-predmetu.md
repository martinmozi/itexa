# Umelá inteligencia — prehľad predmetu (8 lekcií)

> **Cieľ predmetu:** porozumieť, ako moderná AI funguje *zvnútra* — od klasického strojového učenia cez neurónové siete a transformery až po prácu s dnešnými LLM (RAG, fine-tuning, agenti). Dôraz je na **porozumení, nie memorovaní**: každý kľúčový mechanizmus si prepočítame na malom príklade alebo naprogramujeme vlastnými rukami, až potom siahneme po hotovom frameworku.

**Ako predmet funguje:** teória (dokumenty nižšie) + dve väčšie praktické zadania, ktoré sa tiahnu cez viac lekcií. Každý dokument končí kontrolnými otázkami — ak na ne viete odpovedať vlastnými slovami, lekciu ste pochopili.

---

## Mapa predmetu

```text
  ČASŤ I — ZÁKLADY                          ČASŤ II — MODERNÉ LLM
  ────────────────                          ─────────────────────
  1. Princípy AI a ML                       5. Ako sa trénuje LLM + krajina modelov
  2. Klasické modely (stromy, XGBoost)      6. Embeddingy a RAG          ── zadanie 2A
  3. Feed-forward siete a učenie ─┐         7. Fine-tuning (LoRA), RAG vs FT ── zadanie 2B
  4. Transformery a attention     │         8. Claude Code a agenti (LangChain)
                                  └─ zadanie 1
```

Červená niť: **typ dát a úlohy určuje model** (lekcie 1–4) a **dáta + loss určujú, čo sa model naučí** (lekcie 5–7). Kto pochopí tieto dva princípy, vie sa zorientovať v čomkoľvek novom, čo v AI vyjde.

---

## Lekcia 1 — Princípy umelej inteligencie

**Materiál:** [umela-inteligencia-prehlad.md](umela-inteligencia-prehlad.md) (úvod až po „tri režimy učenia")

Čo je AI a čo nie je; symbolická AI vs. strojové učenie; taxonómia (AI ⊃ ML ⊃ neurónové siete ⊃ deep learning). Tri režimy učenia: s učiteľom, bez učiteľa, posilňované. Spoločná schéma učenia: predikcia → porovnanie s pravdou (loss) → úprava parametrov.

**Po lekcii viete:**
- vysvetliť rozdiel medzi „pravidlá píše človek" a „vzory sa učí z dát" a kedy má ktorý prístup zmysel,
- zaradiť ľubovoľnú úlohu do správneho režimu učenia,
- rozlíšiť tabuľkové vs. neštruktúrované dáta — najdôležitejšia intuícia pri výbere modelu.

---

## Lekcia 2 — Klasické modely: stromy, Random Forest, XGBoost

**Materiál:** [umela-inteligencia-prehlad.md](umela-inteligencia-prehlad.md) (sekcie 1, 2, 5 a záverečná tabuľka „ktorý model kedy")

Rozhodovacie stromy a ako sa učia (Gini/entropia); prečo jeden strom preučí a ako to riešia ansámble — Random Forest (bagging, paralelne, znižuje rozptyl) vs. XGBoost (boosting, sekvenčne opravuje chyby, znižuje skreslenie). **Kľúčové posolstvo: na tabuľkové dáta je XGBoost dodnes prvá voľba — nie neurónová sieť.**

**Po lekcii viete:**
- prečítať a obhájiť rozhodnutie stromu; vysvetliť overfitting na jednom strome,
- vysvetliť rozdiel bagging vs. boosting vlastnými slovami,
- pre danú úlohu (úverové riziko, detekcia podvodov, rozvrh…) vybrať vhodný klasický model.

---

## Lekcia 3 — Feed-forward siete a ich učenie

**Materiál:** [umela-inteligencia-prehlad.md](umela-inteligencia-prehlad.md) (sekcie 3, 4) + [adam-optimalizator.md](adam-optimalizator.md) → **[Zadanie 1: rozpoznávanie obrázkov](zadania/rozpoznavanie-obrazkov.md)**

Neurón (vážený súčet + bias + aktivácia), viacvrstvový perceptrón, prečo nelinearita robí sieť univerzálnym aproximátorom. Tréningová slučka: forward → loss → backpropagation → update. Optimalizátor Adam do detailu (momentum, adaptívny krok, bias correction) — tak, aby ste ho vedeli naprogramovať. **Poznámky o iných typoch sietí:** CNN pre obraz (konvolúcia, weight sharing, hierarchia príznakov) a prehľad, na čo sa ktorá architektúra hodí.

**Po lekcii viete:**
- ručne prepočítať výstup neurónu a jeden Adam update,
- vysvetliť, čo počíta backpropagation a prečo sieť potrebuje nelineárne aktivácie,
- povedať, prečo na obraz CNN a nie MLP (a prečo na tabuľky ani jedno).

**Zadanie 1** (cez lekcie 3–4): vlastná feed-forward sieť v NumPy vrátane backpropu a Adama, potom to isté v PyTorch, porovnanie. Klasifikácia obrázkov + rozpoznanie vlastného nakresleného vstupu.

---

## Lekcia 4 — Transformery a attention

**Materiál:** [transformer-siete.md](transformer-siete.md) (+ mechanika s číslami v [embeddings.md](embeddings.md), Časť 1)

Prečo RNN nestačili (sekvenčnosť, krátka pamäť) a čo priniesol „Attention Is All You Need". Self-attention krok po kroku: Query/Key/Value, skóre, softmax, vážený súčet — každý token sa „pozrie" na všetky ostatné naraz. Multi-head, positional encoding, maskovaná attention. Encoder / decoder / decoder-only a autoregresívne generovanie textu token po tokene.

**Po lekcii viete:**
- prepočítať self-attention pre tri tokeny na papieri (Q·K → softmax → vážený súčet V),
- vysvetliť roly Q, K, V analógiou s vyhľadávaním,
- povedať, prečo je attention kvadratická v dĺžke vstupu a čo z toho plynie pre dlhý kontext,
- opísať, ako z „predpovedz ďalší token" vzniká generovanie celých odpovedí.

---

## Lekcia 5 — Ako sa trénuje LLM a krajina dnešných modelov

**Materiál:** [llm-trening.md](llm-trening.md) + [llm-modely.md](llm-modely.md) + [llm-trendy.md](llm-trendy.md) (sekcia trendov)

Celá tréningová pipeline: dáta (filtrovanie, deduplikácia, mix) → **pretraining** (predikcia ďalšieho tokenu, self-supervised, scaling laws) → **base model** (dokončovač textu) → **SFT / instruction tuning** (chat šablóna, loss len na odpovedi) → **Instruct model**; výhľad na RLHF/DPO. Potom prehľad trhu: **proprietárne vs. open-weight vs. plne open-source** (s tréningovými dátami) a výber modelu podľa úlohy (OCR, kódovanie, tabuľky, embeddingy, lokálny beh…).

**Po lekcii viete:**
- vysvetliť, prečo base model „neposlúcha" a čo presne opraví instruction tuning,
- rozlíšiť tri stupne otvorenosti modelov a ich praktické dôsledky (audit, licencie, nasadenie),
- pre konkrétnu firemnú úlohu vybrať kategóriu aj konkrétny model a rozhodnutie obhájiť.

---

## Lekcia 6 — Embeddingy a RAG

**Materiál:** [embeddings.md](embeddings.md) → **[Zadanie 2, úloha A: RAG](zadania/RAG_Fine_tunning.md)**

Cesta textu na vektor: tokenizácia (BPE) → embedding matica → transformer vrstvy → pooling → normalizácia — celé prepočítané ručne na malom príklade. Podobnosť (cosine, dot product), prečo sú modely vzájomne nekompatibilné (kontrastívne učenie). RAG pipeline: chunking (veľkosť, overlap, parent-child), indexovanie (FAISS, flat vs. ANN), retrieval, reranking (bi-encoder vs. cross-encoder), výpočtové nároky.

**Po lekcii viete:**
- prepočítať cosine similarity a vysvetliť, prečo sa vektory normalizujú,
- navrhnúť chunking stratégiu pre konkrétny typ dokumentov a obhájiť veľkosť chunku,
- vysvetliť, prečo sa reranker púšťa len na top-k a nie na celú databázu,
- postaviť kompletný RAG od dokumentu po odpoveď (= zadanie 2A).

---

## Lekcia 7 — Fine-tuning a rozhodovanie RAG vs. fine-tuning

**Materiál:** [llm-trendy.md](llm-trendy.md) (sekcie 3–6) + [llm-trening.md](llm-trening.md) (SFT) → **[Zadanie 2, úloha B: fine-tuning](zadania/RAG_Fine_tunning.md)**

Fine-tuning ako „Fáza 2 v malom": LoRA/QLoRA — prečo stačia malé adaptéry namiesto celého modelu. Kedy sa fine-tuning oplatí (štýl, formát, distillation, edge) a kedy nie (nové fakty → RAG; často sa meniace dáta). Moderný retrieval: hybrid search, query transformation, small-to-big, agentic RAG. Halucinácie a ako ich meria testovacia sada s „chytákmi".

**Po lekcii viete:**
- rozhodnúť RAG vs. fine-tuning vs. long context pre konkrétny prípad a rozhodnutie obhájiť,
- vysvetliť, čo je LoRA a prečo šetrí pamäť,
- navrhnúť vyhodnotenie (baseline surového modelu, testovacie otázky, chytáky).

**Zadanie 2** (cez lekcie 6–7): malý open model z HF + dlhý neznámy dokument; sprístupniť jeho obsah cez RAG **alebo** LoRA fine-tuning, zmerať proti baseline, porovnať prístupy.

---

## Lekcia 8 — Claude Code a agentové frameworky

**Materiál:** [llm-trendy.md](llm-trendy.md) (sekcie 1, 2, 5) + živé demá na hodine

Čo robí z LLM **agenta**: slučka model → nástroj → výsledok → model (ReAct). Tool use / function calling, MCP ako štandard pripájania nástrojov. **Claude Code** ako ukážka hotového agenta: práca s repozitárom, spúšťanie príkazov, kedy mu (ne)veriť. **LangChain / LangGraph** ako framework: reťazenie krokov, vlastné nástroje, jednoduchý agent — a kedy framework *ne*použiť (jednoduchý vzor bez frameworku býva lepší štart). Bezpečnosť agentov: prompt injection, least-privilege prístup k nástrojom, sandboxing. Context engineering a evaluácia/observability agentov.

**Po lekcii viete:**
- vysvetliť agentovú slučku a rozdiel medzi „chatbot" a „agent",
- napísať jednoduchý agent s jedným-dvoma nástrojmi (bez frameworku aj v LangChain),
- vymenovať hlavné riziká (prompt injection) a základné obrany,
- efektívne používať Claude Code pri vlastnej práci.

---

## Zhrnutie: dva princípy, ktoré sa oplatí odniesť

1. **Typ dát a úlohy určuje model.** Tabuľky → XGBoost. Obraz → CNN. Text/sekvencie → transformer. Optimalizácia bez gradientu → GA. Neťahajte LLM tam, kde jednoduchší model spraví lacnejšiu a vysvetliteľnejšiu prácu.
2. **Dáta + loss určujú, čo sa model naučí.** Rovnaká sieť a rovnaká slučka (forward → loss → backprop → Adam) dá dokončovač textu, asistenta aj embedding model — podľa toho, aké dáta a akú loss jej dáte. Kto rozumie tejto mechanike, rozumie celému modernému AI stacku.

---

### Všetky dokumenty predmetu

| Dokument | Obsah | Lekcia |
|---|---|---|
| [umela-inteligencia-prehlad.md](umela-inteligencia-prehlad.md) | taxonómia AI, stromy, XGBoost, MLP, CNN, GA | 1–3 |
| [vyvojove-prostredie.md](vyvojove-prostredie.md) | inštalácia (PyTorch, CUDA/MPS), vLLM, hardvér, cloud | príručka |
| [adam-optimalizator.md](adam-optimalizator.md) | tréningová slučka, backprop, Adam do detailu | 3 |
| [transformer-siete.md](transformer-siete.md) | attention, multi-head, positional encoding, generovanie | 4 |
| [llm-trening.md](llm-trening.md) | pretraining → base → SFT → Instruct | 5 |
| [llm-modely.md](llm-modely.md) | proprietárne / open-weight / open-source, výber podľa úlohy | 5 |
| [embeddings.md](embeddings.md) | tokenizácia, embeddingy, similarity, RAG pipeline | 6 |
| [llm-trendy.md](llm-trendy.md) | trendy, moderný retrieval, agentic RAG, kedy fine-tuning | 5, 7, 8 |
| [zadania/rozpoznavanie-obrazkov.md](zadania/rozpoznavanie-obrazkov.md) | **zadanie 1** — vlastná sieť + Adam + PyTorch | 3–4 |
| [zadania/RAG_Fine_tunning.md](zadania/RAG_Fine_tunning.md) | **zadanie 2** — RAG alebo LoRA fine-tuning | 6–7 |

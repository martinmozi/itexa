# Prehľad súčasných modelov — proprietárne, open-weight a open-source

> **Poradie čítania:** ← [Ako sa trénuje LLM](02-llm-trening.md) · **lekcia 5** · [Embeddingy](04-embeddings.md) →

> **Cieľ dokumentu:** zorientovať sa v dnešnej ponuke veľkých modelov. Kľúčom je pochopiť **tri stupne otvorenosti** (proprietárne API → otvorené váhy → plne otvorené vrátane tréningových dát) a vedieť si vybrať model **podľa úlohy** — OCR, kódovanie, tabuľkové dáta, embeddingy, lokálne nasadenie…
>
> *Stav: júl 2026. Krajina modelov sa mení každých pár mesiacov — konkrétne verzie berte ako momentku, kategórie a princípy výberu platia dlhodobo.*

Nadväzuje na [typy modelov](../02-typy-modelov/README.md) a [02-llm-trening.md](02-llm-trening.md) (ako sa LLM trénujú — vysvetľuje aj pojmy *base* a *Instruct*, ktoré sa v tabuľkách nižšie objavujú).

---

## Tri stupne otvorenosti

Toto je najdôležitejšie rozlíšenie — často sa všetko nesprávne hádže do jedného vreca „open source":

| Stupeň | Čo máte k dispozícii | Čo NEmáte | Dôsledok |
|---|---|---|---|
| **Proprietárny** | len API (platíte za tokeny) | váhy, dáta, detaily architektúry | najvyššia kvalita, ale závislosť od dodávateľa, dáta idú cez cudzí server |
| **Open-weight** | **váhy** na stiahnutie (Hugging Face), beží u vás | tréningové dáta, často aj tréningový kód | plná kontrola nad nasadením a dátami; tréning ale nezreprodukujete ani neauditujete |
| **Open-source (plne otvorený)** | váhy **+ tréningové dáta + kód + checkpointy** | nič podstatné | plná reprodukovateľnosť a audit — ideál pre výskum a výučbu; kvalitou zatiaľ za špičkou |

> **Pozor na licencie pri open-weight:** „stiahnuteľné váhy" ≠ „rob si s tým, čo chceš". Apache 2.0 a MIT sú skutočne voľné; Llama má vlastnú komunitnú licenciu s podmienkami; niektoré modely obmedzujú komerčné použitie. Pred nasadením vždy čítať licenciu.

---

## 1. Proprietárne modely (API)

| Rodina | Poskytovateľ | Silné stránky | Typické použitie |
|---|---|---|---|
| **Claude** (Fable 5, Opus 5, Sonnet 5, Haiku 4.5) | Anthropic | kódovanie, dlhodobé agentické úlohy, dlhý kontext (1M tokenov), práca s dokumentmi | programátorskí agenti (Claude Code), analýza dokumentov, enterprise asistenti |
| **GPT rodina** (GPT-5.x, o-séria) | OpenAI | všeobecná všestrannosť, reasoning modely, multimodalita, veľký ekosystém | chatboty, všeobecné aplikácie, hlasoví agenti |
| **Gemini** (2.5/3) | Google | natívna multimodalita (video, audio), veľmi dlhý kontext, integrácia s Google | spracovanie videa/audia, vyhľadávanie, Workspace |
| **Mistral Large** (API verzia) | Mistral AI | európsky poskytovateľ (GDPR argument), dobrý pomer cena/výkon | EU-hosted nasadenia |

**Kedy proprietárne API:** chcete najvyššiu kvalitu bez starostí o infraštruktúru, objem je malý až stredný a dáta smú opustiť firmu (alebo má poskytovateľ vhodné garancie).

---

## 2. Open-weight modely (váhy dostupné, dáta nie)

| Rodina | Vydavateľ | Licencia | Silné stránky |
|---|---|---|---|
| **Qwen 3 / 3.5** | Alibaba | Apache 2.0 | najsilnejší všestranný open-weight; kódovanie, reasoning, ~200 jazykov |
| **DeepSeek** (R1, V3) | DeepSeek | MIT | reasoning a matematika; destilované malé varianty |
| **Llama 4** (Scout…) | Meta | Llama licencia | extrémne dlhý kontext (až 10M tokenov), veľký ekosystém |
| **GLM-5** | Zhipu AI | MIT | agentické kódovanie, dlhý kontext |
| **Kimi K2** | Moonshot AI | vlastná | agentické úlohy vo veľkej škále |
| **Mistral / Ministral / Magistral** | Mistral AI | Apache 2.0 | efektivita — výkon na malom hardvéri, edge |
| **Gemma 3** | Google | Gemma licencia | kvalitné malé modely (1–27B) na lokálny beh |
| **Phi-4** | Microsoft | MIT | veľmi malé modely, edge/on-device |
| **Whisper** (ASR) | OpenAI | MIT | prepis reči na text — de facto štandard |

**Kedy open-weight:** dáta nesmú opustiť firmu, potrebujete predvídateľné náklady pri veľkom objeme, fine-tuning na vlastnú doménu (LoRA/QLoRA — viď [zadanie](../../zadania/RAG_Fine_tunning.md)), alebo offline/edge nasadenie. Menšie varianty (1–8B) bežia aj na bežnom GPU či kvantované na CPU.

---

## 3. Plne open-source (aj tréningové dáta)

Modelov, kde je verejné **všetko** — váhy, dáta, kód, priebežné checkpointy — je len hŕstka. Nie sú na špici benchmarkov, ale sú **jediné plne auditovateľné a reprodukovateľné**, preto sú zlatým štandardom pre výskum a výučbu:

| Model | Vydavateľ | Čo je otvorené | Poznámka |
|---|---|---|---|
| **OLMo 3** | Allen AI (AI2) | váhy, dáta (Dolma), kód, logy, checkpointy | najkompletnejší „úplne otvorený" moderný model |
| **Pythia** | EleutherAI | váhy, dáta (The Pile), kód, checkpointy | séria veľkostí — ideálna na štúdium, ako schopnosti rastú s veľkosťou |
| **SmolLM 2/3** | Hugging Face | váhy, dáta, kód | malé modely (135M–3B), skvelé na experimenty na notebooku |
| **LLM360 K2** | LLM360 | váhy, dáta, kód, celý tréningový priebeh | 65B „360°-otvorený" model |
| **StarCoder 2** | BigCode | váhy, dáta (The Stack) | otvorený kódovací model s auditovateľným korpusom |
| **BLOOM** | BigScience | váhy, dáta (ROOTS) | historicky prvý veľký plne otvorený model (2022), dnes prekonaný |

**Prečo na tom záleží:** len pri týchto modeloch viete odpovedať na otázky „*čo presne model videl pri tréningu?*", „*je v dátach môj text?*", „*prečo model vie X a nevie Y?*" — pri open-weight modeloch sú to dohady. Ako presne tréningové dáta formujú model, rozoberá [02-llm-trening.md](02-llm-trening.md).

---

## Výber modelu podľa úlohy

| Úloha | Odporúčanie | Poznámka |
|---|---|---|
| **OCR / extrakcia z dokumentov** | multimodálny LLM (Claude, Gemini, Qwen-VL) na komplexné dokumenty; klasické OCR (Tesseract, PaddleOCR) na jednoduchý čistý text | LLM zvláda tabuľky, formuláre, rukopis a rovno štruktúruje výstup (JSON) |
| **Kódovanie / programátorský agent** | Claude (Opus/Sonnet) cez API; open-weight: Qwen 3.5, GLM-5, DeepSeek | agentické kódovanie = model + nástroje (viď lekcia 8) |
| **Tabuľkové dáta** (predikcia, skóring) | ❌ **nie LLM** → **XGBoost / stromy** ([02-random-forest-a-xgboost.md](../02-typy-modelov/02-random-forest-a-xgboost.md)) | LLM sa hodí nanajvýš na *rozhranie* nad tabuľkou (text → SQL), nie na samotnú predikciu |
| **Embeddingy / RAG retrieval** | špecializované embedding modely: `bge-m3`, `multilingual-e5`, prípadne API embeddingy | malý model stačí; detaily v [04-embeddings.md](04-embeddings.md) |
| **Reranking** | `bge-reranker-v2-m3`, Cohere Rerank | cross-encoder, viď [04-embeddings.md](04-embeddings.md) |
| **Reasoning / matematika** | o-séria, DeepSeek R1, Claude s extended thinking | „premýšľajúce" modely — viac výpočtu pri inferencii |
| **Slovenčina / multilingválne** | veľké proprietárne modely; open-weight: Qwen, Gemma | malé open modely na slovenčine citeľne strácajú (aj kvôli tokenizácii — viď [04-embeddings.md](04-embeddings.md)) |
| **Lokálny beh na notebooku** | Qwen/Llama/Gemma 1–8B kvantované (Ollama, llama.cpp); na experimenty SmolLM | 4-bit kvantizácia zníži pamäť ~4× za malú stratu kvality |
| **Prepis reči (ASR)** | Whisper (open-weight) | beží aj lokálne |
| **Klasifikácia obrázkov (úzka úloha)** | vlastná malá **CNN** ([05-konvolucne-siete.md](../02-typy-modelov/05-konvolucne-siete.md)), prípadne fine-tunovaný ViT | nasadiť LLM na „je na páse chybný výrobok?" je zbytočne drahé |
| **Firemný chatbot nad dokumentmi** | RAG: embedding model + LLM (API alebo open-weight podľa citlivosti dát) | viď [04-embeddings.md](04-embeddings.md) a [zadanie](../../zadania/RAG_Fine_tunning.md) |

### Rozhodovací postup (zjednodušene)

```text
Je to tabuľková predikcia? ──► XGBoost, žiadny LLM.
Je to úzka obrazová úloha? ──► CNN / malý vision model.
        │
        ▼ (je to text / dokumenty / kód / dialóg)
Smú dáta von z firmy a je objem malý? ──► proprietárne API (najvyššia kvalita, nula infraštruktúry)
Dáta musia ostať doma / veľký objem / fine-tuning? ──► open-weight (Qwen, Llama, Mistral…)
Výskum, audit, výučba, reprodukovateľnosť? ──► plne open-source (OLMo, Pythia, SmolLM)
```

> **Konkrétne verzie vs. rodiny.** Tabuľky vyššie zámerne uvádzajú **rodiny** (Qwen, Llama, Gemma),
> nie presné Hugging Face ID — tie sa menia každých pár mesiacov. V [zadaní 2](../../zadania/RAG_Fine_tunning.md)
> nájdete overené ID staršej, ale stabilnej generácie; ak si na HF nájdete novšiu, pokojne ju použite,
> pipeline je rovnaká.

---

## Právne a etické mantinely

Výber modelu nie je len technické rozhodnutie — spolu s ním si vyberáte aj právny režim. Štyri veci,
ktoré treba vyriešiť **pred** nasadením, nie po ňom:

- **Kam tečú dáta.** Poslať prompt do proprietárneho API znamená odoslať jeho obsah tretej strane.
  Pri osobných údajoch to je podľa **GDPR** spracovanie, ktoré potrebuje právny základ, spracovateľskú
  zmluvu a ošetrený prenos mimo EÚ. Toto je najčastejší dôvod, prečo firma siahne po open-weight modeli
  bežiacom vo vlastnej infraštruktúre — nie cena, ale právo.
- **Licencia modelu.** Ako spomíname vyššie: Apache 2.0/MIT sú voľné, Llama a Gemma majú vlastné
  komunitné licencie s podmienkami, niektoré modely zakazujú komerčné použitie. Licencia sa **dedí**
  aj na to, čo modelom vygenerujete a na čom ho dotrénujete.
- **Autorské práva na tréningové dáta.** Pri open-weight modeloch nemôžete overiť, čo model videl —
  presne preto sú plne open-source modely cenné pre audit. Ak fine-tunujete na cudzích dátach, musíte
  mať právo ich na tento účel použiť.
- **EU AI Act.** Nariadenie triedi systémy podľa rizika. Pre bežnú firemnú aplikáciu z toho plynú
  hlavne dve povinnosti: **transparentnosť** (používateľ musí vedieť, že hovorí so strojom, a generovaný
  obsah má byť označiteľný) a pri **vysokorizikových** použitiach (nábor, úvery, vzdelávanie, zdravotníctvo,
  polícia) navyše dokumentácia, ľudský dohľad a riadenie rizík. Chatbot nad internou dokumentáciou je
  nízke riziko; skórovanie žiadateľov o úver je vysoké — a to je presne ten prípad z lekcie 2, kde je
  vysvetliteľný XGBoost lepšia voľba než čierna skrinka.

A jedna vec, ktorá nie je právna, ale etická: model preberá **skreslenia (bias)** z tréningových dát.
Ak historické dáta obsahujú diskriminačný vzor, model sa ho naučí ako čokoľvek iné — a nasadený vo
veľkom ho zopakuje tisíckrát denne. Meranie kvality na priemernej presnosti to nezachytí; treba sa
pozrieť na chybovosť **po skupinách** (viď [04-metriky.md](../01-prehlad/04-metriky.md)).

---

## Kontrolné otázky

1. Aký je rozdiel medzi *open-weight* a *open-source* modelom? Prečo na tom záleží pri audite?
2. Firma chce chatbota nad internými zmluvami, ktoré nesmú opustiť firmu. Ktorú kategóriu modelov zvolíte a prečo?
3. Prečo na predikciu rizika úveru z tabuľky nenasadíme LLM, hoci „vie všetko"?
4. Prečo embedding model v RAG nemusí byť veľký, kým generatívny model áno?
5. Čo všetko musí byť zverejnené, aby ste vedeli overiť, či model „nevidel" váš testovací dataset pri tréningu?
6. Firma chce LLM nasadiť na predbežné triedenie životopisov. Aké povinnosti z toho plynú a čo by ste namietli ešte pred technickým riešením?

---

### Zdroje

- [LLM360: Towards Fully Transparent Open-Source LLMs](https://arxiv.org/pdf/2312.06550), [LLM360 K2](https://arxiv.org/pdf/2501.07124) — čo presne znamená „plne otvorený" model
- [Hugging Face — Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) a [modelové karty](https://huggingface.co/models) — primárny zdroj na aktuálne verzie a licencie
- [Nariadenie EÚ o umelej inteligencii (AI Act) — oficiálny text](https://eur-lex.europa.eu/legal-content/SK/TXT/?uri=OJ:L_202401689)

> Sekundárne prehľady a „top 10" blogy zastarávajú do pár mesiacov — pri overovaní konkrétnej verzie
> vždy uprednostnite modelovú kartu na Hugging Face pred článkom.

### Súvisiace dokumenty

- [prehlad-predmetu.md](../../prehlad-predmetu.md) — prehľad celého predmetu (8 lekcií)
- [02-llm-trening.md](02-llm-trening.md) — **prvá polovica lekcie 5**: ako sa LLM trénujú
- [tutorials/02-typy-modelov](../02-typy-modelov/README.md) — stromy, XGBoost, MLP, CNN
- [04-embeddings.md](04-embeddings.md) — **nasledujúca lekcia**: embedding modely a RAG pipeline
- [01-vyvojove-prostredie.md](../00-prostredie/01-vyvojove-prostredie.md) — na čom vybraný model reálne spustíte

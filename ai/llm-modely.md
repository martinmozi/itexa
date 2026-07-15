# Prehľad súčasných modelov — proprietárne, open-weight a open-source

> **Cieľ dokumentu:** zorientovať sa v dnešnej ponuke veľkých modelov. Kľúčom je pochopiť **tri stupne otvorenosti** (proprietárne API → otvorené váhy → plne otvorené vrátane tréningových dát) a vedieť si vybrať model **podľa úlohy** — OCR, kódovanie, tabuľkové dáta, embeddingy, lokálne nasadenie…
>
> *Stav: júl 2026. Krajina modelov sa mení každých pár mesiacov — konkrétne verzie berte ako momentku, kategórie a princípy výberu platia dlhodobo.*

Nadväzuje na [umela-inteligencia-prehlad.md](umela-inteligencia-prehlad.md) (typy modelov) a [llm-trening.md](llm-trening.md) (ako sa LLM trénujú — vysvetľuje aj pojmy *base* a *Instruct*, ktoré sa v tabuľkách nižšie objavujú).

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
| **Claude** (Fable 5, Opus 4.8, Sonnet 5, Haiku 4.5) | Anthropic | kódovanie, dlhodobé agentické úlohy, dlhý kontext (1M tokenov), práca s dokumentmi | programátorskí agenti (Claude Code), analýza dokumentov, enterprise asistenti |
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

**Kedy open-weight:** dáta nesmú opustiť firmu, potrebujete predvídateľné náklady pri veľkom objeme, fine-tuning na vlastnú doménu (LoRA/QLoRA — viď [zadanie](zadania/RAG_Fine_tunning.md)), alebo offline/edge nasadenie. Menšie varianty (1–8B) bežia aj na bežnom GPU či kvantované na CPU.

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

**Prečo na tom záleží:** len pri týchto modeloch viete odpovedať na otázky „*čo presne model videl pri tréningu?*", „*je v dátach môj text?*", „*prečo model vie X a nevie Y?*" — pri open-weight modeloch sú to dohady. Ako presne tréningové dáta formujú model, rozoberá [llm-trening.md](llm-trening.md).

---

## Výber modelu podľa úlohy

| Úloha | Odporúčanie | Poznámka |
|---|---|---|
| **OCR / extrakcia z dokumentov** | multimodálny LLM (Claude, Gemini, Qwen-VL) na komplexné dokumenty; klasické OCR (Tesseract, PaddleOCR) na jednoduchý čistý text | LLM zvláda tabuľky, formuláre, rukopis a rovno štruktúruje výstup (JSON) |
| **Kódovanie / programátorský agent** | Claude (Opus/Sonnet) cez API; open-weight: Qwen 3.5, GLM-5, DeepSeek | agentické kódovanie = model + nástroje (viď lekcia 8) |
| **Tabuľkové dáta** (predikcia, skóring) | ❌ **nie LLM** → **XGBoost / stromy** ([prehľad](umela-inteligencia-prehlad.md)) | LLM sa hodí nanajvýš na *rozhranie* nad tabuľkou (text → SQL), nie na samotnú predikciu |
| **Embeddingy / RAG retrieval** | špecializované embedding modely: `bge-m3`, `multilingual-e5`, prípadne API embeddingy | malý model stačí; detaily v [embeddings.md](embeddings.md) |
| **Reranking** | `bge-reranker-v2-m3`, Cohere Rerank | cross-encoder, viď [embeddings.md](embeddings.md) |
| **Reasoning / matematika** | o-séria, DeepSeek R1, Claude s extended thinking | „premýšľajúce" modely — viac výpočtu pri inferencii |
| **Slovenčina / multilingválne** | veľké proprietárne modely; open-weight: Qwen, Gemma | malé open modely na slovenčine citeľne strácajú (aj kvôli tokenizácii — viď [embeddings.md](embeddings.md)) |
| **Lokálny beh na notebooku** | Qwen/Llama/Gemma 1–8B kvantované (Ollama, llama.cpp); na experimenty SmolLM | 4-bit kvantizácia zníži pamäť ~4× za malú stratu kvality |
| **Prepis reči (ASR)** | Whisper (open-weight) | beží aj lokálne |
| **Klasifikácia obrázkov (úzka úloha)** | vlastná malá **CNN** ([prehľad](umela-inteligencia-prehlad.md)), prípadne fine-tunovaný ViT | nasadiť LLM na „je na páse chybný výrobok?" je zbytočne drahé |
| **Firemný chatbot nad dokumentmi** | RAG: embedding model + LLM (API alebo open-weight podľa citlivosti dát) | viď [embeddings.md](embeddings.md) a [zadanie](zadania/RAG_Fine_tunning.md) |

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

---

## Kontrolné otázky

1. Aký je rozdiel medzi *open-weight* a *open-source* modelom? Prečo na tom záleží pri audite?
2. Firma chce chatbota nad internými zmluvami, ktoré nesmú opustiť firmu. Ktorú kategóriu modelov zvolíte a prečo?
3. Prečo na predikciu rizika úveru z tabuľky nenasadíme LLM, hoci „vie všetko"?
4. Prečo embedding model v RAG nemusí byť veľký, kým generatívny model áno?
5. Čo všetko musí byť zverejnené, aby ste vedeli overiť, či model „nevidel" váš testovací dataset pri tréningu?

---

### Zdroje (stav júl 2026)

- [Best Open-Source LLMs — AceCloud](https://acecloud.ai/blog/best-open-source-llms/), [TECHSY leaderboard](https://techsy.io/en/blog/best-open-source-llms-2026), [D-Central self-host guide](https://d-central.tech/best-local-llm-2026-pleb-open-weight-model-guide/)
- [Olmo 3 — plne otvorený model AI2](https://www.digitalocean.com/community/tutorials/olmo-3-allen-ai-open-source-llm), [LLM360 K2 paper](https://arxiv.org/pdf/2501.07124), [LLM360: Towards Fully Transparent Open-Source LLMs](https://arxiv.org/pdf/2312.06550)
- [Open Source LLMs 2026 — Morph](https://www.morphllm.com/open-source-llm), [PocketLLM license ranking](https://pocketllm.app/blog/best-open-source-llm-2026/)

### Súvisiace dokumenty

- [prehlad-predmetu.md](prehlad-predmetu.md) — prehľad celého predmetu (8 lekcií)
- [llm-trening.md](llm-trening.md) — ako sa LLM trénujú (pretraining → Instruct)
- [umela-inteligencia-prehlad.md](umela-inteligencia-prehlad.md) — stromy, XGBoost, MLP, CNN
- [llm-trendy.md](llm-trendy.md) — aktuálne trendy (RAG, agenti, fine-tuning)
- [embeddings.md](embeddings.md) — embedding modely a RAG pipeline

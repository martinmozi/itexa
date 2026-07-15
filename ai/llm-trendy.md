# Aktuálne trendy vo veľkých jazykových modeloch (2025–2026)

> **Cieľ dokumentu:** rýchly prehľad toho, kam sa LLM svet hýbe a čo sa oplatí vedieť — moderný retrieval, agentic RAG a rozhodovanie, kedy má fine-tuning ešte zmysel. Základy, na ktoré tento text nadväzuje: [transformery](transformer-siete.md), [embeddingy a RAG](embeddings.md), [tréning LLM](llm-trening.md); prehľad konkrétnych modelov je v [llm-modely.md](llm-modely.md).

## 1. Čo je nové/na hrane (2025–2026)

- Reasoning modely ako predvolený "engine" a test-time compute scaling namiesto len scaling parametrov.
- Bezpečnosť agentov – prompt injection, sandboxing – kritické, keď agent má prístup k terminálu/kódu/prehliadaču.
- Multimodálni a voice-first agenti (real-time hlas/video).
- Menšie destilované a lokálne modely popri veľkých (náklady/latencia).

---

## 2. Priority na učenie v polovici 2026

1. **MCP** a agentová interoperabilita
2. **Context engineering** namiesto len prompt engineering
3. **Agentová evaluácia a observability**
4. **Bezpečnosť agentov** – least-privilege prístup k nástrojom, obrana proti prompt injection
5. Vedieť postaviť **jednoduchý** agentový vzor bez frameworku aj poznať kedy siahnuť po ťažšom (LangGraph a pod.)
6. Efektívna práca s **reasoning modelmi** – jasné zadanie, menej "promptových trikov"
7. Hybrid RAG + rozhodovanie RAG vs. long context vs. fine-tuning/distillation
8. **Distillation** ako nástroj na lacné špecializované modely

---

## 3. Moderné vyhľadávanie (retrieval) – detail

- **Hybrid search** – kombinácia vektorového (dense/embedding) + lexikálneho (BM25/sparse, napr. SPLADE) vyhľadávania. Vektor chytí sémantiku, BM25 chytí presné zhody (kódy, skratky, mená, ID).
- **Reranking** – najprv sa vytiahne širší set (top 20–50) hybridným searchom, potom cross-encoder reranker (Cohere Rerank, BGE-reranker, ColBERT) vyberie skutočný top-3–5 do promptu. Jedno z najlacnejších a najúčinnejších vylepšení kvality.
- **Query transformation** – prepis/rozšírenie query, HyDE (najprv vygeneruj hypotetickú odpoveď, tú embedduj a hľadaj podľa nej), rozklad zloženej otázky na podotázky (multi-hop).
- **Metadata filtering** – kombinácia vektorového searchu so štruktúrovanými filtrami (dátum, zdroj, oddelenie, ACL/permissions) – podporujú prakticky všetky vektorové DB natívne.
- **Chunking** – posun od "fixný počet tokenov" k semantic/structure-aware chunkingu (rešpektuje nadpisy, tabuľky, code bloky) + overlap medzi chunkami.
- **Embedding modely** – výber má veľký vplyv (multilingválnosť, Matryoshka embeddings na škálovanie dimenzie, multi-vector/late-interaction ako ColBERT pre vyššiu presnosť).

---

## 4. Small-to-big retrieval

**Princíp:** indexuješ/vyhľadávaš na malých kusoch (presnosť matchu), ale do LLM posielaš väčší, súvislý kontext – malé chunky sa dobre *vyhľadávajú*, ale samostatne často chýba kontext (halucinácie, neúplné odpovede).

### Techniky
- **Parent-child chunking** – malé "child" chunky majú v metadátach `parent_id` na väčší nadradený blok/sekciu/dokument. Search nájde child, aplikácia dotiahne parent z docstore.
- **Sentence-window retrieval** – indexujú sa jednotlivé vety, pri nájdení sa vráti okno ±N viet okolo nej.
- **Auto-merging retriever** – hierarchický strom chunkov; ak sa nájde dosť child uzlov pod jedným parentom, zlúčia sa a vráti sa rovno parent.
- **RAPTOR-style** – strom sumárov na viacerých úrovniach abstrakcie (chunk → sumár sekcie → sumár dokumentu), retrieval siaha na úroveň podľa granularity otázky.

### Implementácia / voľba databázy
Toto je väčšinou vzor na úrovni aplikácie (vektor → id → fetch plného textu), nie špeciálna vlastnosť DB:

| Nástroj/DB | Poznámka |
|---|---|
| **LlamaIndex** | `AutoMergingRetriever`, hierarchical node parser – hotový vzor |
| **LangChain** | `ParentDocumentRetriever` – hotový vzor |
| **Pinecone, Weaviate, Qdrant, Milvus, Chroma** | vektorová vrstva, parent-child cez metadata + samostatný lookup |
| **Postgres + pgvector** | čoraz obľúbenejšie – parent/child ako relačné tabuľky s FK + vektor search v tej istej DB |
| **Elasticsearch/OpenSearch** | natívny hybrid search (BM25 + vektor), časté v enterprise |
| **Graf DB (Neo4j a pod.)** | keď ide viac o vzťahy medzi entitami než o hierarchiu (GraphRAG) |

Malé chunky idú do vektorovej DB, plné dokumenty do jednoduchého docstore (Redis, Mongo, lokálny file store).

---

## 5. Čo je agentic RAG

Namiesto pevného "retrieve raz → stuff do promptu → generuj", LLM ako agent sám rozhoduje:

- **či** je retrieval vôbec potrebný,
- **čo** presne hľadať (vie si query preformulovať),
- **koľkokrát** hľadať – iteratívne, multi-hop (nájde niečo, zistí že to nestačí, hľadá znova s inou query),
- **kde** hľadať – router rozhoduje medzi vektorovou DB, SQL databázou, webom, internými API,
- **kedy má dosť info** na odpoveď, prípadne si vie odpoveď spätne overiť voči zdrojom (self-check/reflection).

### Typické vzory
- **ReAct slučka** (Thought → Action/search → Observation → opakuj)
- **Corrective RAG / Self-RAG** – over, či retrieved dokumenty naozaj podporujú návrh odpovede, inak re-query alebo fallback
- **Multi-agentový RAG** – samostatný agent na retrieval, samostatný na verifikáciu

### Kedy sa oplatí
Pri komplexných/multi-hop otázkach a heterogénnych zdrojoch dát.

**Cena:** vyššia latencia a náklady (viac LLM volaní), náročnejšie debugovanie – vyžaduje poriadny observability/eval setup.

---

## 6. Kedy má fine-tuning stále zmysel

### Áno, oplatí sa:
- **Distillation** – natrénovať malý/lacný model na výstupoch (alebo reasoning trace) veľkého modelu pre úzku úlohu → lacná a rýchla inferencia vo veľkom objeme.
- **Konzistentný formát/štýl výstupu** – spoľahlivý JSON/function-calling formát, firemný tón, doménový žargón (právo, medicína) – lacnejšie než dlhé inštrukcie/few-shot príklady v každom volaní.
- **Edge/on-device modely** – keď treba bežať lokálne (privacy, latencia, offline) s malým modelom dobrým na úzku úlohu.
- **Chain-of-thought distillation** – trénovanie malých "reasoning" modelov na CoT stopách veľkého modelu.
- **Preference optimization** (DPO a podobne) – ladenie preferovaného štýlu/kvality z porovnávacích párov, bez plného RLHF.
- Keď sú **vyčerpané prompt/RAG optimalizácie** a presnosť/formát stále nestačí.

### Nie, neoplatí sa:
- Model "nepozná fakt X" → to je problém pre RAG/kontext, nie fine-tuning (fine-tuning je zlý na vkladanie nových faktov, dobrý na naučenie ŠTÝLU).
- Dáta sa často menia → fine-tuning treba opakovať, drahšie než aktualizovať retrieval korpus.
- Dobrý prompt + few-shot na silnom reasoning modeli už problém rieši → netreba pridávať zložitosť.

---

## Kontrolné otázky

1. Prečo hybrid search (vektor + BM25) prekonáva čisto vektorové vyhľadávanie? Uveďte príklad dotazu, kde zlyhá samotný vektor.
2. Vysvetlite princíp small-to-big retrievalu: prečo sa vyhľadáva na malých chunkoch, ale do promptu ide väčší text?
3. Čím sa agentic RAG líši od klasického „retrieve raz → generuj"? Aká je cena za tú flexibilitu?
4. Model „nepozná fakt X" — prečo to fine-tuning nerieši dobre a čo je správne riešenie?
5. Vymenujte dve situácie, kedy sa fine-tuning naopak oplatí.

---

*Poznámka: ide o syntézu trendov bez prístupu k živému vyhľadávaniu – pri úplne najnovších zmenách (posledné týždne pred dátumom čítania) odporúčam doplniť aktuálne zdroje.*

### Súvisiace dokumenty

- [prehlad-predmetu.md](prehlad-predmetu.md) — prehľad celého predmetu (8 lekcií)
- [llm-modely.md](llm-modely.md) — prehľad modelov (proprietárne / open-weight / open-source)
- [llm-trening.md](llm-trening.md) — ako sa LLM trénujú (pretraining → Instruct)
- [embeddings.md](embeddings.md) — embeddingy a RAG pipeline do detailu
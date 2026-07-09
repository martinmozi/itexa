# Zadanie: Odpovedanie na otázky z dlhého textu (RAG alebo fine-tuning)

## Cieľ

Vziať **malý otvorený jazykový model** z Hugging Face (napr. Llama, Mistral, Qwen, Phi…),
ktorý daný text **nepozná**, a upraviť pipeline tak, aby vedel **spoľahlivo odpovedať na
otázky** z jedného dostatočne dlhého, modelu neznámeho dokumentu (článok, kapitola, kniha,
technická dokumentácia).

Máte na výber **dva prístupy** — vyberte si (alebo vám bude pridelený) **jeden**:

- **Úloha A — RAG (Retrieval-Augmented Generation):** dokument sa neučí do váh modelu.
  Rozdelí sa na kúsky, zaindexuje cez embeddingy a pri každej otázke sa relevantné kúsky
  **vyhľadajú a vložia do promptu** ako kontext.
- **Úloha B — Fine-tuning:** obsah dokumentu (resp. dvojice otázka–odpoveď z neho) sa
  **doučí priamo do modelu** (typicky metódou **LoRA / QLoRA**), takže model odpovedá
  „z hlavy" bez externého kontextu.

Podstatná je **rovnaká výstupná schopnosť** (odpovedať na otázky z textu) dosiahnutá
**dvoma principiálne odlišnými cestami** — a pochopenie, **kedy sa ktorá oplatí**.

> Teoretické podklady k RAG (tokenizácia, embeddingy, chunking, vyhľadávanie, similarity)
> sú v samostatnom dokumente `../embeddings.md`.
> Kontext RAG vs. fine-tuning a moderné trendy sú v `../llm-trendy.md`.

---

## Voľba dokumentu (spoločné pre obe úlohy)

Vyberte si **jeden dostatočne dlhý text**, ktorý model **nemá napamäť** — tzn. **nie** slávnu
knihu z tréningových dát (žiadny Harry Potter, Biblia, Wikipedia „Slovensko"…). Cieľom je, aby
model **bez vašej úpravy odpovedať nevedel**.

Vhodné zdroje:

- **vlastný / firemný dokument, skriptá, diplomovka, manuál, zmluva,**
- **odborný článok alebo `arXiv` preprint** (novší než tréningové dáta modelu),
- **menej známa kniha z [Project Gutenberg](https://www.gutenberg.org/)**,
- **dokumentácia knižnice**, ktorú model nepozná.

Požiadavky na text:

- **dĺžka aspoň ~5–10 normostrán** (rádovo tisíce slov; kniha je vítaná),
- **súvislý obsah** s faktami, na ktoré sa dá pýtať (mená, čísla, definície, deje).

> **Dôkaz „neznámosti" (povinné):** ešte pred akoukoľvek úpravou položte **surovému modelu**
> 3–5 otázok z textu a zapíšte jeho odpovede. Mali by byť **nesprávne, vymyslené (halucinácie)
> alebo „neviem"**. Toto je vaša **baseline** — s ňou budete porovnávať výsledok.

### Testovacia sada otázok

Pripravte **10–15 otázok** k dokumentu spolu so **správnymi odpoveďami** (referencia). Mix:

- **faktické** („Koľko…", „Kedy…", „Kto…"),
- **na porozumenie** (vyžadujúce spojenie informácií z viacerých miest),
- **2–3 chytáky** — otázky, ktorých odpoveď v texte **nie je** (správna odpoveď = „v texte to
  nie je uvedené"; testuje, či si model nevymýšľa).

---

## Voľba modelu (spoločné)

Vyberte si **jeden** malý **instruct** model z Hugging Face, ktorý sa zmestí na vaše zdroje
(GPU, prípadne CPU kvantovaný). Máte na výber z viacerých rodín — orientačný prehľad:

| Model (Hugging Face ID) | Veľkosť | Rodina | Poznámka |
|---|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Llama | najmenší, rýchly; vyžaduje súhlas s licenciou na HF |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Llama | silnejší Llama, stále nenáročný |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Llama | výkonný, odporúčaná kvantizácia |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Mistral | osvedčený, silný na 7B triede |
| `mistralai/Ministral-8B-Instruct-2410` | 8B | Mistral | novší, dobrý na dlhší kontext |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Qwen | najmenší na rozbehnutie aj na CPU |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Qwen | výborný pomer kvalita/veľkosť, viacjazyčný |
| `Qwen/Qwen2.5-3B-Instruct` / `-7B-Instruct` | 3–7B | Qwen | silnejšie varianty |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | Phi | dobrý na málo zdrojoch, dlhý kontext |
| `google/gemma-2-2b-it` | 2B | Gemma | kompaktný, kvalitný |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | SmolLM | plne otvorený, ľahký |
| `tiiuae/Falcon3-3B-Instruct` | 3B | Falcon | alternatíva mimo veľkých rodín |

> *Tip:* ak máte málo pamäte, použite **4-bit kvantizáciu** (`bitsandbytes`) alebo menší model
> (0.5–2B). Modely s licenciou (Llama, Gemma) vyžadujú na HF **prijatie podmienok** a prihlásenie
> (`huggingface-cli login`). Pokojne si porovnajte **viacero modelov** medzi sebou — je to plus.
> Dôležitejšie než veľkosť je, aby ste **celý proces prešli** a vedeli ho porovnať s baseline.

### Povolené nástroje

- **Python** + **Hugging Face** `transformers`, `datasets`, `accelerate`.
- **Úloha A (RAG):** knižnica na embeddingy (`sentence-transformers`) + vektorové vyhľadávanie
  (`faiss`, `chromadb`, alebo aj vlastný výpočet kosínusovej podobnosti cez NumPy).
- **Úloha B (fine-tuning):** `peft` (LoRA/QLoRA), `trl` (`SFTTrainer`), `bitsandbytes`.

---

# Úloha A — RAG

Cieľ: model odpovedá **s pomocou vyhľadaného kontextu**, váhy sa nemenia.

## A1 — Príprava dát (indexovanie, offline)

1. **Chunking** — dokument rozdeľte na kúsky (napr. 200–500 tokenov s prekryvom ~50).
   Zvážte delenie po odsekoch/vetách, nie naslepo v strede vety.
   > *Podklad:* stratégie chunkingu a metadáta sú rozpísané v `../embeddings.md` (Časť 2).
2. **Embeddingy** — každý chunk preveďte na vektor embeddovacím modelom
   (napr. `sentence-transformers/all-MiniLM-L6-v2` alebo viacjazyčný `intfloat/multilingual-e5-small`).
3. **Index** — vektory (a k nim pôvodný text + metadáta) uložte do vektorovej databázy /
   FAISS indexu.
   > *Hint:* embeddingy **normalizujte** a používajte kosínusovú podobnosť — prečo, viď
   > `../embeddings.md` (sekcia o normalizácii).

## A2 — Dotaz (online)

1. Otázku embedujte **tým istým** modelom ako chunky.
2. Nájdite **top-k** najpodobnejších chunkov (napr. `k = 3–5`).
3. Zostavte **prompt** vo formáte:
   ```
   Odpovedaj IBA na základe nasledujúceho kontextu. Ak odpoveď v kontexte nie je,
   povedz „v texte to nie je uvedené".

   Kontext:
   {vyhľadané chunky}

   Otázka: {otázka}
   ```
4. Prompt pošlite LLM a vypíšte odpoveď **spolu s tým, z ktorých chunkov čerpala** (zdroje).

## A3 — Experimenty (RAG)

Zdokumentujte vplyv nastavení do tabuľky:

| Veľkosť chunku | Prekryv | k (počet chunkov) | Embed model | Správne odpovede (z 15) |
|---|---|---|---|---|
| 500 | 50 | 3 | MiniLM | ? |
| 250 | 50 | 5 | MiniLM | ? |
| 500 | 50 | 3 | e5 | ? |

> *Na zamyslenie:* čo sa stane pri **priveľkom** `k` (šum v kontexte) a čo pri **primalom**
> (chýbajúca informácia)? Ako sa RAG správa pri **chytákoch** (odpoveď nie je v texte)?

---

# Úloha B — Fine-tuning

Cieľ: obsah **doučiť do váh** modelu metódou **LoRA/QLoRA**, aby odpovedal bez externého
kontextu.

## B1 — Príprava tréningových dát

Model sa neučí zo surového textu dobre — potrebuje **inštrukčný formát**. Vytvorte dataset
**dvojíc otázka → odpoveď** (prípadne inštrukcia → odpoveď) postavených nad vaším dokumentom:

1. Z textu vygenerujte **desiatky až stovky** dvojíc Q&A pokrývajúcich fakty z dokumentu.
   > *Hint:* dvojice môžete pripraviť **ručne**, alebo si ich nechať **vygenerovať silnejším
   > LLM** z jednotlivých pasáží (a potom prekontrolovať). Napíšte, ako ste ich získali.
2. Naformátujte ich do **chat/inštrukčnej šablóny** daného modelu
   (`tokenizer.apply_chat_template`).
3. Časť dvojíc **odložte na test** (nesmú byť v tréningu) — aby ste merali, či sa model naozaj
   naučil obsah, a nie len zapamätal konkrétne vety.

## B2 — Tréning (LoRA / QLoRA)

1. Načítajte model (ideálne **4-bit**, QLoRA) a pridajte **LoRA adaptéry**
   (`peft`, `LoraConfig` — `r`, `alpha`, `target_modules`).
2. Trénujte cez `trl.SFTTrainer` (alebo vlastnú slučku) niekoľko epôch, sledujte **loss**.
3. Uložte **LoRA adaptér** (nie celý model — stačia váhy adaptéra).

> *Hint:* sledujte **overfitting** — pri malom datasete a veľa epochách si model zapamätá
> vety doslova. Cieľ je, aby vedel odpovedať aj na **inak formulovanú** otázku.

## B3 — Experimenty (fine-tuning)

| LoRA `r` | Epochy | LR | Počet Q&A | Správne odpovede (z 15) |
|---|---|---|---|---|
| 8 | 3 | 2e-4 | ? | ? |
| 16 | 3 | 2e-4 | ? | ? |
| 16 | 6 | 2e-4 | ? | ? |

> *Na zamyslenie:* ako sa fine-tunovaný model správa pri **chytákoch**? Má tendenciu
> **halucinovať** viac než RAG? Čo sa stane, keď sa dokument **zmení** — čo treba prerobiť?

---

## Vyhodnotenie (spoločné pre obe úlohy)

1. Položte model **rovnakých 10–15 testovacích otázok** ako baseline (surový model).
2. Do tabuľky porovnajte: **surový model** vs. **vaše riešenie** (RAG alebo fine-tuned).

   | Otázka | Správna odpoveď | Surový model | Vaše riešenie | OK? |
   |---|---|---|---|---|
   | … | … | … | … | ✓/✗ |

3. Spočítajte **úspešnosť** (koľko z N otázok zodpovedané správne) a zvlášť vyhodnoťte
   **chytáky** (nevymýšľa si model, keď odpoveď v texte nie je?).

---

## Diskusná otázka (do správy)

Stručne odpovedzte (podklad: `../llm-trendy.md`, sekcia „Kedy má fine-tuning stále zmysel"):

- Kedy sa oplatí **RAG** a kedy **fine-tuning**? Uveďte po 2 konkrétne situácie z praxe.
- Ako každý z prístupov rieši **aktualizáciu obsahu** (dokument sa zmení / pribudne nový)?
- Ktorý prístup viac **halucinuje** a prečo? Ako sa dá halucinácie obmedziť?
- Čo je **LoRA** a prečo doučujeme len malé adaptéry namiesto celého modelu (pamäť, čas)?
- (Ak ste robili len jednu úlohu) V čom by druhý prístup na **vašom konkrétnom** dokumente
  fungoval lepšie/horšie?

---

## Odovzdanie

1. **Zdrojový kód** — celá pipeline zvolenej úlohy (A alebo B), spustiteľná.
2. **Zvolený dokument** (alebo odkaz naň) a **testovacia sada** 10–15 otázok so správnymi
   odpoveďami.
3. **Uložený artefakt** — A: vektorový index; B: LoRA adaptér.
4. **Krátka správa (1–2 strany):**
   - zvolený model, dokument a dôkaz, že ho model **nepoznal** (baseline odpovede),
   - popis pipeline a nastavení,
   - **tabuľka experimentov** (A3 alebo B3),
   - **porovnanie surový vs. upravený model** vrátane chytákov,
   - odpoveď na diskusnú otázku.

## Hodnotenie (orientačne)

| Kritérium | Body |
|---|---|
| Voľba vhodného (neznámeho) dokumentu + baseline dôkaz | 10 % |
| Testovacia sada otázok vrátane chytákov | 10 % |
| Funkčná pipeline (RAG **alebo** fine-tuning) | 35 % |
| Experimenty s nastaveniami + tabuľka | 20 % |
| Vyhodnotenie a porovnanie so surovým modelom | 15 % |
| Diskusná otázka v správe | 10 % |
| Bonus: **obe** úlohy (RAG aj fine-tuning) a ich porovnanie | +15 % |
| Prehľadnosť kódu a správy | 5 % |

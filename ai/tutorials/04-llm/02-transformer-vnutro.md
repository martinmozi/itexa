# Vnútro transformera — od tokenu po odpoveď

> **Poradie čítania:** ← [Transformery a attention](01-transformer-siete.md) · **lekcia 4** · [Ako sa trénuje LLM](03-llm-trening.md) →

> **Cieľ dokumentu:** dopovedať to, čo [predchádzajúci dokument](01-transformer-siete.md) nechal na úrovni myšlienky — **konkrétne čísla a celú cestu textu modelom**, od tokenizéra až po hotovú vetu. Dokument má tri časti; ich mapa je hneď pod referenčnou tabuľkou.

Predchádzajúci dokument vysvetlil **mechanizmus** (Q, K, V, softmax, multi-head, maska). Tento vysvetľuje **inžinierstvo okolo neho**: čo model vlastne dostáva na vstupe, prečo má práve 4096-rozmerné vektory, prečo 32 vrstiev, prečo z tisícok vstupných tokenov vzíde jeden jediný, koľko pamäte si vyžiada kontext 100 000 tokenov a prečo je prvý token odpovede pomalý a ďalšie rýchle.

Všetky čísla nižšie počítam na jednom **referenčnom modeli** veľkosti ~8B (rozmery zodpovedajú Llama-3-8B; iné modely majú iné čísla, ale rovnakú štruktúru):

| Symbol | Význam | Hodnota |
|---|---|---|
| `d_model` | rozmer vektora jedného tokenu (skrytý stav) | **4096** |
| `n_layers` | počet transformer blokov nad sebou | **32** |
| `n_heads` | počet attention hláv | **32** |
| `d_head` | rozmer jednej hlavy (`d_model / n_heads`) | **128** |
| `n_kv_heads` | počet hláv pre Key/Value (GQA — nižšie) | **8** |
| `d_ff` | vnútorný rozmer feed-forward vrstvy | **14336** |
| `vocab` | veľkosť slovníka tokenov | **128 256** |
| `n_ctx` | kontextové okno (max. počet tokenov) | **8192** |

---

## Mapa dokumentu

| Časť | Sekcie | Na akú otázku odpovedá |
|---|---|---|
| **A — Vstup** | [1. Tokenizácia](#1-tokenizácia-ako-sa-z-textu-stanú-čísla) · [2. Z ID na vektor](#2-z-token-id-na-vektor-embedding-matica-a-pozícia) · [okno vs. šírka](#dve-osi-vstupnej-matice-šírka-a-kontextové-okno) | Čo presne model dostáva, keď mu pošlem vetu, čo v tom ohraničuje kontextové okno a čo kvôli tokenom vôbec nevidí? |
| **B — Priechod** | [3. Rozmery, parametre a škálovanie](#3-odkiaľ-sa-berie-veľkosť-vektorov) · [4. Jedna vrstva](#4-cesta-jedného-vektora-jednou-vrstvou) · [5. Feed-forward a MoE](#5-feed-forward-vrstva-každý-vektor-sám-za-seba) | Prečo 4096 a 32 vrstiev, čím sa od toho líši 405B model, a kde je v modeli uložená znalosť? |
| **C — Výstup a limity** | [6. Výstupný token](#6-ako-vznikne-výstupný-token--a-prečo-len-jeden) · [7. Autoregresia a KV cache](#7-ďalší-prechod-autoregresia-a-kv-cache) · [8. Kontext](#8-kontext-krátka-správa-dlhá-správa-a-prečo-má-okno-strop) · [9. Učenie v kontexte](#9-učenie-v-kontexte-in-context-learning) · [10. Parametre](#10-prehľad-parametrov-transformera) | Prečo vypadne len jeden token, prečo je prvý pomalý, prečo okno nie je nekonečné a ako sa model „učí" z promptu? |

---

## ČASŤ A — Vstup: z textu na vektory

> Čo model naozaj dostáva na vstupe: ako tokenizér rozdelí text na čísla a ako sa z čísel stanú vektory, s ktorými sa už dá počítať.

### 1. Tokenizácia: ako sa z textu stanú čísla

Model nikdy nevidí písmená. Na vstupe dostáva **postupnosť celých čísel** — indexov do slovníka. Prevod textu na tieto indexy robí **tokenizér**, ktorý **nie je súčasťou neurónovej siete**: je to samostatný, deterministický kus kódu s vlastným natrénovaným slovníkom a vlastnými súbormi (`tokenizer.json`, `tokenizer.model`). Neobsahuje žiadne váhy, ktoré by sa učili gradientom, a pri inferencii beží na CPU za mikrosekundy.

Dôležitý dôsledok: **model a tokenizér tvoria nerozlučný pár.** Token ID `4521` znamená v každom modeli niečo úplne iné — je to len číslo riadku v embedding matici. Použiť tokenizér od GPT-4 na Llamu znamená dostať nezmysly, nie horšiu kvalitu.

```text
"Aké je hlavné mesto Slovenska?"
        │
        ▼   tokenizér (CPU, bez váh, deterministický)
[128000, 32, 962, 1662, 6156, 22801, 61464, 30]
        │
        ▼   model (GPU, 8 miliárd váh)
[128000, 32, 962, 1662, 6156, 22801, 61464, 30, 40522]   ← pribudol jeden token
        │
        ▼   detokenizér
"Aké je hlavné mesto Slovenska? Bratislava"
```

*(ID sú ilustračné.)*

#### Prečo subwordy a nie slová alebo znaky

Voľba jednotky je kompromis medzi veľkosťou slovníka a dĺžkou sekvencie — a obe krajnosti majú svoju cenu:

| Jednotka | Veľkosť slovníka | Dĺžka sekvencie | Hlavný problém |
|---|---|---|---|
| **znaky / bajty** | 256 | 4–6× dlhšia | attention je `O(n²)` → 4× dlhšia sekvencia = 16× drahšia; model musí časť kapacity spotrebovať na skladanie slov |
| **slová** | milióny (a v slovenčine každý pád zvlášť) | najkratšia | neznáme slová (*OOV*), preklepy, `lm_head` obrovský, riedke gradienty |
| **subwordy** | 32 000 – 256 000 | kompromis | hranice tokenov nesedia s hranicami slov (viď dôsledky nižšie) |

Subwordový slovník sa **neurčuje ručne** — učí sa štatisticky z korpusu algoritmom BPE (*Byte-Pair Encoding*): začne sa jednotlivými znakmi/bajtmi a opakovane sa zlučuje najčastejšia dvojica susedných symbolov, kým slovník nedosiahne cieľovú veľkosť. Vďaka tomu dostanú časté kmene vlastný token a zriedkavé slová sa poskladajú z menších kúskov. **Celý algoritmus krok po kroku, s ručne prepočítaným príkladom, je v [05-embeddings.md](05-embeddings.md#ako-sa-slovník-vôbec-naučí-bpe-krok-po-kroku)** — tu ho nebudeme opakovať a pozrieme sa na to, čo z neho vyplýva pre generatívne modely.

#### Tri rodiny tokenizérov

| Rodina | Ako pracuje | Ako kóduje medzeru | Kto ju používa |
|---|---|---|---|
| **byte-level BPE** | BPE nad **bajtmi** UTF-8 (nie znakmi) — slovník vždy obsahuje všetkých 256 bajtov | medzera sa lepí na **začiatok nasledujúceho slova** (`Ġnárok` v GPT-2 zápise) | GPT-2/3/4/4o (`tiktoken`), Llama 3, Qwen, DeepSeek, novšie Mistral |
| **SentencePiece** | text sa berie ako surový reťazec vrátane medzier, bez predbežného delenia podľa medzier; vnútri buď BPE alebo *unigram* model | medzera je viditeľný znak `▁` (U+2581) | Llama 1/2, Mistral 7B, Gemma, T5, mBART |
| **WordPiece** | greedy longest-match proti slovníku | pokračovanie slova sa značí `##` (`dovolen`, `##ky`) | BERT a jeho potomkovia — dnes hlavne encoder/embedding modely |

Pre generatívne LLM dnes **jednoznačne vedie byte-level BPE**, a to z jedného praktického dôvodu: keďže slovník obsahuje všetkých 256 bajtov, **nikdy nevznikne `[UNK]`**. Emoji, čínsky znak, ľubovoľná binárna postupnosť, preklep — všetko sa dá vždy rozložiť aspoň na jednotlivé bajty a vždy sa dá spätne presne zložiť. Detokenizácia je bezstratová.

Reálne modely a ich tokenizéry:

| Model | Tokenizér | `vocab` | Poznámka |
|---|---|---|---|
| GPT-2 | byte-level BPE | 50 257 | prvý široko použitý byte-level BPE |
| GPT-3.5 / GPT-4 | `tiktoken` `cl100k_base` | 100 277 | vlastné tokeny pre odsadenie kódu |
| GPT-4o | `tiktoken` `o200k_base` | ~200 000 | väčší slovník → citeľne menej tokenov v neanglických jazykoch |
| Llama 2 | SentencePiece BPE | 32 000 | číslice **po jednej** |
| Llama 3 | byte-level BPE (`tiktoken` štýl) | 128 256 | 128 000 bežných + 256 špeciálnych |
| Mistral 7B | SentencePiece BPE | 32 000 | |
| Gemma | SentencePiece | 256 000 | veľký multilingválny slovník |
| Qwen 2.5 / 3 | byte-level BPE | ~151 000 | silná čínština aj kód |

Skok z 32k na 128k+ za posledné roky nie je kozmetika: väčší slovník = kratšie sekvencie = lacnejší `O(n²)` prefill a menej prechodov pri generovaní. Platí sa zaň väčšou embedding maticou a `lm_head` (pri referenčnom modeli 2 × 525 M parametrov, viď [sekciu 3](#3-odkiaľ-sa-berie-veľkosť-vektorov)).

#### Ako vyzerá delenie na tokeny v praxi

```text
"Zamestnanec má nárok na 25 dní dovolenky."

byte-level BPE (␣ = medzera patriaca k tokenu):
  ["Z", "ames", "tan", "ec", "␣má", "␣ná", "rok", "␣na", "␣25", "␣d", "ní", "␣do", "volen", "ky", "."]
                                                                                              ≈ 15 tokenov
tá istá veta po anglicky:
  ["Employee", "␣is", "␣entitled", "␣to", "␣25", "␣days", "␣of", "␣vacation", "."]
                                                                                              ≈ 9 tokenov
```

*(Ilustračné delenie; presný výsledok závisí od konkrétneho tokenizéra.)* Päť pravidiel, ktoré z tohto delenia vyplývajú — a v praxi sa s nimi naozaj narazí:

1. **Medzera patrí k nasledujúcemu slovu.** `"nárok"` a `"␣nárok"` sú **dva rôzne tokeny s rôznymi ID**. Preto sa prompt nikdy nemá končiť medzerou — model potom musí vyrobiť token *bez* úvodnej medzery, čo je vzor, ktorý v tréningu videl zriedka, a kvalita spadne.
2. **Veľké písmeno mení token.** `"Nárok"`, `"nárok"` a `"NÁROK"` sú rôzne tokeny (a `"NÁROK"` sa navyše rozpadne na viac kúskov). Model sa ich ekvivalenciu učí, nedostáva ju zadarmo.
3. **Čísla nie sú čísla.** Novšie tokenizéry delia číslice na skupiny najviac troch (`"2025"` → `"202"` + `"5"`), Llama 2 po jednej. Model teda nevidí hodnotu, ale reťazec kúskov — odtiaľ pramenia chyby v aritmetike a preto sa na počítanie volá kalkulačka ako [nástroj](../05-prakticke/02-agenti-a-nastroje.md).
4. **Diakritika stojí bajty.** `á`, `č`, `ž` majú v UTF-8 **dva bajty**. Ak slovník nemá slovenský merge, rozpadnú sa na bajtové tokeny — v logoch to vidno ako tokeny, ktoré samy o sebe netvoria platný znak.
5. **Kód má vlastné tokeny.** `cl100k_base` a novšie majú samostatné tokeny pre 4, 8, 12… medzier odsadenia — preto je Python v nich výrazne lacnejší než v GPT-2.

#### Najvýraznejší dôsledok: slovenčina spotrebuje viac tokenov

| Jazyk | Tokenov na slovo (rádovo) | Čo to znamená |
|---|---|---|
| angličtina | 1,2 – 1,4 | referencia |
| slovenčina / čeština | 2,0 – 3,0 | **~2× viac tokenov na ten istý obsah** |

Tá istá informácia teda po slovensky:

- **stojí ~2× viac** (platí sa za tokeny na vstupe aj výstupe),
- **zaberie ~2× viac kontextu** (do okna 8k sa zmestí polovica textu oproti angličtine — dôležité pri [chunkovaní pre RAG](06-rag.md)),
- **generuje sa ~2× dlhšie**, lebo každý token je jeden prechod celým modelom ([sekcia 7](#7-ďalší-prechod-autoregresia-a-kv-cache)).

To je aj jeden z dôvodov, prečo malé modely trénované hlavne na angličtine strácajú na slovenčine viac, než by zodpovedalo množstvu dát: kmene slov sú rozsekané na kúsky bez vlastného významu.

**Nikdy neodhadujte počet tokenov od oka** — zmerajte ho tým istým tokenizérom, aký používa model:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
ids = tok.encode("Zamestnanec má nárok na 25 dní dovolenky.")

print(len(ids))                      # koľko tokenov
print(tok.convert_ids_to_tokens(ids))  # ako presne sa text rozdelil
```

#### Špeciálne a chat tokeny

Okrem obsahových tokenov má slovník aj **riadiace tokeny**. Sú to bežné riadky embedding matice a model sa ich význam učí ako pri ktoromkoľvek inom tokene — zvláštne je len to, že **tokenizér ich z používateľského textu nikdy nevyrobí**. Aj keby používateľ napísal doslova `<|eot_id|>`, rozreže sa to na obyčajné znakové tokeny. Práve preto sa hranice medzi rolami nedajú „podstrčiť" v texte.

| Token (Llama 3) | Úloha |
|---|---|
| <code>&lt;&#124;begin_of_text&#124;&gt;</code> | začiatok sekvencie |
| <code>&lt;&#124;start_header_id&#124;&gt;</code> … <code>&lt;&#124;end_header_id&#124;&gt;</code> | obal okolo názvu roly (`system`, `user`, `assistant`) |
| <code>&lt;&#124;eot_id&#124;&gt;</code> | koniec repliky — **na tomto tokene sa generovanie zastaví** |
| <code>&lt;&#124;end_of_text&#124;&gt;</code> | koniec dokumentu (z pretrainingu) |

Konverzácia sa do modelu nikdy nedostane ako štruktúra — **serializuje sa do jedného plochého reťazca tokenov** pomocou *chat šablóny*:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Si asistent pre HR otázky.<|eot_id|><|start_header_id|>user<|end_header_id|>

Koľko mám dní dovolenky?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

Posledný riadok je zámerne otvorený: model dostane sekvenciu končiacu hlavičkou `assistant` a jeho úlohou je pokračovať. Celé „rozprávanie sa" je teda **dopĺňanie textu** — role sú len tokeny. Formát sa medzi modelmi líši, preto sa nikdy nepíše ručne, ale cez `tokenizer.apply_chat_template()` (viď [03-llm-trening.md](03-llm-trening.md#chat-šablóna-a-špeciálne-tokeny)).

#### Cesta späť: detokenizácia a streamovanie

Detokenizácia je jednoduchá — ID sa nahradia reťazcami zo slovníka a zreťazia (`Ġ`/`▁` sa premení späť na medzeru). Má však jeden praktický háčik: pri byte-level BPE môže byť jeden znak rozdelený medzi **dva tokeny** (napr. dva bajty písmena `č`). Dekodér preto musí neúplné bajty **bufferovať**, kým nedostane platný UTF-8.

Preto sa pri streamovaní odpovede občas objaví „prázdny" kus streamu: token prišiel, ale zobraziteľný znak z neho ešte nevznikol. A preto sa aj **stop sekvencie** vyhodnocujú nad detokenizovaným textom, nie nad ID — hľadaný reťazec sa môže rozložiť naprieč tokenmi inak, než čakáte.

---

#### Čo model kvôli tokenom nevidí

Tokenizácia nie je len technický medzikrok — určuje, čo je pre model vôbec **viditeľné**.
Model nikdy nevidí písmená; vidí ID celých kusov textu. Odtiaľ pochádza väčšina „hlúpych"
zlyhaní, ktoré u inak schopného modelu prekvapia:

| Úloha | Prečo zlyháva |
|---|---|
| *„Koľko `r` je v slove strawberry?"* | model vidí `["str", "aw", "berry"]` — tri ID, nie desať písmen. Počítať písmená v niečom, čo nevidí po písmenách, je ako počítať slabiky v telefónnom čísle. |
| *„Napíš to slovo odzadu"* | to isté: poradie znakov vnútri tokenu nie je v reprezentácii priamo prítomné |
| Dlhé násobenie, sčítanie veľkých čísel | čísla sa krájajú na kusy (`1234` môže byť `12`+`34`), takže „jednotky pod jednotky" model nemá z čoho poskladať. Novšie tokenizéry preto číslice delia po jednej. |
| Rýmy a počítanie slabík v slovenčine | tá istá príčina plus málo slovenčiny v dátach |

Dôležité je, že to **nie sú chyby uvažovania, ale chyby vstupu** — a preto sa dajú obísť
troma spôsobmi: dať modelu **nástroj** (kalkulačka, Python — [lekcia 8](../05-prakticke/02-agenti-a-nastroje.md)),
prinútiť ho **rozpísať si to po znakoch** v odpovedi (`s-t-r-a-w-b-e-r-r-y`), čím sa písmená
stanú samostatnými tokenmi, alebo úlohu vôbec nezadávať v takej podobe.

Druhá polovica toho istého obmedzenia je časová: na **jeden token** má model **pevne daný
výpočet** (32 vrstiev, nič viac), takže „ťažšiu" otázku nevie premyslieť dlhšie — iba napísať
viac tokenov ([prečo premýšľanie znamená viac tokenov](#prečo-premýšľanie-znamená-viac-tokenov)).

---

### 2. Z token ID na vektor: embedding matica a pozícia

Prvá vrstva modelu nie je nič inteligentné — je to **tabuľka**. Embedding matica `E` má tvar `[vocab, d_model]`, u referenčného modelu `[128 256, 4096]`, a token ID je **index riadku**:

```text
ID 22801  →  riadok 22801 matice E  →  [0.031, -0.118, 0.442, …, 0.094]   (4096 čísel)
```

Žiadne násobenie, žiadna nelinearita — len skopírovanie riadku. (Matematicky je to násobenie one-hot vektorom `[1, 128 256] × [128 256, 4096]`, ale nikto to tak nepočíta, bolo by to 525 M operácií namiesto jedného `memcpy`.) Vstupom prvej transformer vrstvy je teda matica

```text
n tokenov  →  X = [n, 4096]     riadok = jeden token
```

a **tento tvar sa nezmení až po poslednú vrstvu** (sekcia 4). Vektor v tomto momente ešte nevie nič o kontexte: `"banka"` má rovnaký riadok vo vete o financiách aj o rieke. Kontext doň dostane až attention.

#### Dve osi vstupnej matice: šírka a kontextové okno

Matica `X` má dva rozmery a každý z nich znamená niečo úplne iné. Kto si ich raz oddelí,
prestane mať zmätok v tom, čo je `d_model`, čo je kontextové okno a prečo sa jedno s druhým
nedá vymeniť:

```text
                      ← šírka: d_model = 4096 stĺpcov →       (pevná, daná váhami)
                    ┌───────────────────────────────────┐
   token 1  "Aké"   │  0.03  -0.12   0.44   …    0.09   │   ↑
   token 2  "je"    │ -0.51   0.08  -0.17   …    0.33   │   │  kontext: n riadkov
   token 3  "hlavné"│  0.22   0.61   0.05   …   -0.44   │   │  (iný pri každej požiadavke,
      ⋮             │   ⋮                               │   │   strop = n_ctx = 8192)
   token n  "?"     │  0.14  -0.09   0.28   …    0.71   │   ↓
                    └───────────────────────────────────┘
```

| Os matice `X` | Čo znamená | Čo ju určuje | Mení sa počas behu? |
|---|---|---|---|
| **stĺpce — `d_model`** (*šírka*) | koľko informácie unesie **jeden token** | architektúra; je zapečená vo váhach | **nie** — konštanta modelu |
| **riadky — `n`** (*kontext*) | koľko tokenov model **vidí naraz** | vaša požiadavka | **áno** — pri každom volaní, strop je `n_ctx` |

Z tohto rozdelenia plynú štyri veci:

1. **Kontextové okno nie je „vstupná vrstva".** V žiadnej matici váh sa číslo `n`
   nevyskytuje — váhy majú tvar `[4096, …]`, teda rozmer **jedného tokenu**. Model je funkcia,
   ktorá sa aplikuje buď na každý riadok zvlášť (FFN, normalizácie), alebo na dvojice riadkov
   (attention). Preto nemá zmysel otázka „kde je v modeli uložených 8192 pozícií" — nikde.
   Tie isté váhy spracujú 5 riadkov aj 5000 ([sekcia 8](#8-kontext-krátka-správa-dlhá-správa-a-prečo-má-okno-strop)).
2. **Šírka a okno sú dve nezávislé páky.** `d_model` je vlastnosť *modelu* — zmeniť ju
   znamená natrénovať iný model. `n` je vlastnosť *požiadavky* a jeho strop `n_ctx` nevyplýva
   z tvaru váh, ale z toho, po akú dĺžku bol model trénovaný a koľko pamäte spotrebuje KV
   cache ([prečo sa okno nedá len tak zväčšiť](#prečo-sa-okno-nedá-len-tak-zväčšiť)).
3. **Každá os sa platí inou menou.** Stĺpce stoja **parametre** (`N ≈ 12 · n_layers · d_model²`
   — [sekcia 3](#ako-model-rastie-šírka-vs-hĺbka)), teda VRAM a cenu tréningu. Riadky stoja
   **výpočet a pamäť pri inferencii** (attention je `n²`, KV cache rastie lineárne s `n`).
   Široký model je *drahý model*; dlhý kontext je *drahá otázka*.
4. **Do okna sa počíta aj to, čo model práve píše.** Každý vygenerovaný token pridá do `X`
   ďalší riadok, takže `n_ctx` zdieľa prompt aj odpoveď. Preto pri okne 8192 a prompte s
   7000 tokenov nedostanete odpoveď dlhú 4000 tokenov.

> **Častý omyl:** „model má okno 128k, takže má 128 000 vstupných neurónov". Nemá žiadne.
> Má 4096 stĺpcov a ľubovoľný počet riadkov až po strop — a práve táto vlastnosť odlišuje
> transformer od [feed-forward siete](../02-typy-modelov/04-feed-forward-siete.md) či
> [CNN](../02-typy-modelov/05-konvolucne-siete.md), ktoré majú vstup pevnej veľkosti.

#### Kde je informácia o poradí

Pôvodný transformer (2017) k embeddingu **pripočítal** sínusový pozičný vektor — tak je to rozpísané v [05-embeddings.md](05-embeddings.md#pozičné-kódovanie--aby-model-vedel-poradie). Dnešné LLM (Llama, Mistral, Qwen, Gemma) to robia inak — používajú **RoPE** (*rotary position embedding*):

| | Sínusové PE (2017) | RoPE (dnešné LLM) |
|---|---|---|
| Kedy sa aplikuje | raz, pred prvou vrstvou | **v každej vrstve** |
| Na čo | na vstupný embedding (sčítanie) | na `Q` a `K` (rotácia dvojíc zložiek) |
| Čo kóduje | absolútnu pozíciu | **relatívnu** — skóre `Q·K` závisí len od *rozdielu* pozícií |
| Vstup vrstvy 1 | embedding + pozícia | čistý embedding |

RoPE otočí každú dvojicu zložiek vektora o uhol úmerný pozícii tokenu. Tri dôsledky, ktoré sa ozvú neskôr v tomto dokumente:

- **`K` v KV cache je uložený už otočený** na svojej absolútnej pozícii — preto sa cache nedá „posunúť" a odrezať zo začiatku histórie bez prepočtu ([sekcia 7](#7-ďalší-prechod-autoregresia-a-kv-cache)).
- Rozšírenie okna sa robí **preškálovaním frekvencií** RoPE (`rope_theta`, YaRN) — je to spojitá funkcia pozície, nie tabuľka, takže sa dá „natiahnuť" ([sekcia 8](#prečo-sa-okno-nedá-len-tak-zväčšiť)).
- Ten istý token na inej pozícii má **rovnaký vstupný embedding**, ale iné `Q`/`K` — teda inú rolu v attention.

Zhrnutie celej cesty tvarov, od reťazca po jeden nový token:

```text
"Aké je hlavné mesto Slovenska?"
   │ tokenizér                          CPU, bez váh
   ▼
[128000, 32, 962, 1662, 6156, 22801, 61464, 30]        n = 8 ID
   │ lookup v E [128 256, 4096]         0 FLOPs
   ▼
X = [8, 4096]
   │ 32 × transformer blok (attention + FFN), RoPE vnútri       ~8 mld. váh
   ▼
H = [8, 4096]                                          8 vektorov na výstupe
   │ vyberie sa POSLEDNÝ riadok                        ← sekcia 6
   ▼
h = [4096] → final norm → lm_head [4096, 128 256] → logity [128 256] → 1 token
```

#### Nie každý token pochádza z textu

Rovnaká cesta funguje aj pre obrázky — a práve preto multimodálne modely **nepotrebujú inú architektúru**. Obrázok sa nerozreže tokenizérom, ale na **patche** (napr. 14×14 pixelov), ktoré prejdú vizuálnym enkodérom (ViT) a lineárnou projekciou do toho istého rozmeru `d_model`:

```text
text     "Čo je na obrázku?"  → tokenizér  → ID → lookup v E  ─┐
                                                               ├─► X = [n, 4096] → 32 vrstiev
obrázok  → patche 14×14 → ViT → projekcia ─────────────────────┘
```

Od tohto miesta model nerozlišuje, odkiaľ riadok prišiel — attention mieša obrazové a textové tokeny rovnako. Dve praktické veci z toho: obrázok **stojí tokeny** (podľa rozlíšenia rádovo stovky až tisíce, a pri veľkom rozlíšení sa delí na dlaždice), a preto sa do kontextu zmestí menej textu. Zvuk sa rieši analogicky (rámce namiesto patchov); výstup ostáva textový — generovanie obrázkov beží na inom type modelu.

---

## ČASŤ B — Priechod modelom

> Čo sa s tými vektormi deje vo vnútri: odkiaľ sú ich rozmery, kde sú uložené parametre, ako vyzerá jedna vrstva a čo robí feed-forward časť, ktorá drží väčšinu modelu.

### 3. Odkiaľ sa berie veľkosť vektorov

Vektor tokenu má vo vnútri modelu **stále rovnakú dĺžku `d_model`** — od embedding vrstvy až po poslednú vrstvu. Attention aj feed-forward vrstva ho dočasne premietnu do iného rozmeru, ale na výstupe bloku je vždy zase `d_model`. Práve preto sa dajú bloky ukladať na seba do ľubovoľnej hĺbky.

`d_model` **nie je vypočítané z ničoho** — je to voľba návrhára modelu, hyperparameter. Riadi sa štyrmi pravidlami:

1. **Kapacita.** Väčšie `d_model` = viac miesta na informáciu v jednom tokene. Rastie s veľkosťou modelu: 768 (GPT-2 small) → 4096 (8B) → 8192 (70B).
2. **Deliteľnosť hlavami.** Musí platiť `d_model = n_heads · d_head`, pričom `d_head` býva **64 alebo 128** — empiricky najlepší kompromis medzi „hlava má dosť miesta" a „hláv je dosť veľa".
3. **Hardvér.** Rozmery sú násobky 64/128, aby maticové násobenia sadli na tensor cores GPU. `d_model = 4000` by bežalo citeľne pomalšie ako 4096.
4. **Pomer hĺbka : šírka.** Neplatí „radšej hlbšie" ani „radšej širšie" — *scaling laws* ukazujú, že pre daný počet parametrov existuje optimálny pomer. Pomer `d_model / n_layers` býva rádovo **60–130** (GPT-2 small 64, Llama 3 70B 102, Llama 3 8B 128).

A `d_ff` (šírka feed-forward vrstvy) je tradične **4 × `d_model`**. Pri modernej aktivácii SwiGLU sú v FFN **tri** matice namiesto dvoch, takže sa `d_ff` znižuje na ≈ `8/3 × d_model`, aby počet parametrov ostal rovnaký (Llama 2 7B: 11008 pri `d_model` 4096). Novšie modely idú aj nad toto pravidlo — Llama 3 8B má `d_ff = 14336`, teda zámerne širšie FFN na úkor iných rozmerov.

| Model | `d_model` | `n_layers` | `n_heads` | `d_ff` | `vocab` | parametre | `d_model / n_layers` |
|---|---|---|---|---|---|---|---|
| GPT-2 small | 768 | 12 | 12 | 3072 | 50 257 | 124 M | 64 |
| GPT-2 XL | 1600 | 48 | 25 | 6400 | 50 257 | 1,5 B | 33 |
| GPT-3 | 12288 | 96 | 96 | 49152 | 50 257 | 175 B | 128 |
| Llama 3 8B | 4096 | 32 | 32 | 14336 | 128 256 | 8 B | 128 |
| Llama 3 70B | 8192 | 80 | 64 | 28672 | 128 256 | 70 B | 102 |
| Llama 3.1 405B | 16384 | 126 | 128 | 53248 | 128 256 | 405 B | 130 |

Z tabuľky vystupuje jedno prekvapivé číslo. Medzi Llama 3 8B a Llama 3.1 405B je **50× viac
parametrov**, ale vrstiev pribudlo len **3,9×** (32 → 126). Väčšina rastu išla do **šírky**:
`d_model` 4×, `d_ff` 3,7×, `n_heads` 4×. Veľký model teda nie je „ten istý model, len oveľa
hlbší" — je to predovšetkým **oveľa širší** model, ktorý je *zároveň* o kus hlbší. Pomer
`d_model / n_layers` (*aspect ratio*) pritom ostáva v pásme ~100–130, čo nie je náhoda
(vysvetlenie je v [Ako model rastie: šírka vs. hĺbka](#ako-model-rastie-šírka-vs-hĺbka)).

Ešte výraznejšie to vidno pri **MoE** modeloch, kde šírka „odteká" do paralelných expertov
(mechanizmus je v [sekcii 5](#keď-je-ffn-priveľká-mixture-of-experts-moe), tu sú len ich rozmery):

| Model (MoE) | `d_model` | `n_layers` | `d_ff` jedného experta | expertov (aktívnych) | parametre (aktívne) |
|---|---|---|---|---|---|
| Mixtral 8×7B | 4096 | 32 | 14336 | 8 (2) | 46,7 B (12,9 B) |
| Qwen3 235B-A22B | 4096 | 94 | 1536 | 128 (8) | 235 B (22 B) |
| DeepSeek V3 | 7168 | 61 | 2048 | 256 smerovaných + 1 zdieľaný (8) | 671 B (37 B) |

Všimnite si Qwen3 235B: má **rovnaké `d_model` 4096 ako 8B model** a jeho FFN sú dokonca
9× užšie než v Llame 3 8B. Tridsaťkrát viac parametrov nie je ani v hĺbke, ani v šírke jednej
matice — je **vedľa seba**, v 128 nezávislých expertoch.

#### Rozloženie parametrov v modeli

Spočítajme referenčný model — je to obyčajné sčítanie veľkostí matíc a hneď z neho vidieť, kde sa kapacita modelu spotrebúva:

```text
JEDNA VRSTVA
  attention:  W_Q 4096×4096 = 16.8 M
              W_K 4096×1024 =  4.2 M     (len 8 KV hláv × 128 = 1024)
              W_V 4096×1024 =  4.2 M
              W_O 4096×4096 = 16.8 M
                              ────────
                               42.0 M
  feed-forward (SwiGLU = 3 matice):
              3 × 4096×14336 = 176.2 M
                              ────────
  spolu jedna vrstva          218.2 M   →  × 32 vrstiev = 6.98 B

EMBEDDINGY
  vstupná embedding matica  128 256×4096 = 525 M
  výstupná (lm_head)        128 256×4096 = 525 M
                                          ──────
CELKOM                                    ≈ 8.03 B parametrov
```

Dve veci, ktoré z toho stoja za zapamätanie:

- **Feed-forward vrstvy zaberajú ~80 % parametrov jednej vrstvy** (176 M z 218 M) a **~70 % celého modelu** (zvyšok pripadá na embeddingy) — nie attention. Attention je koncepčne zaujímavejšia, objemom však dominuje FFN.
- **Embedding matice sú netriviálne** — pri malých modeloch dokonca dominujú (GPT-2 small: 39 M zo 124 M). Preto sa často **zdieľajú** (*weight tying*): tá istá matica sa použije na vstupe aj na výstupe.
- **Toto je rozpočet *dense* modelu**, kde každý token prejde cez všetky váhy. Modely typu **MoE** ho lámu na polovicu: parametrov majú násobne viac, ale na jeden token ich použijú len zlomok — [sekcia 5](#keď-je-ffn-priveľká-mixture-of-experts-moe).

#### Ako model rastie: šírka vs. hĺbka

Z rozpočtu vyššie sa dá vytiahnuť jednoduchý vzorec. Na jednu vrstvu pripadá ~4 `d_model²`
v attention (pri GQA menej) a ~8 `d_model²` vo feed-forward, teda dokopy zhruba:

```text
N ≈ 12 · n_layers · d_model²          (parametre dense modelu, bez embeddingov)
```

Overenie na dvoch modeloch z tabuľky:

- Llama 3.1 405B: `12 · 126 · 16384² = 406 mld.` ✓
- Llama 3 8B: `12 · 32 · 4096² = 6,4 mld.` + 1,05 mld. embeddingov ≈ 7,5 mld. (skutočnosť
  8,0 mld. — model má FFN mierne širšie, než pravidlo predpokladá)

A práve tu je celá pointa: **`n_layers` vystupuje vo vzorci lineárne, `d_model` kvadraticky.**
Z toho plynie asymetria, ktorá rozhoduje o tvare veľkých modelov:

| Zdvojnásobím… | parametre | FLOPs na token | **sériových krokov na token** | veľkosť jednej matice | ako to rozdelím na 8 GPU |
|---|---|---|---|---|---|
| `n_layers` (hĺbka) | 2× | 2× | **2×** | rovnaká | pipeline — GPU sa striedajú |
| `d_model` (šírka) | **4×** | 4× | **rovnako** | 4× | tensor parallel — GPU pracujú naraz |

Zdvojnásobenie šírky teda dá **štyrikrát viac parametrov za rovnaký počet sériových krokov**,
zatiaľ čo zdvojnásobenie hĺbky dá dvakrát viac parametrov a *dvakrát predĺži reťaz*, ktorou
musí prejsť každý jeden token. Pri rovnakom počte parametrov je preto širší a plytší model
**lepšie paralelizovateľný** — a to je dôvod, prečo 405B model má 126 vrstiev a nie 500.

#### Prečo sa oplatí ísť skôr do šírky: problém s paralelizáciou

Porovnajme dva hypotetické modely s **rovnakým počtom parametrov** (25,8 mld.):

| | **A — široký a plytký** | **B — úzky a hlboký** |
|---|---|---|
| `n_layers` | 32 | 128 |
| `d_model` | 8192 | 4096 |
| parametre (`12·n·d²`) | 25,8 B | 25,8 B |
| FLOPs na token | rovnaké | rovnaké |
| bajtov váh na token (decode) | rovnako | rovnako |
| **sériových krokov na token** | **32** | **128** |
| kolektívnych synchronizácií na token pri tensor parallel na 8 GPU | **64** | **256** |
| veľkosť typického maticového násobenia | 8192 × 8192 | 4096 × 4096 |

Na papieri sú rovnaké. V reálnom nasadení je A citeľne rýchlejší, a to z troch nezávislých
dôvodov.

**1. Šírka sa paralelizuje *vnútri* vrstvy (tensor parallelism).** Široké matice sa dajú
rozrezať po stĺpcoch a rozdať GPU: každá si spočíta svoj kus `W_Q`, svoje hlavy, svoj výsek
FFN — všetky naraz. Zladiť sa treba len dvakrát za vrstvu (po attention a po FFN), a to
sčítaním jedného vektora dĺžky `d_model`.

```text
ŠÍRKA — tensor parallelism (vrstva rozrezaná pozdĺžne)
   GPU0 │ GPU1 │ GPU2 │ GPU3      všetky štyri počítajú súčasne
   ─────┴──────┴──────┴──────
              ↓ all-reduce (vektor d_model)   ← 1 synchronizácia

HĹBKA — pipeline parallelism (vrstvy rozdelené medzi GPU)
   GPU0: vrstvy 1–32  ──►  GPU1: 33–64  ──►  GPU2: 65–96  ──►  GPU3: 97–128
         počíta              čaká             čaká             čaká
```

Dôležitý pomer: **výpočet vo vrstve rastie s `d_model²`, ale komunikácia len s `d_model`.**
Dvojnásobná šírka znamená dvakrát viac výpočtu na každý prenesený bajt — širší model teda
GPU klaster využíva **lepšie**. Šírku si viete „kúpiť" ďalšími GPU.

**2. Hĺbka sa vnútri jedného tokenu paralelizovať *nedá*.** Vrstva 33 potrebuje výstup
vrstvy 32 — je to reťaz, nie množina nezávislých úloh. Keď vrstvy rozdelíte medzi GPU
(*pipeline parallelism*), GPU na seba čakajú. Podiel času, keď GPU stojí naprázdno, je pri
`P` stupňoch a `M` mikro-dávkach:

```
bublina = (P − 1) / (M + P − 1)
```

- Pri **tréningu** sa dá bublina prekryť: pustíte veľa mikro-dávok naraz (`M = 64`,
  `P = 8` → bublina ~10 %).
- Pri **generovaní** ale beží token po tokene, takže `M = 1` a bublina je `(P−1)/P` —
  na 8 GPU **87 % času nečinnosti**. Preto sa pipeline pri interaktívnej inferencii
  prakticky nepoužíva a hĺbka sa premieta **priamo do latencie** odpovede.

**3. Každá vrstva má fixnú réžiu, ktorá sa platí za každý token.** Spustenie kernelov,
načítanie váh z HBM, dve kolektívne synchronizácie pri tensor parallel — každá s vlastnou
latenciou. Model B ich má 256 na token, model A 64. Táto réžia sa **nedá kúpiť ďalšími GPU**:
pridaním GPU sa jedna vrstva zrýchli, ale počet vrstiev ostane. Navyše menšie matice horšie
sýtia tensor cores — násobenie `4096 × 4096` využije GPU slabšie než `8192 × 8192`.

**A ešte tréning.** Hlboký model má dlhšiu reťaz, ktorou musí prejsť gradient — viac
sériových krokov backpropu, viac príležitostí na miznúci či explodujúci gradient a citlivejšie
ladenie (viď [02-problemy-pri-uceni.md](../03-ucenie/02-problemy-pri-uceni.md)). Široký model
robí to isté ako jedno väčšie maticové násobenie — presne to, v čom je GPU najsilnejšia.

**MoE je táto logika dotiahnutá do konca.** Experti sú **nezávislé** matice vedľa seba, takže
sa dajú rozložiť na rôzne GPU (*expert parallelism*) a počítať súčasne; komunikuje sa len
smerovanie tokenov, nie váhy. Preto má Qwen3 235B `d_model` stále 4096 a nafukuje sa „do
strán" na 128 expertov namiesto toho, aby išiel na 300 vrstiev.

**Prečo teda nie model s 8 vrstvami a `d_model = 100 000`?** Lebo hĺbka nie je len spôsob,
ako pridať parametre — je to **počet za sebou idúcich krokov spracovania**. Otázka typu
„v ktorom meste sa narodil autor knihy X" vyžaduje najprv nájsť autora a až potom jeho mesto;
jedna vrstva na to nestačí a niekoľko vrstiev musí ísť *po sebe*. Extrémne plytký model tieto
zložené úlohy nezvládne, nech je akokoľvek široký — deľba práce medzi vrstvami je rozpísaná
v [Čo robia jednotlivé vrstvy](#čo-robia-jednotlivé-vrstvy). Okrem toho `d_head` nad 128 už kvalitu
nedvíha a embedding s `lm_head` rastú s `d_model` lineárne, takže v plytkom širokom modeli
by spotrebovali neúmernú časť rozpočtu.

Empiricky sa preto kvalita drží skoro rovnaká v pomerne širokom pásme `d_model / n_layers`
≈ 60–130 a mimo neho klesá. **Praktické pravidlo:** vyberte pomer z tohto pásma — a keď sa
v ňom dá voliť, choďte do **šírky**, lebo tú vám paralelizmus zaplatí, kým hĺbku platíte
latenciou pri každom jednom tokene.

> **Ako sa to prejaví na vás:** ak si vyberáte medzi dvoma modelmi rovnakej veľkosti, ten
> plytší a širší bude generovať rýchlejšie (nižšia latencia na token) a ľahšie sa rozloží na
> viac GPU. Pri lokálnom behu na jednej karte je rozdiel malý — tam rozhoduje hlavne to, či
> sa váhy vojdú do VRAM.

---

### 4. Cesta jedného vektora jednou vrstvou

Celá vrstva na jednom obrázku — vľavo attention s maticou váh a kauzálnou maskou, vpravo dole tri nezávislé feed-forward dráhy:

![Detail jednej transformer vrstvy v pre-norm podobe: tri tokeny ako stĺpce, najprv RMSNorm, potom projekcie Q, K a V; z nich sa spočíta matica attention váh 3×3 s kauzálnou maskou, výstup tokenu je vážená zmes Value vektorov; nasleduje pripočítanie rezídua, druhá RMSNorm a feed-forward sieť, ktorá spracúva každý token samostatne rovnakými váhami (4096 → 14336 → 4096), medzi stĺpcami nič netečie; na záver opäť reziduum a výstup do ďalšej z 32 vrstiev](../../images/transformer-vrstva-detail.svg)

Označme `n` počet tokenov na vstupe. Vstup do vrstvy je matica `X` tvaru `[n, 4096]` — jeden riadok = jeden token. Tá istá cesta zapísaná v tvaroch tenzorov:

```text
X  [n, 4096]
│
├─► RMSNorm                      [n, 4096]        ← pre-norm (viď rámček nižšie)
│     │
│     ├─► Q = X·W_Q  →  [n, 4096] → reshape → [n, 32, 128] → transpose → [32, n, 128]
│     ├─► K = X·W_K  →  [n, 1024] → reshape → [n,  8, 128] → transpose → [ 8, n, 128]
│     ├─► V = X·W_V  →  [n, 1024] → reshape → [n,  8, 128] → transpose → [ 8, n, 128]
│     │      (hlavy NIE sú samostatné matice — je to jedna projekcia a potom reshape)
│     │
│     ├─► GQA: každá K/V hlava obslúži 4 Q hlavy → K, V sa zopakujú 4× → [32, n, 128]
│     │
│     ├─► skóre = Q·Kᵀ / √128  →  [32, n, n]   ⚠ TU je tá kvadratika
│     ├─► + kauzálna maska (horný trojuholník = −∞)
│     ├─► softmax po riadkoch   →  [32, n, n]
│     ├─► výstup = váhy·V       →  [32, n, 128] → transpose+reshape → [n, 4096]
│     └─► · W_O                 →  [n, 4096]
│
├──(+)  reziduálne spojenie: X = X + attention_out
│
├─► RMSNorm                      [n, 4096]
│     └─► FEED-FORWARD (každý riadok zvlášť) → [n, 4096]
│
└──(+)  X = X + ffn_out          →  [n, 4096]   ← vstup ďalšej vrstvy
```

> **Pre-norm vs. post-norm — v čom sa dnešné modely líšia od článku z 2017.** Pôvodný transformer normalizoval **až po** bloku: `x = Norm(x + attention(x))` (*post-norm*). Dnešné LLM normalizujú **pred** blokom a rezíduum normalizáciu obchádza: `x = x + attention(Norm(x))` (*pre-norm*) — hlboké modely sa tak trénujú stabilnejšie, lebo cez reziduálnu vetvu tečie gradient nedotknutý. Preto má pre-norm model na konci ešte jednu **finálnu normalizáciu** pred `lm_head` ([sekcia 6](#6-ako-vznikne-výstupný-token--a-prečo-len-jeden)). Tento dokument aj obrázok používajú pre-norm; schéma v [lekcii 4](01-transformer-siete.md#celková-architektúra) a ručný prepočet v [lekcii 6](05-embeddings.md#krok-3-transformer-vrstvy--tu-sa-deje-pochopenie-kontextu) ukazujú pôvodný post-norm. **Pri vlastnej implementácii si vyberte jedno a držte sa toho.**

Kľúčové pozorovanie, ktoré vysvetľuje celý zvyšok dokumentu:

> **Attention je jediné miesto, kde si tokeny navzájom vymieňajú informáciu.** Všetko ostatné vo vrstve — normalizácia, feed-forward, reziduá — beží nad **každým tokenom nezávisle**. Attention = komunikácia medzi tokenmi, FFN = spracovanie vnútri tokenu.

Odtiaľ plynie aj rozdelenie nákladov: attention rastie s **druhou mocninou** počtu tokenov (matica `n × n`), FFN len **lineárne** (n riadkov, každý rovnako drahý).

#### Reziduálny prúd: vrstvy nie sú séria, ale zbernica

Zápis `X = X + attention_out` a `X = X + ffn_out` vyzerá ako technický detail proti miznúcim gradientom ([03-ucenie](../03-ucenie/02-problemy-pri-uceni.md)). V skutočnosti určuje, **ako je celý model organizovaný**.

Všimnite si, že `X` sa nikdy neprepíše — vždy sa k nemu len **pripočíta**. Vektor `[4096]` každého tokenu je teda **zbernica** (*residual stream*), ktorá tečie zdola nahor cez všetkých 32 vrstiev, a každý blok z nej **číta** (cez normalizáciu) a **pripíše** do nej svoj príspevok:

```text
       vrstva 1        vrstva 2                       vrstva 32
  x₀ ──────●───────────────●──────── … ──────────────────●────────► lm_head
           ▲               ▲                             ▲
           │ +attn +ffn    │ +attn +ffn                  │ +attn +ffn
        (číta x₀)       (číta x₁)                    (číta x₃₁)
```

Tri dôsledky, ktoré inak vyzerajú ako záhady:

- **Preto je `d_model` konštantné.** Nie je to estetika — ak má každá vrstva pripočítavať do tej istej zbernice, musí mať zbernica stále rovnakú šírku.
- **Preto model neprestane fungovať, keď mu vrstvu vyberiete.** Vrstvy nie sú reťaz, kde prerušenie znamená koniec; sú to prírastky. Odstránenie jednej z 32 kvalitu zhorší, ale model ďalej funguje — na tom stojí *layer pruning* aj *early exit*.
- **Preto sa dá „nazrieť" doprostred modelu.** Ak zoberiete `X` po 12. vrstve a pustíte ho rovno cez `lm_head`, dostanete zmysluplnú (len horšiu) predpoveď — technika známa ako *logit lens*. Predpoveď sa v zbernici postupne „vyostruje".

#### Čo robia jednotlivé vrstvy

Interpretačné práce naznačujú hrubú deľbu práce, ktorá sa opakuje naprieč modelmi:

| Vrstvy | Čo sa v nich hlavne deje |
|---|---|
| **rané** (1–8) | poskladanie tokenov na slová, slovné druhy, lokálna syntax — zbernica ešte nesie skôr „aké písmená" než „aký význam" |
| **stredné** (9–24) | význam v kontexte, rozlíšenie entít, **faktické vybavovanie** („hlavné mesto Slovenska" → Bratislava); sem cielia aj metódy editácie znalostí |
| **neskoré** (25–32) | prevod zámeru na konkrétny **ďalší token** — formát, gramatický tvar, interpunkcia |

Toto je hrubá mapa, nie ostrý predel; presné roly sa medzi modelmi líšia a jedna vrstva robí viac vecí naraz. Užitočné je z toho jedno: **hĺbka nie je opakovanie toho istého**, preto sa `n_layers` nedá voľne vymeniť za `d_model` — hoci [paralelizácia](#prečo-sa-oplatí-ísť-skôr-do-šírky-problém-s-paralelizáciou) by si to priala.

#### Čo presne robí RMSNorm

Normalizácia drží mierku čísel v zbernici pod kontrolou — bez nej by sa po 32 pripočítaniach hodnoty rozišli do extrémov a softmax v attention by sa nasýtil.

```text
LayerNorm (2017):   y = γ ⊙ (x − μ) / σ  + β        (μ, σ cez 4096 zložiek jedného tokenu)
RMSNorm (dnes):     y = γ ⊙  x / √(mean(x²) + ε)    (bez odčítania priemeru, bez posunu β)
```

RMSNorm teda len **preškáluje dĺžku vektora** na rozumnú mierku a naučeným `γ` (4096 čísel na vrstvu) dovolí modelu niektoré zložky zvýrazniť. Vypustenie `μ` a `β` kvalitu nezhorší a ušetrí réžiu — preto ho používa Llama, Mistral, Qwen aj Gemma.

Kľúčové pre pochopenie zvyšku dokumentu: **normalizácia beží cez 4096 zložiek jedného tokenu, nie naprieč tokenmi.** Neporušuje teda pravidlo, že tokeny sa miešajú iba v attention.

---

### 5. Feed-forward vrstva: každý vektor sám za seba

Po attention má každý token vektor obohatený o kontext. Feed-forward vrstva je obyčajný dvojvrstvový [MLP](../02-typy-modelov/04-feed-forward-siete.md), ktorý sa naň aplikuje — a **na každý token sa aplikujú tie isté váhy**:

```text
klasicky (GPT-2 štýl, tu v našich rozmeroch — samotné GPT-2 small má 768 → 3072 → 768):
                    FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2
                             4096 → 16384 → 4096

moderne (SwiGLU):   FFN(x) = W_down · ( SiLU(W_gate · x) ⊙ (W_up · x) )
                             4096 → 14336 (dve vetvy, násobené po zložkách) → 4096
```

Tri veci, ktoré tu študenti najčastejšie prehliadnu:

**a) Je to naozaj token po tokene.** Na [obrázku vyššie](#4-cesta-jedného-vektora-jednou-vrstvou) sú to tri oddelené dráhy s ✕ medzi nimi. Pri `n = 1000` tokenoch sa tá istá matica `W_1` použije 1000-krát na 1000 rôznych vektorov. Nič sa medzi tokenmi nemieša. Implementačne sa to urobí jedným maticovým násobením `[1000, 4096] × [4096, 14336]`, ale sémanticky sú to 1000 nezávislých priechodov. Preto sa FFN dá triviálne paralelizovať a preto je vo fáze generovania (jeden token) výpočtovo veľmi lacná.

**b) Rozšírenie a zúženie má zmysel.** Vrstva najprv vektor **rozšíri** (4096 → 14336), pustí cez nelinearitu a potom **zúži** späť. Bez rozšírenia by nelinearita mala málo priestoru; bez zúženia by sa rozmer po každej vrstve zväčšoval a bloky by sa nedali skladať.

**c) Toto je pamäť modelu na fakty.** Interpretačné práce ukazujú, že FFN sa dá čítať ako **key-value pamäť**: prvá matica rozhodne „na čo tento vektor vyzerá", nelinearita to prahuje, druhá matica pripočíta zodpovedajúcu informáciu. Faktické znalosti („Bratislava je hlavné mesto Slovenska") sú uložené hlavne tu — čo je konzistentné s tým, že FFN drží 80 % parametrov vrstvy.

> Ručne prepočítaný priechod attention + reziduum + LayerNorm + FFN na 4-rozmerných vektoroch je v [05-embeddings.md, Krok 3](05-embeddings.md#krok-3-transformer-vrstvy--tu-sa-deje-pochopenie-kontextu).

#### Keď je FFN priveľká: Mixture of Experts (MoE)

Z výpočtu v [sekcii 3](#rozloženie-parametrov-v-modeli) vyplýva nepríjemná vec: ak chcem viac znalostí, musím zväčšiť FFN — a tým **každý token predražiť**, lebo pri každom prechode sa prečítajú všetky váhy. **MoE túto väzbu pretína.**

Namiesto jednej FFN má vrstva `E` samostatných FFN (*expertov*) a malý **router** — jedna lineárna vrstva `[4096, E]`, ktorá pre každý token vyberie `k` najvhodnejších (typicky `k = 2` z 8, alebo 8 z 256):

```text
                          ┌─► expert 1   ✗ nepočíta sa
   token ──► router ──────┼─► expert 2   ✓ váha 0.7
   [4096]   [4096, E]     ├─► expert 3   ✗
                          ├─► expert 4   ✓ váha 0.3      → výstup = 0.7·E₂(x) + 0.3·E₄(x)
                          └─► …  expert E ✗
```

Router rozhoduje **pre každý token a každú vrstvu zvlášť** — dva susedné tokeny tej istej vety môžu ísť cez úplne iných expertov. Attention ostáva hustá (*dense*); nahrádza sa len FFN, teda práve tých ~70 % parametrov.

Tým sa počet parametrov rozdelí na dve úplne rozdielne čísla:

| | Čo to je | Čo z toho platíte |
|---|---|---|
| **celkové parametre** | všetky váhy všetkých expertov | **VRAM** — v pamäti musia byť všetci, router ich vyberá až za behu |
| **aktívne parametre** | tie, cez ktoré prejde jeden token | **výpočet a rýchlosť** — koľko sa reálne násobí a číta |

| Model | Celkové | Aktívne | Experti |
|---|---|---|---|
| Mixtral 8×7B | 46,7 B | ~12,9 B | 8, top-2 |
| Qwen3 235B-A22B | 235 B | 22 B | 128, top-8 |
| DeepSeek V3 | 671 B | 37 B | 256 smerovaných + 1 zdieľaný, top-8 |

Preto sa MoE oplatí **poskytovateľom služieb**, nie pri lokálnom behu na notebooku: DeepSeek V3 má kvalitu veľkého modelu za cenu výpočtu ~37B modelu, ale do pamäte potrebuje celých 671 miliárd váh. Pre lokálne nasadenie je [8B dense model](#3-odkiaľ-sa-berie-veľkosť-vektorov) často praktickejší než MoE s rovnakým počtom aktívnych parametrov.

Dve veci, ktoré s MoE prichádzajú v balíku: router sa musí trénovať s **vyvažovacou stratou** (inak by väčšina tokenov smerovala k niekoľkým expertom a ostatní by sa nenaučili nič), a pri dávkovaní sa tokeny jednej dávky **rozptýlia medzi rôznych expertov**, čo komplikuje efektívnu inferenciu.

> Keď teda v karte modelu (*model card*) uvidíte zápis typu **`235B-A22B`**, čítajte ho ako „235 miliárd v pamäti, 22 miliárd na token".

---

## ČASŤ C — Generovanie, kontext a limity

> Ako z posledného vektora vznikne token, slovo a veta, čo sa pri ďalšom prechode ukladá do cache a prečo má kontextové okno strop.

### 6. Ako vznikne výstupný token — a prečo len jeden

Toto je miesto, kde sa najčastejšie stráca intuícia: do modelu vojde napríklad 2000 tokenov, prejdú 32 vrstvami — a von vyjde **jediné slovo**. Čo sa stalo so zvyšnými 1999 výstupmi?

#### Jeden prechod = n vstupov, n výstupov, 1 použitý

Nikam sa nestratili. Model je funkcia, ktorá zobrazuje `[n, 4096]` na `[n, 4096]`: **koľko tokenov vojde, toľko vektorov vyjde**. Pri inferencii sa z nich ale použije len posledný:

```text
vstup: n = 8 tokenov                        výstup 32. vrstvy: H = [8, 4096]

 1 <|begin_of_text|> ─┐                     h₁  ──► predpovedal by token č. 2   ✗ zahodí sa
 2 A                  │                     h₂  ──► predpovedal by token č. 3   ✗ zahodí sa
 3 ké                 │                     h₃  ──► …                           ✗
 4 ␣je                │   32 transformer    h₄  ──► …                           ✗
 5 ␣hlavné            ├──► blokov      ──►  h₅  ──► …                           ✗
 6 ␣mesto             │                     h₆  ──► …                           ✗
 7 ␣Slovenska         │                     h₇  ──► …                           ✗
 8 ?                 ─┘                     h₈  ──► predpovedá token č. 9       ✓ POUŽIJE SA
                                                    │
                                    final norm → lm_head → logity → 1 token
```

Prečo je to tak? Kvôli **kauzálnej maske**. Vektor `hᵢ` videl tokeny 1…`i` a predpovedá token na pozícii `i+1`. Lenže tokeny 2…8 **už poznáme** — sú to tokeny promptu. Predpoveď `h₁` („čo nasleduje po `<|begin_of_text|>`") je zaujímavá pri tréningu, pri generovaní je zbytočná. Jediná pozícia, ktorá predpovedá niečo **nové**, je tá posledná.

Z toho plynú dve veci, ktoré stoja za zapamätanie:

- **Pri tréningu sa použijú všetky.** Jeden prechod nad 2000-tokenovým blokom textu dá 2000 trénovacích príkladov naraz (každá pozícia predpovedá tú nasledujúcu) — presne preto je pretraining výpočtovo únosný. Detaily v [03-llm-trening.md](03-llm-trening.md).
- **Pri inferencii sa `lm_head` počíta len pre posledný riadok.** Inferenčné servery (vLLM, TGI) preto pri prefille hidden stavy `h₁…h₇` **vôbec nepustia cez `lm_head`** — ušetria `(n−1) × 525 M` operácií. Naivná implementácia v čistom PyTorchi to nerobí a zbytočne počíta logity pre celý prompt.


#### Prečo nie viac tokenov naraz

Námietka sa ponúka sama: keby model vyrobil päť tokenov na jeden prechod, generoval by päťkrát rýchlejšie. Prečo to teda nerobí?

**Dôvod nie je výpočtový, ale pravdepodobnostný.** Model je natrénovaný modelovať rozdelenie

```text
P(token₁, token₂, …, tokenₘ)  =  P(t₁) · P(t₂ | t₁) · P(t₃ | t₁,t₂) · … · P(tₘ | t₁…tₘ₋₁)
```

Každý činiteľ je **podmienený tým, čo bolo vybrané pred ním** — a práve túto podmienenosť počíta jeden prechod modelom. Výstup na poslednej pozícii je rozdelenie pre **nasledujúci** token, nie pre dva. Aby model vedel povedať niečo o tokene `n+2`, musí najprv vedieť, ktorý token `n+1` bol skutočne vybraný; ten sa totiž stáva vstupom ďalšieho prechodu.

Prečo na tom záleží, vidno na slovenčine, kde jedno slovo býva viac tokenov:

```text
kontext: „Zamestnanec má nárok na 25 dní dovolen"

ak sa vyberie   "ky"   → pokračovanie musí byť „…dovolenky." (nominatív/akuzatív)
ak sa vyberie   "ku"   → pokračovanie musí byť „…dovolenku." (iný tvar, iná väzba)
```

Predpoveď pre druhú pozíciu je v oboch prípadoch iná. Keby model vyrobil obe naraz z toho istého vektora, musel by o druhom tokene rozhodnúť **bez toho, aby vedel, ako dopadol prvý** — dostal by priemer cez všetky možnosti. Pri jednom kroku by sa to ešte dalo uniesť, pri piatich vznikne nesúvislý text — neistota sa s každou ďalšou pozíciou násobí.

Architektonicky je to tá istá vec z druhej strany: na poslednej pozícii existuje **jeden vektor** `h`, a `lm_head` z neho urobí **jedno** rozdelenie. Druhé rozdelenie by vyžadovalo druhú hlavu, natrénovanú na úlohu „token o dva ďalej" — a tá je principiálne neistejšia, lebo nevie, čo padne medzitým.

**Čiastočne sa to obísť dá — a v praxi sa to robí.** Len nie tak, že by sa presnosť obetovala:

| Prístup | Ako obchádza sekvenčnosť | Cena |
|---|---|---|
| **Špekulatívne dekódovanie** ([sekcia 7](#špekulatívne-dekódovanie-ako-obísť-memory-bound-strop)) | návrh `k` tokenov niečím lacným + overenie jedným prechodom veľkého modelu | žiadna — výstupné rozdelenie ostáva **presne rovnaké** |
| **Multi-token prediction** (Medusa, EAGLE, DeepSeek V3) | prídavné hlavy predpovedajú `n+2`, `n+3`… naraz | samostatne nepresné, preto slúžia len ako **návrh na overenie** |
| **Difúzne jazykové modely** | celá odpoveď sa generuje paralelne a iteratívne sa „vyostruje" | iná architektúra aj tréning; zatiaľ výskumný smer |

Prvé dva riadky hovoria to isté: viac tokenov za prechod sa dá dostať, ale len ako **návrh, ktorý musí model potvrdiť**. Autoregresívna závislosť sa nedá zrušiť, dá sa len uhádnuť dopredu a overiť.

> A všimnite si, že jeden prípad paralelného spracovania v modeli už máte — **prefill**. Tam prejde tisíc tokenov naraz práve preto, že sú **známe**; nič sa nehádalo. Generovanie je pomalé nie kvôli architektúre, ale preto, že vstup ďalšieho kroku ešte neexistuje.

#### Z posledného vektora na token

```text
h  [4096]                      posledný vektor poslednej vrstvy
│
├─► final RMSNorm              [4096]
│
├─► lm_head:  h · W_U          W_U má tvar [4096, 128 256]
│                              ↓
│   logity  [128 256]          jedno reálne číslo pre KAŽDÝ token slovníka
│                              napr.  "␣Bratislava" 18.3
│                                     "␣Praha"       9.1
│                                     "␣hlavné"      7.4  …
│
├─► (voliteľne) úpravy logitov: teplota, repetition penalty, top-k / top-p, zákaz tokenov
│
├─► softmax                    [128 256] pravdepodobností so súčtom 1
│                              "␣Bratislava" 0.93, "␣Praha" 0.01, …
│
└─► výber jedného tokenu       → id 43217 → detokenizácia → "␣Bratislava"
```

Poznámky, ktoré sa oplatí vedieť:

- **Výstupná matica je obrovská.** Jedno násobenie `4096 × 128 256` = 525 M operácií len na to, aby vznikol jeden token. Pri veľkých slovníkoch je to citeľná časť nákladov na generovanie.
- **Logity nie sú pravdepodobnosti.** Sú to ľubovoľné reálne čísla; pravdepodobnosti z nich robí až softmax. Všetky triky s dekódovaním (teplota, top-p) sa robia **na logitoch alebo tesne po softmaxe** — nie v modeli. Model je deterministický; náhoda je až vo výbere.
- **Kandidátov je vždy všetkých 128 256** — vrátane špeciálnych a chat tokenov. Preto sa dá generovanie obmedziť jednoducho tým, že sa nechceným tokenom nastaví logit na `−∞` (tak funguje *structured output* / JSON mód: pri každom kroku sa povolia len tokeny, ktoré neporušia gramatiku).
- **Sampling** (greedy / teplota / top-p / top-k) je rozpísaný v [predchádzajúcom dokumente](01-transformer-siete.md#ako-presne-sa-vyberá-ďalší-token-dekódovanie).

#### Ako z tokenov vznikne slovo a veta

Slovo často **nie je** jeden token — takže ani nevzniká naraz. Vzniká ako niekoľko nezávislých prechodov modelom, medzi ktorými sa jediné, čo „prežije", je predĺžená sekvencia tokenov (a KV cache):

```text
prompt:  "Zamestnanec má nárok na"          ← n = 6 tokenov

prechod 1 →  "␣25"        sekvencia: … nárok na 25                 (n = 7)
prechod 2 →  "␣dní"       sekvencia: … na 25 dní                   (n = 8)
prechod 3 →  "␣dovolen"   sekvencia: … 25 dní dovolen              (n = 9)
prechod 4 →  "ky"         sekvencia: … dní dovolenky               (n = 10)
prechod 5 →  "."          sekvencia: … dovolenky.                  (n = 11)
prechod 6 →  <|eot_id|>   → STOP                                   (n = 12)
```

Slovo *„dovolenky"* teda nikde v modeli neexistuje ako celok — je to výsledok dvoch samostatných prechodov 8 miliardami váh. Model si medzi prechodmi **nepamätá žiadny zámer**: keď v prechode 4 vyberá `"ky"`, jedinou informáciou o tom, že práve píše slovo *dovolenky*, je token `"␣dovolen"` v sekvencii. Celý „stav" konverzácie je vždy len text (resp. tokeny) — model je bezstavový.

Preto tiež platí, že **dlhšia odpoveď = viac prechodov = lineárne viac času a peňazí**, a preto sa slovenská odpoveď generuje pomalšie než anglická rovnakého obsahu ([sekcia 1](#1-tokenizácia-ako-sa-z-textu-stanú-čísla)).

#### Kedy sa generovanie zastaví

Model sám od seba neprestane — cyklus beží, kým ho niečo nezastaví:

| Čo generovanie zastaví | Ako funguje | Kde sa vyhodnocuje |
|---|---|---|
| **EOS token** (<code>&lt;&#124;eot_id&#124;&gt;</code>, `</s>`) | model ho **vybral samplovaním** ako hociktorý iný token; je to bežný riadok logitov, ktorého pravdepodobnosť sa naučil v SFT | v samplovacej slučke |
| `max_new_tokens` | tvrdý strop počtu prechodov | server / knižnica |
| `stop` sekvencie | hľadanie reťazca v detokenizovanom výstupe | server, nad textom |
| kontextové okno | `n` dosiahlo `n_ctx` → nemá kam pridať ďalší token | server (typicky chyba alebo orezanie) |
| zrušenie požiadavky | klient zavrel stream | server |

Že model vie prestať, **nie je vlastnosť architektúry, ale tréningu**: base model po pretrainingu nepozná koniec odpovede a pokračuje donekonečna. Umiestnenie EOS na správne miesto je jedna z hlavných vecí, ktoré sa učí pri [inštrukčnom ladení](03-llm-trening.md).

#### Prečo „premýšľanie" znamená viac tokenov

Jeden prechod modelom má **pevnú cenu**: zhruba `2 × počet parametrov` operácií na token, nech je otázka akokoľvek ťažká. Model nemá ako nad jedným tokenom „zostať dlhšie" — neexistuje slučka, ktorá by mu dala viac výpočtu na tej istej pozícii.

Jediný spôsob, ako do úlohy naliať viac výpočtu, je teda **vygenerovať viac tokenov**. Odtiaľ pochádza celá rodina techník:

- `chain-of-thought` („rozmýšľaj krok po kroku") funguje preto, lebo medzivýsledky sa stanú **tokenmi v sekvencii** — a tie sú pre ďalšie prechody čitateľné cez attention. Sekvencia je poznámkový blok modelu, KV cache jeho pracovná pamäť.
- **Reasoning modely** (o-séria, DeepSeek R1, Claude s extended thinking) sú na toto natrénované posilňovaným učením na overiteľných úlohách: pred odpoveďou vygenerujú dlhý reťazec uvažovania, ktorý API často skryje, ale **účtuje ako výstupné tokeny**.
- Preto platí, že kvalita pri týchto modeloch rastie s **rozpočtom na premýšľanie** — a rovnako tak latencia a cena. Pri `max_tokens` treba počítať aj so skrytými tokenmi, inak sa odpoveď oreže ešte pred svojím začiatkom.

Pekne to uzatvára celú sekciu: model nevie myslieť „potichu". Všetko, čo si potrebuje zapamätať medzi dvoma prechodmi, musí **napísať**.

---

### 7. Ďalší prechod: autoregresia a KV cache

Vybraný token sa pripojí na koniec sekvencie a celé sa to opakuje. Naivne by to znamenalo: pri generovaní 500. tokenu prepočítať celý model nad 500 tokenmi. To by bolo kvadratické plytvanie — a **nie je to potrebné**.

Kľúčové pozorovanie: **vďaka kauzálnej maske nový token nemení nič, čo už bolo spočítané.** Token č. 500 nemôže ovplyvniť vektory tokenov 1–499, lebo tie sa naň nesmú pozerať. Ich `K` a `V` vektory sú teda **navždy platné** a stačí si ich odložiť.

#### KV cache

```text
FÁZA 1 — PREFILL (prompt, n tokenov naraz)
  ─────────────────────────────────────────
  všetkých n tokenov prejde všetkými vrstvami paralelne
  cestou sa do cache uložia K a V každého tokenu v každej vrstve
  cena: O(n²) v attention  →  prvý token odpovede trvá najdlhšie

FÁZA 2 — DECODE (každý ďalší token, jeden po druhom)
  ─────────────────────────────────────────
  do modelu vstúpi JEDEN vektor (posledný token)
    ├─ spočíta sa jeho Q, K, V                (1 riadok, nie n)
    ├─ jeho K, V sa PRIPOJÍ do cache
    ├─ attention: jeho Q proti VŠETKÝM K, V z cache   → O(n), nie O(n²)
    ├─ FFN nad jedným vektorom
    └─ lm_head → ďalší token → späť na začiatok fázy 2
```

**Čo sa ukladá do cache:** `K` a `V` každého tokenu, v každej vrstve, pre každú KV hlavu.
**Čo sa do cache neukladá:** `Q` (nový token má vlastné a staré už netreba), aktivácie FFN (pre nový token sú vždy nové), attention váhy (závisia od nového `Q`).

Veľkosť KV cache sa počíta priamočiaro:

```text
bajtov na 1 token = 2 (K a V) × n_layers × n_kv_heads × d_head × veľkosť_typu

referenčný model, fp16:
  2 × 32 × 8 × 128 × 2 B = 131 072 B = 128 KiB na token

  kontext   8 000 tokenov →   1.05 GB
  kontext  32 000 tokenov →   4.19 GB
  kontext 128 000 tokenov →  16.8 GB   ← viac, než váži samotný model (16 GB v fp16)!

ten istý model BEZ GQA (32 KV hláv namiesto 8):  512 KiB/token → 67 GB pri 128k

(1 KiB = 1024 B, ale GB tu počítam ako 10⁹ B — tak to uvádzajú aj výrobcovia GPU)
```

Preto moderné modely takmer bez výnimky používajú **GQA** (*grouped-query attention* — viac Q hláv zdieľa jednu K/V hlavu) alebo **MQA** (všetky Q hlavy zdieľajú jednu). Nie kvôli rýchlosti výpočtu, ale kvôli **veľkosti cache**.

#### Prečo je prvý token pomalý a ďalšie rýchle

| | Prefill | Decode (1 token) |
|---|---|---|
| Koľko tokenov vstúpi | `n` | 1 |
| Attention náklad | `O(n²)` | `O(n)` proti cache |
| Čo limituje | **výpočet** (GPU FLOPs) | **priepustnosť pamäte** |
| Typicky (jedna bežná GPU) | 1 000 tokenov ≈ 0,2 s | 15–20 ms/token |

Decode je **memory-bound**, a to je najdôležitejšia praktická vec na tejto stránke: pre každý jediný vygenerovaný token musí GPU prečítať **všetkých 8 miliárd váh** (16 GB v fp16) z pamäte. Pri priepustnosti ~1 TB/s to je ~16 ms na token, teda ~60 tokenov/s — **bez ohľadu na to, aký máte rýchly čip**. Odtiaľ dva dôsledky: kvantizácia na 8/4 bity zrýchľuje generovanie priamo úmerne (menej bajtov na prečítanie) a **dávkovanie (batching)** je zadarmo — tie isté prečítané váhy obslúžia 30 používateľov naraz.

#### Špekulatívne dekódovanie: ako obísť memory-bound strop

Ak je decode limitovaný čítaním váh a nie počítaním, ponúka sa trik: **overiť viac tokenov naraz za jedno prečítanie váh**. Presne to robí *speculative decoding*:

```text
1. malý „draft" model (napr. 1B) vygeneruje rýchlo k = 5 tokenov     ← lacné čítanie
2. veľký model ich overí v JEDNOM prechode (ako prefill nad 5 tokenmi)
3. prijme najdlhšiu zhodnú predponu, prvý nesúhlas prepíše svojím tokenom
4. pokračuje odtiaľ ďalej
```

Overenie 5 tokenov stojí veľký model takmer to isté ako jeden token — váhy sa aj tak čítajú celé, len raz. Pri rozumnej miere prijatia to dáva **1,5–3× rýchlejšie generovanie**, a to pri **matematicky identickom rozdelení** výstupu (odmietanie je navrhnuté tak, aby nemenilo pravdepodobnosti). Varianty sa líšia len tým, odkiaľ návrhy berú: menší model tej istej rodiny, prídavné predikčné hlavy (Medusa, EAGLE), alebo jednoduché hľadanie n-gramov v prompte (užitočné pri sumarizácii, kde sa veľa textu opakuje).

#### Cache o úroveň vyššie: prompt caching

KV cache žije v rámci jednej požiadavky. Ak však posielate **ten istý prefix** znova a znova (systémový prompt, dlhý dokument, definície nástrojov), dá sa uložiť aj medzi požiadavkami — API to ponúkajú ako **prompt caching** (Anthropic `cache_control`, OpenAI automaticky). Zápis do cache stojí o niečo viac, čítanie z nej rádovo menej než bežné vstupné tokeny, a preskočí sa prefill.

Podmienka je jediná, ale nekompromisná: **prefix sa musí zhodovať bajt na bajt**. Z toho vyplýva pravidlo skladania promptov, ktoré platí aj pri agentoch aj pri RAG:

```text
✅ stabilné veci DOPREDU:   systémový prompt → definície nástrojov → dokumenty → história → otázka
❌ meniace sa veci dozadu:  timestamp alebo ID používateľa na začiatku promptu znehodnotí cache celej konverzácie
```

Ďalšie úrovne ukladania do cache, s ktorými sa v praxi stretnete:

| Úroveň | Čo sa ukladá | Kde |
|---|---|---|
| **KV cache** | K/V vektory tokenov | GPU pamäť, v rámci jednej generácie |
| **Prompt / prefix cache** | KV cache spoločného prefixu | server modelu, medzi požiadavkami (vLLM, Anthropic, OpenAI) |
| **PagedAttention** | KV cache v stránkach ako virtuálna pamäť | vLLM — umožní zdieľať prefix medzi používateľmi bez kopírovania |
| **Embedding cache** | vektory už zaindexovaných chunkov | vektorová DB — [RAG](06-rag.md), aby sa embedding nepočítal znova |
| **Response cache** | celé odpovede na rovnaké otázky | aplikačná vrstva (Redis a pod.) |

---

### 8. Kontext: krátka správa, dlhá správa a prečo má okno strop

Model má okno 8192 tokenov, ale používateľ napísal *„Ahoj"* — aj s chat šablónou je to okolo piatich tokenov. Čo robí zvyšných 8187 pozícií? Sú neaktívne? Vyplnené nulami? Odpoveď je prekvapivo jednoduchá: **neexistujú**.

Táto sekcia dopovedá to, čo začali [dve osi vstupnej matice](#dve-osi-vstupnej-matice-šírka-a-kontextové-okno): kontextové okno je **strop na počet riadkov** matice `X`, nie rozmer akejkoľvek vrstvy. Tu sa pozrieme, čo z toho plynie pre pamäť, čas a cenu.

#### Kontextové okno je strop, nie nádoba

Toto je hlavný rozdiel oproti sieťam, ktoré ste videli predtým. [Feed-forward sieť](../02-typy-modelov/04-feed-forward-siete.md) má vstupnú vrstvu s pevným počtom neurónov; [CNN](../02-typy-modelov/05-konvolucne-siete.md) čaká obrázok pevného rozmeru — a keď dáte menší vstup, musíte ho doplniť alebo preškálovať. **Transformer takú vrstvu nemá.**

Pozrite sa späť na [cestu vrstvou](#4-cesta-jedného-vektora-jednou-vrstvou): každá matica váh má tvar `[4096, …]` — teda rozmer **jedného tokenu**. Nikde vo váhach nie je číslo `n`. Tie isté váhy sa aplikujú na 5 riadkov aj na 5000 riadkov; jediné, čo sa mení, je výška matíc, ktoré nimi pretekajú:

```text
"Ahoj" (n = 5)                        dlhý dokument (n = 5000)
  X        [    5, 4096]                X        [ 5000, 4096]
  Q·Kᵀ     [32,  5,    5]               Q·Kᵀ     [32, 5000, 5000]
  H        [    5, 4096]                H        [ 5000, 4096]
  KV cache  5 × 128 KiB = 0.6 MiB       KV cache  5000 × 128 KiB = 0.64 GB
  váhy     8 mld. — ROVNAKÉ            váhy     8 mld. — ROVNAKÉ
```

Takže: **žiadne pozície sa nevypínajú, lebo sa nikdy nezapli.** Nealokuje sa 8192 slotov, neprepočítavajú sa prázdne miesta, nikde nie sú nuly. Krátky prompt teda nie je nevyužitá kapacita — je **priamo úmerne lacnejší**.

#### Kedy padding predsa len existuje: dávkovanie

Jediné miesto, kde sa objavia „neaktívne" pozície, je **spracovanie viacerých požiadaviek naraz**. GPU chce jednu veľkú maticu, ale požiadavky majú rôznu dĺžku — klasické riešenie je doplniť ich `[PAD]` tokenmi na dĺžku najdlhšej a priložiť **attention mask**:

```text
batch 3 požiadaviek, padding sprava:

  req A:  Aké  je  hlavné  mesto  ?                  mask: 1 1 1 1 1
  req B:  Ahoj  ?   PAD    PAD   PAD                 mask: 1 1 0 0 0
  req C:  Koľko dní dovolenky mám ?                  mask: 1 1 1 1 1
```

Maska sa **pripočíta k logitom attention ako `−∞`** ešte pred softmaxom (presne tak isto ako kauzálna maska v [sekcii 4](#4-cesta-jedného-vektora-jednou-vrstvou)), takže po softmaxe majú padované pozície váhu **presne 0** a do výsledku neprispejú ničím. *Toto* sú tie „neaktívne vstupy" — ale sú to artefakty dávkovania, nie súčasť modelu. Pri jedinej požiadavke žiadny padding nevzniká.

Dve praktické poznámky:

- **Pri generovaní sa paduje zľava**, nie sprava. Nový token sa totiž vždy číta z poslednej pozície — a keby tam boli `PAD`, model by pokračoval za výplňou. (`tokenizer.padding_side = "left"` je jedna z najčastejších chýb pri dávkovej inferencii cez `transformers`.)
- **Moderné inferenčné servery padding nepoužívajú vôbec.** vLLM a TGI sekvencie **zreťazia za seba** (*varlen*), do jadra attention pošlú zoznam hraníc a KV cache držia po stránkach (*PagedAttention*). Nulová pamäť navyše, a k tomu *continuous batching*: keď jedna požiadavka skončí, na jej miesto v dávke okamžite nastúpi ďalšia, bez čakania na zvyšok.

#### Ako sa dĺžka prejaví na čase a pamäti

Rádové čísla pre referenčný 8B model v fp16 na jednej bežnej GPU:

| Dĺžka promptu `n` | Attention matica (1 hlava, 1 vrstva) | KV cache | Čas do prvého tokenu (prefill) | Rýchlosť generovania |
|---|---|---|---|---|
| 12 (*„Ahoj, ako sa máš?"*) | 144 čísel | 1,5 MiB | jednotky ms | ~60 tok/s |
| 200 (krátka otázka + systémový prompt) | 40 000 | 25 MiB | ~0,04 s | ~60 tok/s |
| 2 000 (dokument na stranu A4) | 4 milióny | 0,25 GB | ~0,4 s | ~60 tok/s |
| 32 000 (dlhá konverzácia / PDF) | 1 miliarda | 4,2 GB | ~10 s | ~45 tok/s |

Tri veci, ktoré z tabuľky plynú:

1. **Dlhá správa predražuje hlavne prefill**, teda **čakanie na prvý token** — rastie nadlineárne. Používateľ to vníma ako „model dlho premýšľa", hoci ešte nič negeneroval.
2. **Rýchlosť generovania klesá až pri naozaj dlhom kontexte** — decode je [memory-bound](#prečo-je-prvý-token-pomalý-a-ďalšie-rýchle) a KV cache je oproti 16 GB váh do ~4k tokenov zanedbateľná. Pri 32k už tvorí štvrtinu prečítaných dát.
3. **Krátke správy nie sú „pod rozlišovacou schopnosťou" modelu.** Prompt s 5 tokenmi prejde tými istými 32 vrstvami a tými istými 8 miliardami váh ako prompt s 5000 tokenmi. Kvalita nezávisí od toho, koľko okna využijete.

#### Prečo sa okno nedá len tak zväčšiť

`n_ctx` nie je softvérové obmedzenie, ktoré by sa dalo „odomknúť" zmenou konštanty v konfiguráku. Sú to štyri nezávislé steny, z toho dve už poznáte z čísel vyššie:

| Stena | V čom je problém |
|---|---|
| **Kvadratická attention** | matica skóre má `n × n` prvkov v každej hlave a vrstve — z 1k na 10k tokenov je **100× viac** práce v attention |
| **Veľkosť KV cache** | 128 KiB na token ([výpočet v sekcii 7](#kv-cache)) — pri 128k je to 16 GB **na každého používateľa**, viac než váži model |
| **Tréningová dĺžka** | RoPE je funkcia, ktorú model videl len na pozíciách 0…`n_ctx`; za hranicou extrapoluje do neznáma a kvalita spadne |
| **Pokles kvality skôr než hranica** | presnosť práce s informáciou **uprostred** dlhého kontextu je preukázateľne horšia než na jeho začiatku a konci (*lost in the middle*) |

Posledné dve stoja za zdôraznenie, lebo sa na ne zabúda: deklarované okno je **horná hranica, nie odporúčaná pracovná dĺžka**, a rozšírenie okna vyžaduje **doučenie** modelu (RoPE scaling / YaRN + dotrénovanie na dlhých textoch), nie prepísanie čísla v `config.json`.

A k tomu **cena**: vstupné tokeny sa platia. 100 000 tokenov v každej otázke je pri agentovi, ktorý sa pýta stokrát, reálny účet — presne preto existuje [RAG](06-rag.md), ktorý do promptu vloží 5 relevantných odsekov namiesto celej dokumentácie.

#### Čo s tým robia moderné modely

| Technika | Ktorú stenu posúva | Ako |
|---|---|---|
| **FlashAttention** | pamäť pri prefille | nikdy nezmaterializuje celú maticu `n×n`, počíta ju po dlaždiciach; matematicky identický výsledok |
| **GQA / MQA** ([sekcia 7](#kv-cache)) | veľkosť KV cache | menej K/V hláv (8 namiesto 32) → 4× menšia cache |
| **Kvantizácia KV cache** | veľkosť KV cache | K/V v 8 bitoch namiesto 16 → polovičná cache |
| **Sliding window** | kvadratika | token vidí len posledných napr. 4096 tokenov, vzdialenejšie sprostredkovane cez vrstvy |
| **RoPE scaling / YaRN** | tréningová dĺžka | preškáluje pozičné frekvencie + krátke dotrénovanie → rozšíri okno z 8k na 128k |
| **PagedAttention** ([sekcia 7](#cache-o-úroveň-vyššie-prompt-caching)) | fragmentácia pamäte | KV cache po stránkach, zdieľanie prefixu medzi požiadavkami |

#### Konverzácia: prečo `n` rastie aj pri krátkych otázkach

Model je **bezstavový**. Nepamätá si predchádzajúcu otázku — server mu pri každom kole posiela **celú históriu odznova**:

```text
kolo 1:  [systém][otázka 1]                                        →  n =   320
kolo 2:  [systém][otázka 1][odpoveď 1][otázka 2]                   →  n =   980
kolo 3:  [systém][otázka 1][odpoveď 1][otázka 2][odpoveď 2][o. 3]  →  n = 1 740
   ⋮
kolo 20: … celá história …                                          →  n = 18 000
```

Dvadsiata otázka má päť slov, ale prefill je nad 18 000 tokenmi. Preto sa v praxi robia štyri veci:

| Technika | Čo rieši |
|---|---|
| **Prompt caching** ([sekcia 7](#cache-o-úroveň-vyššie-prompt-caching)) | nemenný prefix už znova neprechádza prefillom — najúčinnejší nástroj, preto stabilné veci dopredu |
| **Orezanie histórie** | staré kolá sa jednoducho zahodia (systémovú správu treba chrániť pred orezaním) |
| **Sumarizácia histórie** | staršie kolá nahradí krátke zhrnutie vygenerované modelom |
| **[RAG](06-rag.md)** | namiesto celého dokumentu sa do promptu vloží 5 relevantných odsekov |

A keď `n` aj tak narazí na strop? Model nemá ako „pretiecť" — riešenie je **vždy na aplikačnej vrstve**: API vráti chybu (`context_length_exceeded`), alebo knižnica potichu odreže najstaršie tokeny (pozor, takto sa dá nepozorovane stratiť systémový prompt), alebo má model natrénované sliding window (Mistral 7B: 4096) a staršie tokeny vidí len sprostredkovane.

---

### 9. Učenie v kontexte (in-context learning)

Toto je vlastnosť, ktorá LLM najviac odlišuje od všetkého, čo bolo v [lekcii 2](../02-typy-modelov/README.md):
**model vyrieši úlohu, na ktorú nebol trénovaný, len preto, že mu dáte pár príkladov v prompte
— a neupraví sa pritom ani jediná váha.**

```text
Prompt:
  zlá kvalita zvuku → HARDVÉR
  nesedí faktúra    → FAKTURÁCIA
  aplikácia padá po štarte → ???

Model: SOFTVÉR
```

Žiadny tréning neprebehol. Po odoslaní odpovede model o tejto úlohe **nevie nič** — pri ďalšom
volaní začína od nuly. Napriek tomu sa správa, akoby sa práve niečo naučil. Odtiaľ názov
*in-context learning* (ICL) a jeho praktická podoba, ktorú poznáte z promptovania: **few-shot**
(pár príkladov), **zero-shot** (len inštrukcia).

#### Prečo to funguje

Nie je to záhada, ale priamy dôsledok mechanizmov z predošlých sekcií:

1. **Váhy sú fixné, ale aktivácie nie.** Váhy kódujú „ako spracovať kontext"; obsah kontextu
   je vstup. Keď do okna pridáte príklady, zmení sa **matica `X`**, nie váhy `W` — a s ňou aj
   všetko, čo z nej attention vypočíta. Preto je [dvojica osí `X`](#dve-osi-vstupnej-matice-šírka-a-kontextové-okno)
   tak dôležitá: kontext je jediný vstup, ktorým sa dá správanie modelu meniť za behu.
2. **Attention vie kopírovať a párovať vzory.** Interpretačné práce našli v modeloch
   **indukčné hlavy**: dvojicu attention hláv, ktorá implementuje pravidlo *„ak si niekde
   vyššie videl `[A][B]` a práve teraz vidíš `[A]`, predpovedz `[B]`"*. Prvá hlava sa pozrie
   o token späť, druhá nájde v kontexte predošlý výskyt toho istého tokenu a skopíruje, čo po
   ňom nasledovalo. Na tom stojí celé dopĺňanie vzorov z promptu.
3. **Vzniklo to samo počas pretrainingu.** Nikto ICL netrénoval. Predikcia ďalšieho tokenu
   nad internetom je plná úloh typu „tu je vzor, pokračuj v ňom" (zoznamy, tabuľky, preklady,
   opakujúce sa formáty), takže sa schopnosť dopĺňať vzor oplatí. Objavenie sa indukčných hláv
   vidno v tréningu ako **náhly zlom** na krivke loss.

#### Čo z toho plynie pre prax

- **Príklady sú najsilnejšia páka v prompte.** Ukázať tri príklady požadovaného výstupu je
  spoľahlivejšie než ten istý formát opísať vetami ([3.3 v lekcii 8](../05-prakticke/01-ako-pouzivat-llm.md#33-príklady--najsilnejší-signál-v-celom-prompte)).
- **Formát príkladov nesie viac než ich správnosť.** Experimenty ukázali, že aj keď sa
  v príkladoch **zámerne pomiešajú labely**, väčšina zisku ostane: model si z nich berie hlavne
  *tvar úlohy* a *množinu možných odpovedí*, nie ich vecný obsah. Preto majte príklady
  konzistentné vo formáte a pokrývajúce všetky triedy — a preto samotné „pridám príklady" ešte
  neznamená „naučil som ho fakty".
- **Kontext prebíja naučené.** Model dá pri konflikte väčšinou prednosť tomu, čo má v okne,
  pred tým, čo má vo váhach. **Na tom stojí RAG** ([06-rag.md](06-rag.md)) — a z toho istého
  dôvodu funguje prompt injection ([7. v lekcii 8](../05-prakticke/01-ako-pouzivat-llm.md#7-bezpečnosť-text-zvonku-nie-je-inštrukcia)):
  vložený text je pre model rovnocenný vstup ako vaša inštrukcia.
- **Platí sa za to pri každom volaní.** ICL nič trvalo neuloží: tie isté príklady sa posielajú
  a prefillujú znova a znova. Fine-tuning je opačný obchod — zaplatí sa raz v tréningu a prompt
  je potom krátky. Preto je [rozhodnutie prompt vs. fine-tuning](07-fine-tuning-lora.md#4-kedy-fine-tuning-áno-a-kedy-nie)
  hlavne ekonomické. ([Prompt caching](#cache-o-úroveň-vyššie-prompt-caching) ten rozdiel
  výrazne zmenšuje.)

> **Časté nedorozumenie:** „model sa z našich konverzácií učí". Pri jednom volaní API sa neučí
> nič — po skončení požiadavky sa aktivácie zahodia a váhy sú tie isté. To, čo vyzerá ako
> pamäť medzi kolami, je [história posielaná znova](#konverzácia-prečo-n-rastie-aj-pri-krátkych-otázkach)
> (prípadne pamäť, ktorú si aplikácia sama ukladá a vkladá do promptu). Model sa zmení až
> vtedy, keď niekto spustí **tréning**.

---

### 10. Prehľad parametrov transformera

Slovo „parameter" znamená v kontexte LLM tri rôzne veci — oplatí sa ich nemiešať.

#### a) Architektonické hyperparametre (v `config.json` modelu, meniť sa nedajú)

| Parameter | HF názov | Typicky | Čo ovplyvňuje |
|---|---|---|---|
| `d_model` | `hidden_size` | 768–8192 | kapacitu vektora, väčšinu veľkosti modelu |
| `n_layers` | `num_hidden_layers` | 12–80 | hĺbku uvažovania, latenciu |
| `n_heads` | `num_attention_heads` | 12–64 | koľko typov vzťahov naraz |
| `n_kv_heads` | `num_key_value_heads` | 1–8 (GQA) | **veľkosť KV cache** |
| `d_ff` | `intermediate_size` | 4× (resp. 8/3×) `d_model` | 80 % parametrov vrstvy |
| `vocab` | `vocab_size` | 32k–256k | dĺžku tokenizácie, veľkosť lm_head |
| `n_ctx` | `max_position_embeddings` | 4k–1M | maximálny kontext |
| aktivácia | `hidden_act` | `silu`, `gelu` | nelinearitu v FFN |
| počet expertov | `num_local_experts` | 8–256 (len MoE) | celkové parametre, nároky na VRAM |
| expertov na token | `num_experts_per_tok` | 2–8 (len MoE) | **aktívne** parametre, teda rýchlosť |
| RoPE báza | `rope_theta` | 10 000 – 500 000 | rozsah pozícií |
| norm epsilon | `rms_norm_eps` | 1e-5 | numerickú stabilitu |
| zdieľanie embeddingov | `tie_word_embeddings` | true/false | veľkosť malých modelov |

#### b) Naučené váhy (to, čo tréning nastavuje — „8 miliárd parametrov")

Embedding matica, `W_Q/W_K/W_V/W_O` v každej vrstve, tri FFN matice v každej vrstve, škálovacie parametre noriem, výstupná `W_U`. Nič iné v modeli nie je.

#### c) Inferenčné parametre (nastavujete pri každom volaní)

| Parameter | Čo robí | Kedy meniť |
|---|---|---|
| `temperature` | plochosť rozdelenia pred výberom | 0–0.3 faktický text, 0.7–1.0 kreatívny text |
| `top_p` / `top_k` | orezanie chvosta rozdelenia | 0.9 / 40 ako rozumné východisko |
| `repetition_penalty`, `presence/frequency_penalty` | trestá opakovanie | keď sa model zacyklí |
| `max_new_tokens` | strop dĺžky odpovede | vždy — chráni pred nekontrolovane dlhou odpoveďou |
| `stop` sekvencie | kde skončiť | pri štruktúrovanom výstupe |
| `seed` | zopakovateľnosť samplovania | pri testovaní |
| `dtype` / kvantizácia | fp16, int8, int4 | pamäť vs. kvalita |

Do tejto skupiny patria aj [dekódovacie stratégie](01-transformer-siete.md#ako-presne-sa-vyberá-ďalší-token-dekódovanie) — a stojí za zopakovanie, že **žiadny z týchto parametrov nemení váhy modelu**. Menia len to, ako sa z logitov vyberá token.

#### d) Presnosť čísel a kvantizácia

Všetky doterajšie počty parametrov hovoria, **koľko čísel** model má. Koľko zaberú v pamäti,
určuje až **presnosť**, v ktorej sú uložené:

| Formát | Bajtov na parameter | Kde sa používa |
|---|---|---|
| `fp32` | 4 | master kópia váh pri tréningu; dnes sa v nej neinferuje |
| `bf16` / `fp16` | 2 | **štandard pre inferenciu aj tréning** (`bf16` má väčší rozsah, znáša outliery) |
| `fp8` | 1 | natívne na H100/H200; serverová inferencia s minimálnou stratou |
| `int8` | 1 | kvantizácia váh, kvalita prakticky nerozoznateľná |
| `int4` | 0,5 | lokálny beh na bežnej karte — najčastejší kompromis |

Odtiaľ vzorec, ktorý sa oplatí vedieť naspamäť:

```text
VRAM na váhy ≈ počet parametrov × bajtov na parameter
celková VRAM ≈ váhy × ~1,2  +  KV cache (podľa dĺžky kontextu a počtu používateľov)
```

| Model | `fp16` | `int8` | `int4` | Kam sa vojde v `int4` |
|---|---|---|---|---|
| 8 B | 16 GB | 8 GB | ~5 GB | notebook s 8GB GPU |
| 70 B | 140 GB | 70 GB | ~40 GB | 2× 24GB karta |
| 405 B | 810 GB | 405 GB | ~230 GB | stále serverový klaster |

**Ako sa kvantizuje.** Nie je to obyčajné zaokrúhlenie: váhy sa delia na malé bloky a každý
dostane vlastnú mierku (škálu), takže sa zachová rozsah aj pri outlieroch. Podľa nástroja
narazíte na:

| Metóda | Kde |
|---|---|
| **GGUF** (`Q4_K_M`, `Q5_K_M`…) | `llama.cpp`, Ollama, LM Studio — beh na CPU aj GPU, najbežnejší formát na lokálny beh |
| **GPTQ**, **AWQ** | kvantizácia váh pre GPU inferenciu (vLLM, TGI); AWQ chráni „dôležité" váhy podľa aktivácií |
| **bitsandbytes NF4** | 4-bit základ pod LoRA adaptérmi = [QLoRA](07-fine-tuning-lora.md#3-qlora--lora-na-kvantizovanom-modeli) |

**Čo sa tým stráca.** Pri 8 bitoch prakticky nič; pri dobrej 4-bitovej metóde stúpne
[perplexita](03-llm-trening.md#ako-sa-meria-pokrok-loss-a-perplexita) len mierne, ale strata sa
neprejaví rovnomerne — najskôr ju vidno na **dlhom kontexte, reasoningu a menej zastúpených
jazykoch** (teda aj na slovenčine), nie na krátkych anglických otázkach, na ktorých sa to
zvyčajne testuje. Pod 4 bity už kvalita padá citeľne.

Praktické pravidlo: **väčší model v `int4` býva lepší než menší model v `fp16`** pri rovnakej
pamäti (14B v 4 bitoch zvyčajne prekoná 7B v 16 bitoch). A dve veci, ktoré kvantizácia
**nerieši**: KV cache (tá sa kvantizuje zvlášť — [sekcia 8](#čo-s-tým-robia-moderné-modely))
a tréning (ten potrebuje vyššiu presnosť, preto QLoRA drží zamrznuté váhy v 4 bitoch, ale
adaptéry trénuje v `bf16`).


---

## Zhrnutie

**Časť A — vstup**

| Otázka | Odpoveď v jednej vete |
|---|---|
| Čo model dostáva na vstupe? | Postupnosť ID tokenov z tokenizéra (byte-level BPE) — nie písmená a nie slová; tokenizér je bez váh a patrí k modelu napevno. |
| Ako sa z ID stane vektor? | Lookup riadku v embedding matici `[128 256, 4096]`; pozíciu pridá až RoPE rotáciou `Q` a `K` v každej vrstve. |
| Ako súvisí okno so vstupnou maticou? | `X = [n, d_model]`: **riadky = kontext** (mení sa, strop `n_ctx`), **stĺpce = šírka** (konštanta zapečená vo váhach). Okno nie je vstupná vrstva — v žiadnej matici váh `n` nie je. |
| Prečo je slovenčina drahšia? | Rozreže sa na ~2× viac tokenov než angličtina → 2× cena, 2× kontext, 2× čas generovania. |
| Ako sa do toho zmestí obrázok? | Ako patche prevedené projekciou na `d_model` — od vstupu do vrstiev je to len ďalší riadok matice `X`. |
| Prečo model nespočíta písmená? | Nevidí písmená, ale ID tokenov (`str` + `aw` + `berry`) — je to chyba vstupu, nie uvažovania; rieši sa nástrojom alebo rozpísaním po znakoch. |

**Časť B — priechod modelom**

| Otázka | Odpoveď v jednej vete |
|---|---|
| Odkiaľ je veľkosť vektorov? | `d_model` je voľba návrhára: `n_heads × d_head` (64/128), násobok 128 kvôli GPU, `d_ff ≈ 4× d_model`. |
| Kde sú v modeli parametre? | ~80 % vrstvy (a ~70 % modelu) vo feed-forward, zvyšok v attention a embeddingoch. |
| Čím sa líši 8B a 405B model? | 50× viac parametrov, ale len 3,9× viac vrstiev — veľký model je hlavne **širší** (`N ≈ 12 · n_layers · d_model²`). |
| Prečo sa rastie do šírky, nie do hĺbky? | Šírka sa paralelizuje vnútri vrstvy (tensor parallel), hĺbka je sériová reťaz: pri generovaní ju nezrýchli žiadny počet GPU, len predlžuje latenciu na token. |
| Kde sa tokeny miešajú? | **Iba v attention.** Norm, FFN aj reziduá bežia per token. |
| Čo drží model pokope? | Reziduálny prúd: vrstvy do spoločného vektora `[4096]` len pripočítavajú — preto je `d_model` konštantné a preto odstránenie jednej vrstvy model nezničí. |
| Čo je MoE? | FFN rozdelená na expertov s routerom: celkové parametre určujú VRAM, aktívne parametre rýchlosť (`235B-A22B` = 235 mld. v pamäti, 22 mld. na token). |

**Časť C — generovanie a kontext**

| Otázka | Odpoveď v jednej vete |
|---|---|
| Ako vznikne token? | Posledný vektor → norm → `lm_head` (4096×128k) → logity → úpravy → softmax → výber. |
| Prečo len jeden token? | Vyjde ich `n`, ale predpovede pre pozície 1…`n−1` sú tokeny, ktoré už poznáme — pri inferencii sa zahodia, pri tréningu sa použijú všetky. |
| Prečo nie viac tokenov naraz? | Rozdelenie ďalšieho tokenu je podmienené tým predošlým; viac naraz sa dá len **navrhnúť** a dať veľkému modelu overiť (špekulatívne dekódovanie). |
| Ako vznikne slovo? | Viacerými prechodmi; medzi nimi model nedrží žiadny zámer — jediný stav je predĺžená sekvencia tokenov a KV cache. |
| Prečo „premýšľanie" = viac tokenov? | Výpočet na jeden token je pevný, takže jediný spôsob, ako pridať výpočet, je napísať viac tokenov. |
| Prečo je prvý token pomalý? | Prefill je `O(n²)` a compute-bound; decode je `O(n)` a memory-bound. |
| Čo sa ukladá do cache? | `K` a `V` každého tokenu v každej vrstve; `Q` a aktivácie FFN nie. |
| Čo s nevyužitým kontextom? | Nič — `n_ctx` je strop, nie nádoba. Nepoužité pozície neexistujú; padding vzniká len pri dávkovaní a je maskovaný na nulu. |
| Prečo je kontext obmedzený? | Kvadratická attention + veľkosť KV cache + tréningová dĺžka RoPE + pokles kvality uprostred. |
| Ako sa model „učí" z promptu? | Nijako trvalo — príklady menia vstup `X`, nie váhy; attention z nich cez indukčné hlavy skopíruje vzor. Po skončení volania nezostane nič. |
| Koľko pamäte model zaberie? | `parametre × bajty na parameter` (fp16 = 2, int4 = 0,5) × ~1,2 + KV cache; väčší model v `int4` býva lepší než menší v `fp16`. |

---

## Kontrolné otázky

**K časti A**

1. Prečo nemôžete použiť tokenizér od GPT-4 s modelom Llama 3, hoci obidva sú byte-level BPE?
2. Ten istý text máte po anglicky aj po slovensky. Ktorá verzia sa bude generovať dlhšie a prečo — vysvetlite cez počet prechodov modelom.
3. Čím sa líši RoPE od pôvodného sínusového kódovania a prečo práve vďaka tomu ide okno modelu rozšíriť dotrénovaním?
4. Kolega tvrdí: „model s oknom 128k má 128 000 vstupných neurónov". Opravte ho cez tvar matice `X` a povedzte, ktorá os matice zodpovedá oknu a ktorá šírke modelu. Ktorá z nich sa platí parametrami a ktorá výpočtom pri inferencii?
5. Model spoľahlivo napíše esej, ale tvrdí, že v slove „strawberry" sú dve `r`. Vysvetlite príčinu cez tokenizáciu a navrhnite dve rôzne riešenia.

**K časti B**

5. Model má `d_model = 4096` a `n_heads = 32`. Aký je rozmer jednej hlavy a prečo nemôže byť `n_heads = 30`?
6. Prečo sa v modeli s 8 miliardami parametrov nachádza väčšina váh vo feed-forward vrstvách a nie v attention? Spočítajte to pre jednu vrstvu.
7. Vysvetlite, prečo musí mať `d_model` vo všetkých vrstvách rovnakú hodnotu. Použite pojem reziduálneho prúdu.
8. Model je označený ako `A22B` pri celkovej veľkosti 235 mld. parametrov. Koľko VRAM budete zhruba potrebovať v fp16 a akú rýchlosť generovania očakávate — a prečo to nie sú dve strany tej istej mince?
9. Prečo RMSNorm neporušuje tvrdenie, že tokeny sa miešajú iba v attention?
10. Model A má `n_layers = 32`, `d_model = 8192`, model B `n_layers = 128`, `d_model = 4096`. Ukážte, že majú približne rovnaký počet parametrov, a vysvetlite, ktorý z nich bude generovať rýchlejšie a prečo.
11. Prečo sa šírka vrstvy dá rozdeliť medzi 8 GPU tak, že počítajú súčasne, kým vrstvy medzi sebou takto rozdeliť nejde?
12. Pipeline má `P = 8` stupňov. Aký podiel času GPU stoja pri tréningu so 64 mikro-dávkami a aký pri generovaní token po tokene? Čo z toho plynie pre hlboké modely v interaktívnej prevádzke?
13. Výpočet vo vrstve rastie s `d_model²`, komunikácia pri tensor parallel len s `d_model`. Prečo z toho vyplýva, že širší model lepšie využije GPU klaster?
14. Qwen3 235B-A22B má rovnaké `d_model` ako Llama 3 8B a užšie FFN. Kde je teda tých zvyšných 227 miliárd parametrov a prečo je taký tvar výhodný pre poskytovateľa služby?
15. Prečo nemá zmysel postaviť model s 8 vrstvami a `d_model = 100 000`, hoci by mal parametrov dosť? Uveďte dva nezávislé dôvody.

**K časti C**

16. Do modelu vojde 2000 tokenov a z poslednej vrstvy vyjde 2000 vektorov. Prečo sa 1999 z nich pri inferencii zahodí a kedy sa naopak použijú všetky?
17. Prečo model nevyrobí päť tokenov na jeden prechod, keď by tým generoval päťkrát rýchlejšie? Vysvetlite to cez rozklad `P(t₁…tₘ)` a povedzte, čo na tom mení špekulatívne dekódovanie.
18. Používateľ pošle prompt s piatimi tokenmi do modelu s oknom 8192. Čo sa deje so zvyšnými 8187 pozíciami? Odpovedzte cez tvary tenzorov.
19. Kedy v LLM reálne vzniknú `[PAD]` tokeny a ako sa zabezpečí, že neovplyvnia výsledok? Prečo sa pri generovaní paduje zľava?
20. Vysvetlite, prečo sa `K` a `V` dajú uložiť do cache, ale `Q` nie. Čo konkrétne to umožňuje — ktorá vlastnosť decoder-only modelu?
21. Spočítajte KV cache pre model s `n_layers = 40`, `n_kv_heads = 8`, `d_head = 128`, fp16, pri kontexte 32 000 tokenov.
22. Aplikácia posiela do modelu na začiatok promptu aktuálny čas. Prečo je to drahé a ako to opraviť?
23. Prečo generovanie 500-tokenovej odpovede trvá skoro rovnako dlho bez ohľadu na to, či mal prompt 200 alebo 2000 tokenov — a čo sa zmení, keď má 100 000?
24. Prečo je generovanie tokenov limitované priepustnosťou pamäte a nie výkonom GPU? Odvoďte z toho, prečo funguje špekulatívne dekódovanie aj prečo je dávkovanie „zadarmo".
25. Model deklaruje okno 200k tokenov, ale pri 150k odpovedá horšie. Vymenujte dve nezávislé príčiny.
26. Reasoning model dostal `max_tokens = 500` a vrátil prázdnu odpoveď. Čo sa stalo?
27. Do promptu vložíte tri príklady „vstup → kategória" a model začne kategorizovať správne. Zmenila sa tým čo i len jedna váha modelu? Čo sa teda zmenilo a kde to po skončení volania skončí?
28. Čo je indukčná hlava a ako vysvetľuje, prečo few-shot príklady fungujú?
29. Používateľ tvrdí: „náš chatbot si pamätá, čo sme mu povedali minulý týždeň — takže sa učí." Vysvetlite, čo sa v skutočnosti deje.
30. Model má 14 miliárd parametrov. Koľko VRAM potrebujú jeho váhy v `fp16` a koľko v `int4`? Prečo môže byť takýto kvantizovaný model lepšou voľbou než 7B model v `fp16` na tej istej karte?
31. Kolega otestoval 4-bitovú kvantizáciu na desiatich krátkych anglických otázkach a nenašiel rozdiel. Prečo to nestačí a kde by sa strata prejavila skôr?

---

## Súvisiace dokumenty

- [01-transformer-siete.md](01-transformer-siete.md) — **predchádzajúci**: mechanizmus attention (Q, K, V, multi-head, maska)
- [03-llm-trening.md](03-llm-trening.md) — **nasledujúci**: ako sa tieto váhy natrénujú
- [05-embeddings.md](05-embeddings.md) — BPE tréning slovníka krok po kroku a ten istý priechod vrstvou prepočítaný ručne na číslach (lekcia 6)
- [06-rag.md](06-rag.md) — ako sa obmedzenému kontextu vyhnúť vyhľadávaním (lekcia 6)
- [04-llm-modely.md](04-llm-modely.md) — konkrétne modely, ktorých čísla (`A22B`, kontext, cena) teraz viete čítať (lekcia 5)
- [01-vyvojove-prostredie.md](../00-prostredie/01-vyvojove-prostredie.md) — koľko GPU pamäte to celé potrebuje
- [05-llm-trendy.md](../05-prakticke/05-llm-trendy.md) — kam sa posúva hranica dlhého kontextu
- [02-agenti-a-nastroje.md](../05-prakticke/02-agenti-a-nastroje.md) — prečo agent na počítanie volá kalkulačku a ako mu história zväčšuje `n`

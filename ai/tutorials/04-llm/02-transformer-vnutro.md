# Vnútro transformera — rozmery, feed-forward, výstupný token a KV cache

> **Poradie čítania:** ← [Transformery a attention](01-transformer-siete.md) · **lekcia 4** · [Ako sa trénuje LLM](03-llm-trening.md) →

> **Cieľ dokumentu:** dopovedať to, čo [predchádzajúci dokument](01-transformer-siete.md) nechal na úrovni myšlienky — **konkrétne rozmery a čísla**. Odkiaľ sa berie veľkosť vektorov, čo presne sa deje s každým vektorom vo feed-forward vrstve, ako z posledného vektora vznikne **jeden konkrétny token**, prečo je kontextové okno **obmedzené** a čo sa pri generovaní ďalšieho tokenu dá **cachovať** (a čo nie).

Predchádzajúci dokument vysvetlil **mechanizmus** (Q, K, V, softmax, multi-head, maska). Tento vysvetľuje **inžinierstvo okolo neho**: prečo má model práve 4096-rozmerné vektory, prečo 32 vrstiev, koľko pamäte zožerie kontext 100 000 tokenov a prečo je prvý token odpovede pomalý a ďalšie rýchle.

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

## 1. Odkiaľ sa berie veľkosť vektorov

Vektor tokenu má vo vnútri modelu **stále rovnakú dĺžku `d_model`** — od embedding vrstvy až po poslednú vrstvu. Attention aj feed-forward vrstva ho dočasne premietnu do iného rozmeru, ale na výstupe bloku je vždy zase `d_model`. Práve preto sa dajú bloky ukladať na seba do ľubovoľnej hĺbky.

`d_model` **nie je vypočítané z ničoho** — je to voľba návrhára modelu, hyperparameter. Riadi sa štyrmi pravidlami:

1. **Kapacita.** Väčšie `d_model` = viac miesta na informáciu v jednom tokene. Rastie s veľkosťou modelu: 768 (GPT-2 small) → 4096 (8B) → 8192 (70B).
2. **Deliteľnosť hlavami.** Musí platiť `d_model = n_heads · d_head`, pričom `d_head` býva **64 alebo 128** — empiricky najlepší kompromis medzi „hlava má dosť miesta" a „hláv je dosť veľa".
3. **Hardvér.** Rozmery sú násobky 64/128, aby maticové násobenia sadli na tensor cores GPU. `d_model = 4000` by bežalo citeľne pomalšie ako 4096.
4. **Pomer hĺbka : šírka.** Neplatí „radšej hlbšie" ani „radšej širšie" — *scaling laws* ukazujú, že pre daný počet parametrov existuje optimálny pomer. Pomer `d_model / n_layers` býva rádovo **60–130** (GPT-2 small 64, Llama 3 70B 102, Llama 3 8B 128).

A `d_ff` (šírka feed-forward vrstvy) je tradične **4 × `d_model`**. Pri modernej aktivácii SwiGLU sú v FFN **tri** matice namiesto dvoch, takže sa `d_ff` znižuje na ≈ `8/3 × d_model`, aby počet parametrov ostal rovnaký (Llama 2 7B: 11008 pri `d_model` 4096). Novšie modely idú aj nad toto pravidlo — Llama 3 8B má `d_ff = 14336`, teda zámerne širšie FFN na úkor iných rozmerov.

| Model | `d_model` | `n_layers` | `n_heads` | `d_ff` | `vocab` | parametre |
|---|---|---|---|---|---|---|
| GPT-2 small | 768 | 12 | 12 | 3072 | 50 257 | 124 M |
| GPT-2 XL | 1600 | 48 | 25 | 6400 | 50 257 | 1.5 B |
| Llama 3 8B | 4096 | 32 | 32 | 14336 | 128 256 | 8 B |
| Llama 3 70B | 8192 | 80 | 64 | 28672 | 128 256 | 70 B |

### Kde v modeli sedia parametre

Spočítajme referenčný model — je to obyčajné sčítanie veľkostí matíc a hneď z neho vidno, kde sa „míňa" kapacita:

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

- **Feed-forward vrstvy zaberajú ~80 % parametrov jednej vrstvy** (176 M z 218 M) a **~70 % celého modelu** (zvyšok ukroja embeddingy) — nie attention. Attention je to zaujímavé, ale FFN je to veľké.
- **Embedding matice sú netriviálne** — pri malých modeloch dokonca dominujú (GPT-2 small: 39 M zo 124 M). Preto sa často **zdieľajú** (*weight tying*): tá istá matica sa použije na vstupe aj na výstupe.

---

## 2. Cesta jedného vektora jednou vrstvou

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

> **Pre-norm vs. post-norm — v čom sa dnešné modely líšia od článku z 2017.** Pôvodný transformer normalizoval **až po** bloku: `x = Norm(x + attention(x))` (*post-norm*). Dnešné LLM normalizujú **pred** blokom a rezíduum normalizáciu obchádza: `x = x + attention(Norm(x))` (*pre-norm*) — hlboké modely sa tak trénujú stabilnejšie, lebo cez reziduálnu vetvu tečie gradient nedotknutý. Preto má pre-norm model na konci ešte jednu **finálnu normalizáciu** pred `lm_head` (sekcia 4). Tento dokument aj obrázok používajú pre-norm; schéma v [lekcii 4](01-transformer-siete.md#celková-architektúra) a ručný prepočet v [lekcii 6](05-embeddings.md#krok-3-transformer-vrstvy--tu-sa-deje-pochopenie-kontextu) ukazujú pôvodný post-norm. **Pri vlastnej implementácii si vyberte jedno a držte sa toho.**

Kľúčové pozorovanie, ktoré vysvetľuje celý zvyšok dokumentu:

> **Attention je jediné miesto, kde si tokeny navzájom vymieňajú informáciu.** Všetko ostatné vo vrstve — normalizácia, feed-forward, reziduá — beží nad **každým tokenom nezávisle**. Attention = komunikácia medzi tokenmi, FFN = spracovanie vnútri tokenu.

Odtiaľ plynie aj rozdelenie nákladov: attention rastie s **druhou mocninou** počtu tokenov (matica `n × n`), FFN len **lineárne** (n riadkov, každý rovnako drahý).

---

## 3. Feed-forward vrstva: každý vektor sám za seba

Po attention má každý token vektor obohatený o kontext. Feed-forward vrstva je obyčajný dvojvrstvový [MLP](../02-typy-modelov/04-feed-forward-siete.md), ktorý sa naň aplikuje — a **na každý token sa aplikujú tie isté váhy**:

```text
klasicky (GPT-2 štýl, tu v našich rozmeroch — samotné GPT-2 small má 768 → 3072 → 768):
                    FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2
                             4096 → 16384 → 4096

moderne (SwiGLU):   FFN(x) = W_down · ( SiLU(W_gate · x) ⊙ (W_up · x) )
                             4096 → 14336 (dve vetvy, násobené po zložkách) → 4096
```

Tri veci, ktoré tu študenti najčastejšie prehliadnu:

**a) Je to naozaj token po tokene.** Na [obrázku vyššie](#2-cesta-jedného-vektora-jednou-vrstvou) sú to tri oddelené dráhy s ✕ medzi nimi. Pri `n = 1000` tokenoch sa tá istá matica `W_1` použije 1000-krát na 1000 rôznych vektorov. Nič sa medzi tokenmi nemieša. Implementačne sa to spraví jedným maticovým násobením `[1000, 4096] × [4096, 14336]`, ale sémanticky sú to 1000 nezávislých priechodov. Preto sa FFN dá triviálne paralelizovať a preto je vo fáze generovania (jeden token) veľmi „tenká".

**b) Rozšírenie a zúženie má zmysel.** Vrstva najprv vektor **rozšíri** (4096 → 14336), pustí cez nelinearitu a potom **zúži** späť. Bez rozšírenia by nelinearita mala málo priestoru; bez zúženia by sa rozmer po každej vrstve zväčšoval a bloky by sa nedali skladať.

**c) Toto je pamäť modelu na fakty.** Interpretačné práce ukazujú, že FFN sa dá čítať ako **key-value pamäť**: prvá matica rozhodne „na čo tento vektor vyzerá", nelinearita to prahuje, druhá matica pripočíta zodpovedajúcu informáciu. Faktické znalosti („Bratislava je hlavné mesto Slovenska") sedia hlavne tu — čo je konzistentné s tým, že FFN drží 80 % parametrov vrstvy.

> Ručne prepočítaný priechod attention + reziduum + LayerNorm + FFN na 4-rozmerných vektoroch je v [05-embeddings.md, Krok 3](05-embeddings.md#krok-3-transformer-vrstvy--tu-sa-deje-pochopenie-kontextu).

---

## 4. Ako vznikne výstupný token

Po poslednej (32.) vrstve máme stále `n` vektorov po 4096 čísel. Pri generovaní nás zaujíma **len ten posledný** — je to jediný, ktorý „videl" celý kontext a má predpovedať, čo nasleduje.

```text
h  [4096]                      posledný vektor poslednej vrstvy
│
├─► final RMSNorm              [4096]
│
├─► lm_head:  h · W_U          W_U má tvar [4096, 128 256]
│                              ↓
│   logity  [128 256]          jedno reálne číslo pre KAŽDÝ token slovníka
│                              napr.  "Bratislava" 18.3
│                                     "Praha"       9.1
│                                     "hlavné"      7.4  …
│
├─► (voliteľne) úpravy logitov: teplota, repetition penalty, top-k / top-p, zákaz tokenov
│
├─► softmax                    [128 256] pravdepodobností so súčtom 1
│                              "Bratislava" 0.93, "Praha" 0.01, …
│
└─► výber jedného tokenu       → id 43217 → detokenizácia → " Bratislava"
```

Poznámky, ktoré sa oplatí vedieť:

- **Výstupná matica je obrovská.** Jedno násobenie `4096 × 128 256` = 525 M operácií len na to, aby vznikol jeden token. Pri veľkých slovníkoch je to citeľná časť nákladov na generovanie.
- **Logity nie sú pravdepodobnosti.** Sú to ľubovoľné reálne čísla; pravdepodobnosti z nich robí až softmax. Všetky triky s dekódovaním (teplota, top-p) sa robia **na logitoch alebo tesne po softmaxe** — nie v modeli. Model je deterministický; náhoda je až vo výbere.
- **Pri tréningu sa počíta výstup pre všetky pozície naraz** (každý token predpovedá ten nasledujúci — odtiaľ „self-supervised"), pri inferencii len pre poslednú. Detaily tréningu sú v [03-llm-trening.md](03-llm-trening.md).
- **Sampling** (greedy / teplota / top-p / top-k) je rozpísaný v [predchádzajúcom dokumente](01-transformer-siete.md#ako-presne-sa-vyberá-ďalší-token-dekódovanie).

---

## 5. Ďalší prechod: autoregresia a čo sa dá cachovať

Vybraný token sa pripojí na koniec sekvencie a celé sa to opakuje. Naivne by to znamenalo: pri generovaní 500. tokenu prepočítať celý model nad 500 tokenmi. To by bolo kvadratické plytvanie — a **nie je to potrebné**.

Kľúčové pozorovanie: **vďaka kauzálnej maske nový token nemení nič, čo už bolo spočítané.** Token č. 500 nemôže ovplyvniť vektory tokenov 1–499, lebo tie sa naň nesmú pozerať. Ich `K` a `V` vektory sú teda **navždy platné** a stačí si ich odložiť.

### KV cache

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

**Čo sa cachuje:** `K` a `V` každého tokenu, v každej vrstve, pre každú KV hlavu.
**Čo sa necachuje:** `Q` (nový token má vlastné a staré už netreba), aktivácie FFN (pre nový token sú vždy nové), attention váhy (závisia od nového `Q`).

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

### Prečo je prvý token pomalý a ďalšie rýchle

| | Prefill | Decode (1 token) |
|---|---|---|
| Koľko tokenov vstúpi | `n` | 1 |
| Attention náklad | `O(n²)` | `O(n)` proti cache |
| Čo limituje | **výpočet** (GPU FLOPs) | **priepustnosť pamäte** |
| Typicky (jedna bežná GPU) | 1 000 tokenov ≈ 0.2 s | 15–20 ms/token |

Decode je **memory-bound**, a to je najdôležitejšia praktická vec na tejto stránke: pre každý jediný vygenerovaný token musí GPU prečítať **všetkých 8 miliárd váh** (16 GB v fp16) z pamäte. Pri priepustnosti ~1 TB/s to je ~16 ms na token, teda ~60 tokenov/s — **bez ohľadu na to, aký máte rýchly čip**. Odtiaľ dva dôsledky: kvantizácia na 8/4 bity zrýchľuje generovanie priamo úmerne (menej bajtov na prečítanie) a **dávkovanie (batching)** je zadarmo — tie isté prečítané váhy obslúžia 30 používateľov naraz.

### Cachovanie o úroveň vyššie: prompt caching

KV cache žije v rámci jednej požiadavky. Ak však posielate **ten istý prefix** znova a znova (systémový prompt, dlhý dokument, definície nástrojov), dá sa uložiť aj medzi požiadavkami — API to ponúkajú ako **prompt caching** (Anthropic `cache_control`, OpenAI automaticky). Zápis do cache stojí o niečo viac, čítanie z nej rádovo menej než bežné vstupné tokeny, a preskočí sa prefill.

Podmienka je jediná, ale nekompromisná: **prefix sa musí zhodovať bajt na bajt**. Z toho vyplýva pravidlo skladania promptov, ktoré platí aj pri agentoch aj pri RAG:

```text
✅ stabilné veci DOPREDU:   systémový prompt → definície nástrojov → dokumenty → história → otázka
❌ meniace sa veci dozadu:  timestamp alebo ID používateľa na začiatku promptu zabije cache celej konverzácie
```

Ďalšie úrovne cachovania, s ktorými sa v praxi stretnete:

| Úroveň | Čo sa ukladá | Kde |
|---|---|---|
| **KV cache** | K/V vektory tokenov | GPU pamäť, v rámci jednej generácie |
| **Prompt / prefix cache** | KV cache spoločného prefixu | server modelu, medzi požiadavkami (vLLM, Anthropic, OpenAI) |
| **PagedAttention** | KV cache v stránkach ako virtuálna pamäť | vLLM — umožní zdieľať prefix medzi používateľmi bez kopírovania |
| **Embedding cache** | vektory už zaindexovaných chunkov | vektorová DB — [RAG](06-rag.md), aby sa embedding nepočítal znova |
| **Response cache** | celé odpovede na rovnaké otázky | aplikačná vrstva (Redis a pod.) |

---

## 6. Prečo je kontextové okno obmedzené

Kontextové okno **nie je** softvérové obmedzenie, ktoré by sa dalo „odomknúť". Je to súčet štyroch nezávislých stien:

**1. Attention je kvadratická.** Matica skóre má `n × n` prvkov v každej hlave a každej vrstve. Prechod z 1 000 na 10 000 tokenov znamená **100× viac** práce v attention. Pri 100 000 tokenoch je to `10⁴×`: jedna matica váh má `(10⁵)² = 10¹⁰` prvkov a takých matíc je 32 vrstiev × 32 hláv — rádovo `10¹³` čísel len na attention váhy. (Preto ich FlashAttention nikdy nezhmotní naraz — počíta ich po dlaždiciach a priebežne zahadzuje.)

**2. KV cache rastie lineárne, ale rýchlo.** Podľa výpočtu vyššie: 128 KiB na token. Kontext 128k tokenov = 16 GB **navyše k modelu**, na každého používateľa zvlášť. Na serveri s 80 GB GPU to znamená, že buď obslúžite veľa krátkych konverzácií, alebo štyri dlhé — nie oboje.

**3. Model bol trénovaný na určitú dĺžku.** Pozičné kódovanie (RoPE) je funkcia, ktorú model videl len na pozíciách 0…`n_ctx`. Za touto hranicou extrapoluje do neznáma a kvalita spadne. Rozšírenie okna preto vyžaduje **doučenie** (RoPE scaling / YaRN a dotrénovanie na dlhých textoch) — nie zmenu konštanty v konfiguráku.

**4. Kvalita klesá skôr než hranica.** Aj keď model formálne zvládne 200k tokenov, presnosť práce s informáciou uprostred dlhého kontextu je preukázateľne horšia než na začiatku a na konci (*lost in the middle*). Deklarované okno je horná hranica, nie odporúčaná pracovná dĺžka.

A k tomu **cena**: vstupné tokeny sa platia. 100 000 tokenov v každej otázke je pri agentovi, ktorý sa pýta stokrát, reálny účet — presne preto existuje [RAG](06-rag.md), ktorý do promptu vloží 5 relevantných odsekov namiesto celej dokumentácie.

### Čo s tým robia moderné modely

| Technika | Čo rieši | Ako |
|---|---|---|
| **FlashAttention** | pamäť pri prefille | nikdy nezmaterializuje celú maticu `n×n`, počíta ju po dlaždiciach; matematicky identický výsledok |
| **GQA / MQA** | veľkosť KV cache | menej K/V hláv (8 namiesto 32) → 4× menšia cache |
| **Sliding window** | kvadratika | token vidí len posledných napr. 4096 tokenov, vzdialenejšie sprostredkovane cez vrstvy |
| **RoPE scaling / YaRN** | tréningová dĺžka | preškáluje pozičné frekvencie + krátke dotrénovanie → z 8k spraví 128k |
| **PagedAttention** | fragmentácia pamäte | KV cache po stránkach, zdieľanie prefixu medzi požiadavkami |
| **Kvantizácia KV cache** | veľkosť cache | K/V v 8 bitoch namiesto 16 → polovičná cache |

---

## 7. Prehľad parametrov transformera

Slovo „parameter" znamená v kontexte LLM tri rôzne veci — oplatí sa ich nemiešať.

### a) Architektonické hyperparametre (v `config.json` modelu, meniť sa nedajú)

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
| RoPE báza | `rope_theta` | 10 000 – 500 000 | rozsah pozícií |
| norm epsilon | `rms_norm_eps` | 1e-5 | numerickú stabilitu |
| zdieľanie embeddingov | `tie_word_embeddings` | true/false | veľkosť malých modelov |

### b) Naučené váhy (to, čo tréning nastavuje — „8 miliárd parametrov")

Embedding matica, `W_Q/W_K/W_V/W_O` v každej vrstve, tri FFN matice v každej vrstve, škálovacie parametre noriem, výstupná `W_U`. Nič iné v modeli nie je.

### c) Inferenčné parametre (nastavujete pri každom volaní)

| Parameter | Čo robí | Kedy meniť |
|---|---|---|
| `temperature` | plochosť rozdelenia pred výberom | 0–0.3 fakty, 0.7–1.0 kreatíva |
| `top_p` / `top_k` | orezanie chvosta rozdelenia | 0.9 / 40 ako rozumný default |
| `repetition_penalty`, `presence/frequency_penalty` | trestá opakovanie | keď sa model zacyklí |
| `max_new_tokens` | strop dĺžky odpovede | vždy — chráni pred útekom |
| `stop` sekvencie | kde skončiť | pri štruktúrovanom výstupe |
| `seed` | zopakovateľnosť samplovania | pri testovaní |
| `dtype` / kvantizácia | fp16, int8, int4 | pamäť vs. kvalita |

Do tejto skupiny patria aj [dekódovacie stratégie](01-transformer-siete.md#ako-presne-sa-vyberá-ďalší-token-dekódovanie) — a stojí za zopakovanie, že **žiadny z týchto parametrov nemení váhy modelu**. Menia len to, ako sa z logitov vyberá token.

---

## Zhrnutie

| Otázka | Odpoveď v jednej vete |
|---|---|
| Odkiaľ je veľkosť vektorov? | `d_model` je voľba návrhára: `n_heads × d_head` (64/128), násobok 128 kvôli GPU, `d_ff ≈ 4× d_model`. |
| Kde sedia parametre? | ~80 % vrstvy (a ~70 % modelu) vo feed-forward, zvyšok v attention a embeddingoch. |
| Ako ide vektor do FFN? | Každý token zvlášť, tie isté váhy: rozšír (4096→14336), nelinearita, zúž späť. |
| Kde sa tokeny miešajú? | **Iba v attention.** Norm, FFN aj reziduá bežia per token. |
| Ako vznikne token? | Posledný vektor → norm → `lm_head` (4096×128k) → logity → úpravy → softmax → výber. |
| Prečo je prvý token pomalý? | Prefill je `O(n²)` a compute-bound; decode je `O(n)` a memory-bound. |
| Čo sa cachuje? | `K` a `V` každého tokenu v každej vrstve; `Q` a aktivácie FFN nie. |
| Prečo je kontext obmedzený? | Kvadratická attention + veľkosť KV cache + tréningová dĺžka RoPE + pokles kvality. |

---

## Kontrolné otázky

1. Model má `d_model = 4096` a `n_heads = 32`. Aký je rozmer jednej hlavy a prečo nemôže byť `n_heads = 30`?
2. Prečo sa v modeli s 8 miliardami parametrov nachádza väčšina váh vo feed-forward vrstvách a nie v attention? Spočítajte to pre jednu vrstvu.
3. Vysvetlite, prečo sa `K` a `V` dajú cachovať, ale `Q` nie. Čo konkrétne to umožňuje — ktorá vlastnosť decoder-only modelu?
4. Spočítajte KV cache pre model s `n_layers = 40`, `n_kv_heads = 8`, `d_head = 128`, fp16, pri kontexte 32 000 tokenov.
5. Aplikácia posiela do modelu na začiatok promptu aktuálny čas. Prečo je to drahé a ako to opraviť?
6. Prečo generovanie 500-tokenovej odpovede trvá skoro rovnako dlho bez ohľadu na to, či mal prompt 200 alebo 2000 tokenov — a čo sa zmení, keď má 100 000?
7. Model deklaruje okno 200k tokenov, ale pri 150k odpovedá horšie. Vymenujte dve nezávislé príčiny.
8. Prečo je generovanie tokenov limitované priepustnosťou pamäte a nie výkonom GPU? Čo z toho vyplýva pre kvantizáciu a pre dávkovanie požiadaviek?

---

### Súvisiace dokumenty

- [01-transformer-siete.md](01-transformer-siete.md) — **predchádzajúci**: mechanizmus attention (Q, K, V, multi-head, maska)
- [03-llm-trening.md](03-llm-trening.md) — **nasledujúci**: ako sa tieto váhy natrénujú
- [05-embeddings.md](05-embeddings.md) — ten istý priechod vrstvou prepočítaný ručne na číslach (lekcia 6)
- [06-rag.md](06-rag.md) — ako sa obmedzenému kontextu vyhnúť vyhľadávaním (lekcia 6)
- [01-vyvojove-prostredie.md](../00-prostredie/01-vyvojove-prostredie.md) — koľko GPU pamäte to celé potrebuje
- [02-llm-trendy.md](../05-prakticke/02-llm-trendy.md) — kam sa posúva hranica dlhého kontextu

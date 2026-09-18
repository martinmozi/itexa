# Tréning transformera — Adam v akcii a čo do hry vnáša attention

> **Poradie čítania:** ← [Vnútro transformera](../04-llm/02-transformer-vnutro.md) · **lekcia 4** · [Ako sa trénuje LLM](../04-llm/03-llm-trening.md) →

> **Cieľ dokumentu:** ukázať tréningovú slučku na transformeri — a hlavne to, **čo je na nej iné** oproti feed-forward sieti z [01-adam-optimalizator.md](01-adam-optimalizator.md). Odpoveď je v jednom slove: **attention**. Nie je to len ďalšia vrstva; mení tvar gradientu, vynucuje si warmup, diktuje škálovanie `1/√d_head`, rozhoduje o spotrebe pamäte a je príčinou väčšiny vecí, ktoré sa pri tréningu LLM pokazia.

**Tento dokument predpokladá tri veci:** tréningovú slučku a AdamW z [01-adam-optimalizator.md](01-adam-optimalizator.md), mechanizmus attention z [01-transformer-siete.md](../04-llm/01-transformer-siete.md) a rozmery (`d_model`, `n_heads`, `d_head`, `d_ff`) z [02-transformer-vnutro.md](../04-llm/02-transformer-vnutro.md). Naopak **nerozoberá** fázy tréningu LLM (pretraining → SFT → RLHF) — tie sú v [03-llm-trening.md](../04-llm/03-llm-trening.md). Tu ide o jeden krok optimalizátora a o to, čo sa v ňom deje.

> **Prečo je tento dokument v kapitole o tréningu, a nie medzi LLM:** je to tá istá slučka `forward → loss → backprop → Adam` ako pri MLP a pri stromoch ide o presne tú istú myšlienku gradientu. Kapitola 03 drží pokope **remeslo tréningu**; kapitola 04 architektúru a použitie modelov.

---

## 1. Čo sa v transformeri vlastne trénuje

Než pustíme optimalizátor, treba vedieť, na aké parametre siaha. V jednom bloku transformera sú štyri skupiny:

| Skupina | Matice | Tvar | Poznámka |
|---|---|---|---|
| **Attention** | `W_Q`, `W_K`, `W_V` | `[d_model, d_model]` každá | rozdelené medzi `n_heads` hláv po `d_head` |
| | `W_O` (výstupná projekcia) | `[d_model, d_model]` | zlepí výstupy hláv späť do jedného vektora |
| **Feed-forward** | `W_1`, `W_2` (pri SwiGLU aj `W_3`) | `[d_model, d_ff]`, `[d_ff, d_model]` | **väčšina parametrov bloku** |
| **Normalizácia** | `gain` (RMSNorm) | `[d_model]` | zanedbateľný počet, veľký vplyv |
| **Mimo blokov** | embedding, unembedding | `[vocab, d_model]` | pri malých modeloch až polovica parametrov |

Odtiaľ pochádza známy odhad `N ≈ 12 · n_layers · d_model²`: štyri attention matice po `d_model²` plus feed-forward `2 · 4 · d_model²` (pri `d_ff = 4·d_model`) dá 12 na blok.

Dve veci, ktoré prekvapia:

- **Attention nemá žiadne parametre pre pozíciu.** RoPE (rotačné polohové kódovanie) je len otočenie vektorov, nie naučená matica — nie je čo trénovať. Model sa naučí pracovať s poradím **výhradne cez `W_Q` a `W_K`**.
- **Softmax v attention nemá parametre ani on.** A predsa je to miesto, ktoré rozhoduje, či sa model bude učiť (sekcia 3).

V kóde si parametre pozriete takto — a pri ladení sa to naozaj oplatí:

```python
from collections import defaultdict

sucty = defaultdict(int)
for meno, p in model.named_parameters():
    if not p.requires_grad:
        continue
    skupina = ("embedding" if "embed" in meno or "lm_head" in meno
               else "attention" if any(k in meno for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
               else "feed-forward" if "mlp" in meno
               else "norm")
    sucty[skupina] += p.numel()

celkom = sum(sucty.values())
for k, v in sorted(sucty.items(), key=lambda x: -x[1]):
    print(f"{k:14s} {v/1e6:8.1f}M  ({v/celkom:5.1%})")
```

---

## 2. Jeden tréningový krok, celý

Slučka je **presne tá istá** ako pri MLP; líši sa tvar dát a to, čo je loss.

```python
import torch, torch.nn.functional as F

for krok, batch in enumerate(loader):
    # batch["input_ids"]: [B, T]  — B sekvencií po T tokenov
    x = batch["input_ids"].to(device)

    # 1) FORWARD — model vráti logity pre KAŽDÚ pozíciu naraz
    logity = model(x)                     # [B, T, vocab]

    # 2) LOSS — predpoveď tokenu t+1 z pozície t (posun o jedna)
    strata = F.cross_entropy(
        logity[:, :-1, :].reshape(-1, logity.size(-1)),   # [(B·(T-1)), vocab]
        x[:, 1:].reshape(-1),                             # [(B·(T-1))]
        ignore_index=-100,                                # padding sa nepočíta
    )

    # 3) BACKWARD
    (strata / akumulacia).backward()

    if (krok + 1) % akumulacia == 0:
        # 4) ORezanie gradientu — pri transformeri povinná výbava
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 5) KROK OPTIMALIZÁTORA a rozvrhu learning rate
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

Tri poznámky k tomu, čo je tu iné než pri obrázkovej sieti:

1. **Jeden forward = `B · (T−1)` tréningových príkladov.** Model predpovedá ďalší token na každej pozícii súčasne a maska sa stará o to, aby pri pozícii `t` nevidel budúcnosť. Toto sa volá **teacher forcing** a podrobne je v [03-llm-trening.md](../04-llm/03-llm-trening.md). Dôsledok pre tréning: „veľkosť batchu" sa pri LLM **neudáva v riadkoch, ale v tokenoch** (`B · T`) — 0,5M tokenov na krok je bežná hodnota, GPT-3 išiel až na 3,2M.
2. **Gradient akumulácia** (`akumulacia` krokov pred jedným `step()`) je tu pravidlom, nie výnimkou: cieľová veľkosť batchu sa do pamäte nezmestí, tak sa poskladá z viacerých menších priechodov. Matematicky je to to isté, časovo je to pomalšie.
3. **Clipping na normu 1,0** nie je poistka do zlých čias — pri transformeroch beží stále. Prečo, ukazuje sekcia 6.

---

## 3. Kde presne do toho vstupuje attention

Toto je jadro dokumentu. Attention mení tréning na troch miestach.

### 3.1 Škálovanie `1/√d_head` je tréningové opatrenie

Pripomeňme vzorec z [01-transformer-siete.md](../04-llm/01-transformer-siete.md):

```text
    A = softmax( Q·Kᵀ / √d_head ) · V
```

Ten deliteľ sa často podá ako technický detail. Nie je. Ak sú zložky `q` a `k` nezávislé s rozptylom 1, skalárny súčin `q·k` má rozptyl **`d_head`** — pri `d_head = 128` teda hodnoty rádovo ±11 namiesto ±1. A softmax z takých čísel je takmer **one-hot**: jedna pozícia dostane váhu 0,999, ostatné nuly.

Čo to urobí s gradientom? Derivácia softmaxu je `s·(1−s)`. Pri `s ≈ 0,999` aj `s ≈ 0,001` je to prakticky **nula**:

| stav softmaxu | najväčšia váha | `s·(1−s)` | gradient do `W_Q`, `W_K` |
|---|---|---|---|
| zdravý (po škálovaní) | 0,1 – 0,4 | 0,09 – 0,24 | **tečie** |
| saturovaný (bez škálovania) | 0,999 | 0,001 | **prakticky nulový** |

Model by teda hneď na začiatku „zamrzol" v náhodne zvolenom vzore pozornosti a nikdy by sa z neho nedostal — je to presne ten istý mechanizmus ako **miznúce gradienty cez sigmoid**, popísaný v [02-problemy-pri-uceni.md](02-problemy-pri-uceni.md). Delenie `√d_head` drží skalárne súčiny v rozsahu, kde softmax ešte má deriváciu.

> Rovnaký jav sa vracia počas tréningu pod menom **attention entropy collapse**: ak sa logity attention rozbehnú do veľkých hodnôt, entropia rozdelenia klesne takmer na nulu, hlavy sa „zaseknú" na jednej pozícii a loss prestane klesať. Lieči sa to znížením learning rate, normalizáciou Q a K (**QK-norm**, dnes bežná v nových modeloch) alebo silnejším clippingom.

### 3.2 Gradient do jednej váhy prichádza z celej sekvencie

V MLP dostane váha gradient od každého riadku v batchi. V transformeri dostane `W_Q` gradient od **každej pozície každej sekvencie** — a navyše cez attention aj od všetkých pozícií, ktoré na ňu pozerali. Pri `B = 8` a `T = 4096` je to 32 768 príspevkov do jednej matice v jedinom kroku.

Dôsledky sú veľmi praktické:

- **Gradient je relatívne stabilný** (veľa vzoriek sa priemeruje), ale jeho **veľkosť veľmi kolíše** medzi krokmi, keď v dátach príde nezvyklý batch. Preto `β₂` (pamäť druhého momentu) pri LLM klesá z obvyklých 0,999 na **0,95**: kratšia pamäť znamená, že sa Adam po skoku rýchlejšie spamätá.
- **Dlhšia sekvencia = iný efektívny batch.** Ak zdvojnásobíte `T`, zdvojnásobíte počet tréningových príkladov na krok, hoci „počet riadkov" zostal rovnaký. Preto sa learning rate ladí na počet tokenov, nie na počet sekvencií.

### 3.3 Reziduálny prúd je diaľnica pre gradient

Blok transformera nepočíta `x ← f(x)`, ale `x ← x + f(x)`. Derivácia takého spojenia je `1 + f′(x)`, takže gradient má **vždy priechodnú cestu** aj vtedy, keď `f′` je malé. Bez toho by sa 80-vrstvový model netrénoval vôbec — presne ten istý argument ako pri ResNetoch v [02-problemy-pri-uceni.md](02-problemy-pri-uceni.md).

S tým súvisí jedna architektonická voľba, ktorá je v skutočnosti **rozhodnutie o tréningu**: kam patrí normalizácia.

```text
   post-LN (pôvodný transformer, 2017)     pre-LN (dnešný štandard)

   x ─┬──────────────┐                     x ─┬──────────────┐
      │              │                        │              │
      ▼              │                        ▼              │
   attention         │                     RMSNorm           │
      │              │                        │              │
      ▼              │                        ▼              │
     (+)◄────────────┘                     attention         │
      │                                       │              │
      ▼                                      (+)◄────────────┘
   LayerNorm                                  │
      │                                       ▼
```

Pri **post-LN** prechádza gradient pri ceste späť cez normalizáciu v každej vrstve a jeho veľkosť sa zmenšuje — hlboký model bez opatrnej rozcvičky diverguje. Pri **pre-LN** ide reziduálna vetva úplne čistá, gradient preteká bez zásahu a tréning je podstatne odolnejší. Preto je dnes pre-LN (spravidla s RMSNorm) vo všetkých veľkých modeloch.

---

## 4. Prečo je pri transformeri Adam prakticky povinný

Pri malej sieti sa dá trénovať aj obyčajným SGD s momentom. Pri transformeri to prakticky nikto nerobí a dôvod je konkrétny: **rôzne skupiny parametrov majú rádovo rôzne gradienty.**

| Parameter | Typická veľkosť gradientu | Prečo |
|---|---|---|
| embedding riadok bežného tokenu | veľmi malý | token sa v batchi objaví párkrát |
| embedding riadok častého tokenu (`the`, medzera) | veľký | objaví sa v každej sekvencii stokrát |
| `W_Q`, `W_K` | stredný, kolísavý | prechádza softmaxom |
| `gain` v RMSNorm | veľký | jeden skalár na celý rozmer |

Jeden globálny learning rate to neuhrá: hodnota, pri ktorej sa pohnú embeddingy zriedkavých tokenov, rozstrelí normalizačné parametre. **Adam delí krok odmocninou druhého momentu, čím dá každému parametru vlastnú mierku** — a práve to robí tréning vôbec možným. Mechanika je v [01-adam-optimalizator.md](01-adam-optimalizator.md), sekcia 3.

Nastavenie, ktoré sa pri transformeroch ustálilo:

```python
# weight decay patrí LEN na maticové váhy — nie na bias, normy a embeddingy
decay, no_decay = [], []
for meno, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (decay if p.dim() >= 2 else no_decay).append(p)

optimizer = torch.optim.AdamW(
    [{"params": decay,    "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4,
    betas=(0.9, 0.95),      # β₂ nižšie než obvyklých 0,999
    eps=1e-8,               # pri bf16 tréningu sa používa aj 1e-5
    fused=True,             # rýchlejší CUDA kernel
)
```

Prečo `p.dim() >= 2`? Weight decay má ťahať k nule **maticové** váhy, kde to znamená regularizáciu. Pri normalizačnom `gain` (vektor) alebo biase by ťahanie k nule model len poškodilo — `gain = 0` znamená vypnutú vrstvu. Tento dvojriadkový trik je v každej serióznej tréningovej implementácii.

Orientačné hodnoty z publikovaných modelov (aby ste vedeli, čo je bežné):

| Model | peak lr | batch (tokeny) | β₁, β₂ | weight decay | clip | rozvrh |
|---|---|---|---|---|---|---|
| GPT-3 175B | 0,6 · 10⁻⁴ | 3,2 M | 0,9 / 0,95 | 0,1 | 1,0 | warmup ~375M tokenov → kosínus na 10 % |
| Llama 2 7B | 3 · 10⁻⁴ | 4 M | 0,9 / 0,95 | 0,1 | 1,0 | warmup 2000 krokov → kosínus na 10 % |
| Llama 2 70B | 1,5 · 10⁻⁴ | 4 M | 0,9 / 0,95 | 0,1 | 1,0 | rovnako |

Vidno pravidlo, ktoré platí naprieč: **čím väčší model, tým menší learning rate** (zhruba nepriamo úmerne `√d_model`) a tým väčší batch.

---

## 5. Warmup — rozcvička, bez ktorej to nejde

Rozvrh learning rate je popísaný v [01-adam-optimalizator.md](01-adam-optimalizator.md), sekcia 8.9. Tu odpovedáme len na otázku, **prečo ho transformer potrebuje tak nutne**, keď MLP sa bez neho zaobíde:

1. **Attention na začiatku nič neznamená.** `W_Q` a `W_K` sú náhodné, takže rozdelenie pozornosti je takmer rovnomerné a každý token „pozerá" na všetky rovnako. Gradienty z tohto stavu sú veľké a ukazujú do náhodných smerov — urobiť podľa nich plný krok znamená rozbiť inicializáciu.
2. **Adam ešte nemá odhad rozptylu.** V prvých krokoch je druhý moment `v` zostavený z jedinej-dvoch vzoriek, takže delenie `√v` je divoké. Bias correction to rieši len čiastočne; warmup to rieši úplne, lebo malý krok nechá momenty ustáliť sa.
3. **Loss na začiatku klesá strmo** (z `ln(vocab)` na polovicu za pár stoviek krokov) a v strmej oblasti veľký krok ľahko prestrelí.

```python
import math
from torch.optim.lr_scheduler import LambdaLR

def rozvrh(krok, warmup=2000, celkom=100_000, min_pomer=0.1):
    if krok < warmup:
        return krok / warmup                                  # lineárny nábeh
    t = (krok - warmup) / max(1, celkom - warmup)              # 0 → 1
    return min_pomer + (1 - min_pomer) * 0.5 * (1 + math.cos(math.pi * t))

scheduler = LambdaLR(optimizer, rozvrh)
```

Typická dĺžka warmupu je **1–5 % celkového počtu krokov**. A dobrý diagnostický signál: ak model diverguje presne vtedy, keď learning rate dosiahne vrchol, warmup bol prikrátky alebo je peak lr privysoký.

---

## 6. Pamäť — kde sa GPU minie

Toto je najčastejší praktický problém pri tréningu transformera. Rozpočet má štyri položky:

| Položka | Koľko | Pri 1B modeli |
|---|---|---|
| parametre (bf16) | 2 B / parameter | 2 GB |
| gradienty (bf16) | 2 B / parameter | 2 GB |
| stavy Adama `m`, `v` (fp32) | 8 B / parameter | 8 GB |
| master kópia váh (fp32) | 4 B / parameter | 4 GB |
| **spolu, bez aktivácií** | **≈ 16 B / parameter** | **16 GB** |
| aktivácie | `≈ B · T · d_model · n_layers · k` | závisí od batchu |

**Preto plný fine-tuning 7B modelu potrebuje vyše 100 GB** a preto existuje LoRA — tam sa trénujú len malé adaptéry, takže riadky „gradienty", „Adam" a „master váhy" sa počítajú len z nich ([07-fine-tuning-lora.md](../04-llm/07-fine-tuning-lora.md)).

Attention pridáva k aktiváciám jednu nepríjemnú položku: matica pozornosti má tvar `[B, n_heads, T, T]`. Pri `B=4`, `n_heads=32`, `T=4096` je to 4 · 32 · 4096² · 2 B ≈ **4,3 GB na jednu vrstvu**. Práve preto je **FlashAttention** taký dôležitý: počíta attention po blokoch a maticu nikdy celú nevytvorí, takže pamäť rastie lineárne s `T`, nie kvadraticky. V PyTorch ju dostanete zadarmo:

```python
import torch.nn.functional as F

# namiesto ručného softmax(Q·Kᵀ/√d)·V
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # použije FlashAttention, ak sa dá
```

Tri páky, keď sa tréning nezmestí — v poradí, v akom ich siahnuť:

```python
# 1) mixed precision: bf16 (na Ampere+ bez GradScaler, na Macu/starších GPU fp16 + GradScaler)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logity = model(x)
    strata = F.cross_entropy(...)

# 2) gradient accumulation: veľký efektívny batch z malých priechodov
(strata / akumulacia).backward()

# 3) gradient checkpointing: aktivácie sa neukladajú, ale pri backwarde prepočítajú
#    (~30 % pomalšie, ale pamäť aktivácií klesne rádovo)
model.gradient_checkpointing_enable()          # Hugging Face
# alebo ručne: torch.utils.checkpoint.checkpoint(blok, x, use_reentrant=False)
```

> **bf16, nie fp16.** Softmax v attention pracuje s exponenciálami a fp16 má maximum 65 504 — jeden veľký logit a máte `inf`, z neho `NaN` a koniec behu. bf16 má rovnaký rozsah exponentu ako fp32 (za cenu menej bitov mantisy) a pri transformeroch je preto štandardom. Podrobne v [02-problemy-pri-uceni.md](02-problemy-pri-uceni.md).

---

## 7. Sanity checky pred veľkým behom

Tréning LLM stojí hodiny GPU. Tieto štyri kontroly trvajú minúty a chytia väčšinu chýb:

1. **Počiatočná loss musí sedieť na `ln(vocab)`.** Neučený model háda rovnomerne, takže cross-entropy vyjde `ln(V)`: pri slovníku 128 256 je to **11,76**. Ak vám prvý krok ukáže 3 alebo 40, máte chybu v loss, v maskovaní alebo v inicializácii — nie „zlý learning rate".
2. **Preučte jeden batch.** Vypnite dropout, pustite 200 krokov na tých istých 8 sekvenciách. Loss musí ísť takmer na nulu. Ak nejde, model nemá dosť kapacity alebo je zlomený gradient — a žiadne ladenie na veľkých dátach to nespraví.
3. **Skontrolujte kauzálnu masku.** Otestujte, že zmena tokenu na pozícii `t` **nezmení** logity na pozíciách `< t`. Ak zmení, model vidí do budúcnosti, loss bude nádherne nízka a model bezcenný. Toto je najzradnejšia chyba v celom tréningu transformera.
4. **Zmerajte pamäť a rýchlosť na 50 krokoch**, nie na treťom — pamäť aktivácií vyskočí až pri najdlhšej sekvencii v dátach.

```python
# 3) test kauzality — musí prejsť
model.eval()
x = torch.randint(0, vocab, (1, 16), device=device)
with torch.no_grad():
    a = model(x)
    x2 = x.clone(); x2[0, 8] = (x2[0, 8] + 1) % vocab      # zmeníme token na pozícii 8
    b = model(x2)
assert torch.allclose(a[0, :8], b[0, :8], atol=1e-4), "maska je deravá — model vidí budúcnosť!"
print("kauzálna maska OK")
```

---

## 8. Katalóg porúch špecifických pre transformer

Všeobecné poruchy (mŕtve neuróny, `NaN`, chyby v dátach, hardvér) sú v [02-problemy-pri-uceni.md](02-problemy-pri-uceni.md). Tieto sa viažu priamo na attention:

| Príznak | Príčina | Riešenie |
|---|---|---|
| Loss diverguje presne pri vrchole lr | prikrátky warmup / privysoký peak lr | predĺžiť warmup, znížiť peak lr o polovicu |
| Náhly **skok loss** uprostred behu (*loss spike*) | zlý batch alebo saturovaná attention | preskočiť batch, znížiť `β₂` na 0,95, clip 1,0, prípadne návrat k poslednému checkpointu |
| Loss klesne na `ln(V)` a stojí | model sa naučil len unigramovú distribúciu — gradient cez attention netečie | overiť škálovanie `1/√d_head`, inicializáciu, pre-LN |
| `NaN` po pár tisíc krokoch v fp16 | pretečenie v softmaxe | prejsť na bf16 |
| Všetky hlavy sa správajú rovnako | entropy collapse | QK-norm, nižší lr, silnejší clipping |
| Validačná loss rastie pri fine-tuningu už po 1 epoche | LLM sa preučí okamžite | 1–3 epochy, nižší lr (1e-5 rádovo), LoRA namiesto plného fine-tuningu |
| Podozrivo nízka loss od začiatku | deravá kauzálna maska alebo dáta v teste aj v tréningu | test z bodu 3 vyššie, deduplikácia dát |
| OOM až po hodine behu | najdlhšia sekvencia v dátach | fixná `T` s paddingom/packingom, checkpointing |

---

## 9. Zhrnutie — čím sa tréning transformera líši

| | feed-forward sieť | transformer |
|---|---|---|
| tréningová slučka | forward → loss → backprop → Adam | **tá istá** |
| optimalizátor | SGD stačí, Adam pomôže | **AdamW prakticky povinný**, β₂ = 0,95 |
| learning rate | konštanta alebo pokles | **warmup + kosínus**, škáluje s veľkosťou modelu |
| clipping | keď nastane problém | **vždy**, norma 1,0 |
| veľkosť batchu | v riadkoch | **v tokenoch** (`B · T`), typicky milióny |
| hlavný žrút pamäte | aktivácie | **stavy Adama + attention aktivácie** |
| typická porucha | mŕtve ReLU, miznúci gradient | **saturovaný softmax, loss spike, deravá maska** |
| weight decay | na všetko | **len na maticové váhy** |

Jednou vetou: **attention nepridáva nový druh učenia, ale robí gradientovú krajinu členitejšou — a všetky „zvláštnosti" tréningu LLM (warmup, clipping, bf16, β₂ = 0,95, `1/√d_head`) sú odpovede práve na to.**

---

## Kontrolné otázky

1. Prečo sa v attention delí `Q·Kᵀ` odmocninou z `d_head`? Čo presne sa stane s gradientom, ak sa to vynechá, a ktorý známy problém z tréningu sietí je to isté?
2. Prečo je pri LLM `β₂ = 0,95` namiesto obvyklých 0,999? Čo sa zlepší a čo sa za to platí?
3. Vysvetlite, prečo transformer potrebuje warmup viac než MLP. Uveďte dva nezávislé dôvody.
4. Prečo sa weight decay nedáva na normalizačné parametre a bias? Čo by sa stalo, keby áno?
5. Model má 1,5 miliardy parametrov. Koľko GB zaberie samotný tréningový stav (váhy, gradienty, Adam, master váhy) v mixed precision? Ukážte výpočet a povedzte, ktorá položka je najväčšia.
6. Načo slúži gradient checkpointing a čo ním platíte? Kedy siahnete skôr po ňom a kedy po gradient accumulation?
7. Prvý krok tréningu ukáže loss 2,5 pri slovníku 32 000 tokenov. Prečo je to podozrivé a čo skontrolujete ako prvé?
8. Čím sa líši pre-LN od post-LN a prečo je táto architektonická voľba v skutočnosti rozhodnutie o tréningu?
9. Prečo je pri tréningu transformera bf16 bezpečnejší než fp16, hoci má menej bitov mantisy? Kde konkrétne v modeli by fp16 pretiekol?

---

### Súvisiace dokumenty

- [01-adam-optimalizator.md](01-adam-optimalizator.md) — tréningová slučka, Adam a AdamW, warmup a kosínusový rozvrh, clipping (mechanika, ktorú tu používame)
- [02-problemy-pri-uceni.md](02-problemy-pri-uceni.md) — všeobecný katalóg porúch tréningu: miznúce gradienty, `NaN`, fp16/bf16, checkpointy
- [03-xgboost-trening-a-inferencia.md](03-xgboost-trening-a-inferencia.md) — tá istá disciplína na tabuľkových dátach, kde krokom nie je update čísla, ale pridanie stromu
- [01-transformer-siete.md](../04-llm/01-transformer-siete.md) — attention, multi-head, maska (architektúra, ktorú tu trénujeme)
- [02-transformer-vnutro.md](../04-llm/02-transformer-vnutro.md) — rozmery, reziduálny prúd, počty parametrov
- [03-llm-trening.md](../04-llm/03-llm-trening.md) — **nasleduje**: fázy tréningu LLM (pretraining → SFT → RLHF), teacher forcing, scaling laws
- [07-fine-tuning-lora.md](../04-llm/07-fine-tuning-lora.md) — ako sa tie isté pamäťové nároky zmenšia na domácu GPU

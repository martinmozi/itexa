# Vývojové prostredie pre AI — čo nainštalovať a na čom to spustiť

> **Cieľ dokumentu:** praktická príručka, ako si pripraviť počítač na prácu s AI — od Python prostredia a PyTorchu cez nastavenie GPU (CUDA na NVIDIA, Metal na Macu) až po lokálnu inferenciu LLM pomocou vLLM. Na záver odporúčania, aký hardvér má zmysel doma a kedy je čas prenajať si GPU v cloude (runpod.io a spol.). Predpokladáme, že Python ovládate.

Príručka pokrýva všetko, čo budete potrebovať na [zadanie 1](zadania/rozpoznavanie-obrazkov.md) (vlastná sieť + PyTorch) aj [zadanie 2](zadania/RAG_Fine_tunning.md) (RAG a LoRA fine-tuning).

---

## 1. Python prostredie

Na AI vývoj stačí **Python 3.10 až 3.12**. Úplne najnovšiu verziu Pythonu sa neoplatí ponáhľať — PyTorch a spol. ju podporia typicky až o pár mesiacov po vydaní.

Prvé pravidlo: **nikdy neinštalujte knižnice do systémového Pythonu.** AI knižnice sú veľké, majú prísne vzájomné závislosti na verziách a jeden pokazený upgrade vie rozbiť celé prostredie. Každý projekt preto dostane vlastné virtuálne prostredie:

```bash
# klasika — venv (súčasť Pythonu)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

Rýchlejšia moderná alternatíva je **uv** — správca balíkov, ktorý robí to isté čo pip + venv, ale rádovo rýchlejšie:

```bash
uv venv                          # vytvorí .venv
uv pip install torch numpy       # inštaluje do neho
```

Kedysi bola v AI svete štandardom **conda**; dnes už nie je potrebná — pip balíky PyTorchu si nesú všetko so sebou (vrátane CUDA knižníc, ako uvidíme nižšie). Ak ju máte radi, funguje tiež, ale v tomto kurze vystačíme s venv/uv.

Na experimentovanie sa hodí **Jupyter** (`pip install jupyterlab`) alebo notebooky priamo vo VS Code — kód sa spúšťa po bunkách a grafy vidno hneď vedľa kódu.

---

## 2. Editor — VS Code a rozšírenia

Editor je vec vkusu, ale ak nemáte vyhranený názor, zvoľte **Visual Studio Code** — je zadarmo, beží všade a má najlepšiu podporu pre Python aj notebooky. Z tisícok rozšírení budete reálne potrebovať týchto pár:

| Rozšírenie | Na čo |
|---|---|
| **Python** (Microsoft) | spúšťanie, debugovanie, výber interpretera; automaticky doinštaluje **Pylance** (napovedanie, kontrola typov) |
| **Jupyter** (Microsoft) | notebooky `.ipynb` priamo v editore — netreba spúšťať JupyterLab v prehliadači |
| **Ruff** | rýchly linter a formátovač Python kódu — udrží kód čistý bez ručného upratovania |
| **Remote – SSH** | vývoj na vzdialenom stroji: pripojíte sa na prenajatý runpod server a pracujete v ňom, akoby bol lokálny (zíde sa v sekcii 8) |
| **WSL** | len pre Windows: otvorí projekt priamo v Ubuntu vo WSL2, kde beží celý AI ekosystém |

Jedno nastavenie, ktoré si treba osvojiť hneď: **výber interpretera.** Po vytvorení virtuálneho prostredia stlačte `Ctrl+Shift+P` → *Python: Select Interpreter* → vyberte `.venv` v projekte. VS Code potom prostredie sám aktivuje v každom novom termináli a debugger aj notebooky používajú správne knižnice. Ak vám import „nefunguje", v deviatich prípadoch z desiatich beží kód proti inému interpreteru, než do ktorého ste inštalovali.

K AI asistentom v editore (Claude Code, Copilot a spol.): pomôžu, ale v tomto kurze je cieľom pochopiť mechaniku vlastnými rukami — pri zadaniach nimi šetrite. Ako sa s nimi pracuje efektívne a kedy im (ne)veriť, je téma lekcie 8.

---

## 3. PyTorch

**PyTorch** je knižnica, na ktorej v tomto kurze stojí všetko od zadania 1 vyššie: tenzory (n-rozmerné polia bežiace na GPU), automatické derivovanie (autograd — základ backpropagation) a hotové vrstvy i optimalizátory. V AI výskume je dnes de facto štandard a stavajú na ňom aj knižnice ako `transformers` či vLLM.

Inštalačný príkaz sa líši podľa operačného systému a GPU, preto si ho vždy nechajte vygenerovať na **[pytorch.org](https://pytorch.org)** (sekcia *Get Started*). Orientačne:

```bash
# Linux / Windows s NVIDIA GPU (verzia CUDA podľa selektora na pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Mac (Apple Silicon) — bez ďalších parametrov
pip install torch
```

Hneď po inštalácii si overte, že PyTorch beží a vidí akcelerátor:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())          # True na stroji s NVIDIA GPU
print(torch.backends.mps.is_available())  # True na Macu s M-čipom
```

Ak obe hlásia `False`, PyTorch pobeží na CPU — na zadanie 1 to stačí (malé siete), na prácu s LLM už nie.

---

## 4. GPU akcelerácia

Neurónové siete sú v jadre násobenie matíc a to je presne úloha, na ktorú je grafická karta stavaná — tréning na GPU býva 10× až 100× rýchlejší než na CPU. Nastavenie sa líši podľa platformy.

### NVIDIA a CUDA (Linux, Windows)

**CUDA** je rozhranie NVIDIA, cez ktoré programy počítajú na ich grafických kartách. Dôležitá vec, ktorá študentov často mätie: **celý CUDA toolkit dnes inštalovať netreba.** Pip balík PyTorchu si nesie vlastné CUDA knižnice so sebou — jediné, čo musí byť v systéme, je **ovládač NVIDIA**. Či je ovládač v poriadku, overíte príkazom:

```bash
nvidia-smi
```

Výpis ukáže model karty, obsadenú pamäť a verziu ovládača (a najvyššiu verziu CUDA, ktorú ovládač podporuje — tá musí byť aspoň taká, akú vyžaduje zvolený PyTorch build). Ak `nvidia-smi` nefunguje, treba najprv doinštalovať ovládač: na Linuxe z balíkov distribúcie, na Windows bežný GeForce ovládač.

Samostatný CUDA toolkit (kompilátor `nvcc`) budete potrebovať, až keď budete niečo kompilovať zo zdrojákov — napríklad špecializované kernely. V tomto kurze taká situácia nenastane.

**Poznámka k Windows:** PyTorch s CUDA funguje na Windows natívne, ale veľká časť AI ekosystému (vrátane vLLM) beží len na Linuxe. Odporúčame preto **WSL2** (Ubuntu vo Windows) — GPU je v ňom plne dostupná a ušetríte si veľa trápenia s nekompatibilnými nástrojmi.

V kóde potom stačí presunúť model a dáta na správne zariadenie:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x = x.to(device)
```

### Mac a Apple Silicon (MPS)

Na Macu CUDA neexistuje — NVIDIA karty sa doň nedajú osadiť. Apple čipy radu M (M1 a novšie) však majú vlastnú GPU a PyTorch ju podporuje cez backend **MPS** (*Metal Performance Shaders*). Netreba nič nastavovať: obyčajný `pip install torch` a namiesto `"cuda"` použijete `"mps"`:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

Veľká výhoda Macov je **zjednotená pamäť** (*unified memory*): GPU zdieľa pamäť s CPU, takže Mac so 32 GB RAM má pre modely k dispozícii podstatne viac „VRAM" než bežná herná karta. Preto Macy prekvapivo dobre zvládajú **inferenciu** väčších modelov. Na **tréning** sú však výrazne pomalšie než desktopová NVIDIA — a ojedinele narazíte na operáciu, ktorú MPS nepodporuje (pomôže premenná prostredia `PYTORCH_ENABLE_MPS_FALLBACK=1`, ktorá ju nechá dobehnúť na CPU).

Univerzálny výber zariadenia, ktorý funguje všade:

```python
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

---

## 5. Knižnice pre tento kurz

Okrem PyTorchu budete postupne potrebovať:

| Knižnica | Na čo | Kde v kurze |
|---|---|---|
| `numpy`, `matplotlib` | polia, grafy | zadanie 1 (sieť v NumPy) |
| `jupyterlab` | notebooky na experimenty | všade |
| `transformers`, `datasets` | modely a datasety z Hugging Face | lekcie 5–7 |
| `sentence-transformers` | embedding modely | zadanie 2A (RAG) |
| `faiss-cpu` | vektorový index | zadanie 2A (RAG) |
| `peft`, `bitsandbytes`, `accelerate` | LoRA/QLoRA fine-tuning | zadanie 2B |
| `vllm` | rýchla inferencia LLM (len NVIDIA) | nižšie |

Pozor na `bitsandbytes` a `vllm` — vyžadujú NVIDIA GPU; na Macu ich preskočte (náhrady spomíname pri každom zadaní a v sekcii o vLLM).

---

## 6. Lokálna inferencia LLM — vLLM

Keď si stiahnete open-weight model z Hugging Face, môžete ho spustiť priamo cez `transformers` — na jednorazové pokusy to stačí. Len čo však chcete model **obsluhovať**: posielať mu veľa požiadaviek, merať RAG pipeline, simulovať API — potrebujete inferenčný server. Štandardom je **vLLM**.

vLLM je optimalizovaný inferenčný engine: vďaka technike **PagedAttention** hospodári s pamäťou KV cache (medzivýsledky attention, ktoré pri generovaní rastú s dĺžkou kontextu) a vďaka **continuous batchingu** obsluhuje veľa súbežných požiadaviek naraz — priepustnosť býva oproti naivnému `model.generate()` niekoľkonásobná. Navonok vystavuje **OpenAI-kompatibilné API**, takže kód napísaný proti OpenAI klientovi funguje bez zmeny aj proti vášmu lokálnemu modelu.

Požiadavky: **Linux (alebo WSL2) a NVIDIA GPU.** Spustenie:

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 4096
```

Model sa pri prvom spustení stiahne z Hugging Face a server počúva na `http://localhost:8000/v1`. Dotazovať sa dá bežným OpenAI klientom:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="nepotrebny")
odpoved = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Vysvetli attention jednou vetou."}],
)
print(odpoved.choices[0].message.content)
```

Užitočné prepínače:

- `--max-model-len` — maximálna dĺžka kontextu; kratší kontext = menšia KV cache = menej VRAM,
- `--gpu-memory-utilization 0.9` — koľko VRAM smie vLLM zabrať (zvyšok nechá systému),
- `--quantization awq` — beh kvantizovaného modelu (na Hugging Face hľadajte varianty s príponou AWQ alebo GPTQ), keď sa plná verzia do VRAM nezmestí.

Celý model vrátane KV cache sa musí zmestiť do VRAM — koľko čo potrebuje, rozoberá ďalšia sekcia.

### A čo na Macu? Ollama a llama.cpp

vLLM Apple GPU nepodporuje. Na Macu (a pokojne aj na slabšom PC) je najjednoduchšou cestou **Ollama** — obal okolo knižnice **llama.cpp**, ktorá beží na Metal, CUDA aj čistom CPU a používa úsporné kvantizované modely vo formáte GGUF:

```bash
brew install ollama            # Linux: curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3.2            # stiahne model a otvorí chat v termináli
```

Ollama tiež vystavuje OpenAI-kompatibilné API (na `http://localhost:11434/v1`), takže kód z príkladu vyššie stačí presmerovať na iný port. Ak preferujete grafické rozhranie, rovnakú službu spraví **LM Studio**.

Praktické delenie: **Ollama** = pohodlie, jeden používateľ, beží všade. **vLLM** = priepustnosť a produkčné API, vyžaduje NVIDIA. Na zadanie 2 vystačíte s ktorýmkoľvek z nich.

---

## 7. Odporúčaný hardvér na doma

Pri výbere karty na AI je najdôležitejšie jediné číslo: **VRAM**. Výkon čipu rozhoduje o tom, ako *rýchlo* model pobeží; VRAM rozhoduje o tom, či pobeží *vôbec*. Hrubý výpočet je jednoduchý:

- **fp16/bf16:** 2 bajty na parameter → 8B model ≈ 16 GB len na váhy,
- **4-bitová kvantizácia:** ~0,5–0,6 bajtu na parameter → 8B model ≈ 5 GB,
- k tomu vždy réžia: KV cache (rastie s dĺžkou kontextu), aktivácie, samotný systém.

| Model | Inferencia fp16 | Inferencia 4-bit | Zmestí sa na |
|---|---|---|---|
| 1–3B | 2–6 GB | 1–2 GB | čokoľvek vrátane notebooku |
| 7–8B | ~16 GB | ~5 GB | 8 GB karta (4-bit), 24 GB (fp16) |
| 13–14B | ~28 GB | ~9 GB | 12 GB karta (4-bit) |
| ~30B | ~60 GB | ~18 GB | 24 GB karta (4-bit) |
| 70B | ~140 GB | ~40 GB | 2× 24 GB, Mac 64 GB+, alebo cloud |

**Tréning žerie omnoho viac než inferencia.** Pri plnom fine-tuningu sa okrem váh držia v pamäti aj gradienty a stavy optimalizátora Adam — dokopy zhruba **16 bajtov na parameter**, takže plný fine-tuning 7B modelu chce vyše 100 GB a patrí do cloudu. Zachraňuje to **QLoRA** (základný model 4-bitový a zmrazený, trénujú sa len malé adaptéry — pozri lekciu 7): fine-tuning 7B modelu sa vojde do ~10–12 GB, teda na slušnú domácu kartu.

Odporúčania podľa rozpočtu (stav v roku 2026, ceny sa hýbu):

- **Vstupná úroveň — NVIDIA s 12–16 GB VRAM** (RTX 3060 12 GB z bazáru, 4060 Ti 16 GB, 5060 Ti 16 GB): zvládne všetky zadania kurzu, 4-bit inferenciu do ~14B aj QLoRA 7B.
- **Nadšenecká úroveň — 24 GB a viac** (bazárová RTX 3090 je dlhodobo najlacnejších 24 GB; 4090, 5090 s 32 GB): pohodlný fp16 beh 7–8B, 4-bit inferencia ~30B, väčší priestor na experimenty.
- **Mac:** M-čip so **16 GB je minimum, 32 GB a viac je na LLM príjemné** — vďaka zjednotenej pamäti výborný na inferenciu cez Ollamu, na tréning rátajte s trpezlivosťou.
- **Bez GPU?** Žiadna tragédia: malé modely v 4-bit bežia cez llama.cpp aj na CPU a **Google Colab dáva zadarmo GPU T4 (16 GB)** — tá pokryje všetky zadania tohto kurzu. Kúpu karty pokojne odložte, kým nebudete vedieť, že ju využijete.

---

## 8. Kedy už doma nestačí — runpod.io a spol.

Hranica je jednoduchá: **keď sa úloha nezmestí do vašej VRAM ani po kvantizácii a QLoRA trikoch, alebo by bežala neúnosne dlho.** Typicky:

- plný fine-tuning čohokoľvek od ~7B vyššie,
- tréning alebo inferencia modelov, ktoré sa nezmestia ani v 4-bit (70B+),
- dlhé behy, ktoré by domácu kartu blokovali na dni,
- potreba viacerých GPU alebo veľkej pamäte jednej karty (A100/H100 s 80 GB).

**[runpod.io](https://runpod.io)** je požičovňa GPU: vyberiete si kartu, hotový image s PyTorch/CUDA, a o minútu máte SSH a Jupyter na stroji s H100 — cez rozšírenie Remote – SSH (sekcia 2) sa naň pripojíte priamo z VS Code a pracujete ako na lokálnom projekte. Platí sa za hodinu behu; orientačne (ceny sa menia, pozrite aktuálny cenník) stojí RTX 4090 desiatky centov za hodinu, A100/H100 rádovo 1–3 $ za hodinu. Dáta medzi behmi prežijú na prenajatom sieťovom disku (*network volume*). Podobné služby: **vast.ai** (aukčný trh, býva najlacnejší), **Lambda**, platený **Colab Pro**; na malé experimenty zadarmo aj **Kaggle** notebooky.

Dve praktické rady:

1. **Vyvíjajte lokálne, trénujte v cloude.** Skript odlaďte doma na malom modeli a vzorke dát; na prenajatej GPU už len spustite hotovú vec. Ladenie preklepov za 2 $/hodinu je zbytočný luxus.
2. **Vypínajte pody.** Účtuje sa každá hodina behu — beh cez zabudnutý víkend stojí viac než celý mesiac experimentov.

A ešte jedno rozhodnutie pred prenájmom: ak nepotrebujete **vlastné váhy** (fine-tuning, plná kontrola, citlivé dáta), býva lacnejšie nevolať žiadnu GPU a použiť hotové **API** (Anthropic, OpenAI, Together…) — platí sa za tokeny, nie za hodiny. Kritériá výberu modelu rozoberá [llm-modely.md](llm-modely.md).

---

## Zhrnutie — úloha → kde ju bežať

| Úloha | Kde |
|---|---|
| zadanie 1 (MLP v NumPy a PyTorch) | CPU / akákoľvek GPU / Colab |
| RAG s malým modelom (zadanie 2A) | GPU 8–16 GB, Mac 16 GB+ s Ollamou, alebo Colab |
| QLoRA fine-tuning 7B (zadanie 2B) | GPU 12–16 GB alebo Colab T4 |
| inferencia 30B (4-bit) | GPU 24 GB alebo Mac 32 GB+ |
| plný fine-tuning 7B+ | cloud (runpod — A100/H100) |
| inferencia 70B | cloud, 2× 24 GB, alebo Mac 64 GB+ (pomaly) |

---

## Kontrolné otázky

Ak viete odpovedať vlastnými slovami, dokument ste pochopili:

1. Prečo dnes pri inštalácii PyTorchu netreba inštalovať celý CUDA toolkit? Čo jediné musí byť v systéme a ako overíte, že funguje?
2. Ako v PyTorch kóde napíšete výber zariadenia tak, aby ten istý skript bežal na NVIDIA stroji, Macu aj na CPU?
3. Koľko VRAM potrebuje inferencia 8B modelu vo fp16 a koľko po 4-bitovej kvantizácii? Ukážte výpočet.
4. Čím sa líši vLLM od Ollamy — technicky aj použitím — a kedy siahnete po ktorom?
5. Prečo potrebuje plný fine-tuning niekoľkonásobne viac pamäte než inferencia toho istého modelu a ako tento problém obchádza QLoRA?
6. Spolužiak s RTX 4060 Ti (16 GB) chce spraviť plný fine-tuning 8B modelu. Čo mu poradíte a aké dve lacnejšie alternatívy mu ponúknete?

---

### Súvisiace dokumenty v repozitári

- [prehlad-predmetu.md](prehlad-predmetu.md) — prehľad celého predmetu (8 lekcií)
- [umela-inteligencia-prehlad.md](umela-inteligencia-prehlad.md) — prehľad prístupov a modelov
- [adam-optimalizator.md](adam-optimalizator.md) — tréningová slučka, backpropagation, Adam
- [llm-modely.md](llm-modely.md) — výber modelu (proprietárne / open-weight / open-source)
- [llm-trendy.md](llm-trendy.md) — LoRA/QLoRA, kedy fine-tuning a kedy RAG
- [zadania/rozpoznavanie-obrazkov.md](zadania/rozpoznavanie-obrazkov.md), [zadania/RAG_Fine_tunning.md](zadania/RAG_Fine_tunning.md) — praktické úlohy

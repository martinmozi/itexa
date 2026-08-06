# Zadanie: Rozpoznávanie obrázkov vlastnou neurónovou sieťou

> **Zadanie 1 z 2** · rieši sa počas lekcií 3–4 · podklady: [01-adam-optimalizator.md](../tutorials/03-ucenie/01-adam-optimalizator.md), [prehľad AI](../tutorials/01-prehlad/README.md), [typy modelov](../tutorials/02-typy-modelov/README.md), [01-vyvojove-prostredie.md](../tutorials/00-prostredie/01-vyvojove-prostredie.md)

## Cieľ

Naprogramovať doprednú (feed-forward) neurónovú sieť na klasifikáciu obrázkov **dvoma spôsobmi**:

1. **Bez PyTorch** — vlastná implementácia siete (forward aj backpropagation) len s NumPy,
   vrátane vlastného optimalizátora **Adam**.
2. **S PyTorch** — tá istá úloha postavená na frameworku.

Sieť natrénujete na zvolenom datasete, natrénovanú sieť použijete na rozpoznanie **vlastného
vstupu**, ktorý si sami vytvoríte (nakreslíte / pripravíte), a na záver **porovnáte obe verzie**.

Dôraz je na pochopení, ako sieť funguje zvnútra — preto najprv vlastná implementácia,
až potom „pohodlný" PyTorch.

> Špecifikácia optimalizátora Adam vrátane matematiky je v samostatnom dokumente
> [01-adam-optimalizator.md](../tutorials/03-ucenie/01-adam-optimalizator.md).

---

## Varianty zadania (vyberte / bude vám pridelený jeden)

Jadro zadania je pre všetkých rovnaké. Líši sa **dataset** a tým aj **vlastný testovací vstup**.
Všetky datasety sú na Hugging Face a načítajú sa knižnicou `datasets` (`pip install datasets`).

| # | Variant | Dataset (Hugging Face) | Vstup | Počet tried | Vlastný vstup |
|---|---|---|---|---|---|
| **1** | **Písmená (EMNIST Letters)** | [`randall-lab/emnist`](https://huggingface.co/datasets/randall-lab/emnist) (konfig. `letters`) | 28×28 grayscale | 26 (A–Z) | nakresli písmeno v MS Paint |
| **2** | **Matematické symboly (HASYv2)** | [`randall-lab/hasy-v2`](https://huggingface.co/datasets/randall-lab/hasy-v2) | 32×32 grayscale | 369 (`+ − × ∑ √ …`) | nakresli symbol v MS Paint |
| **3** | **Oblečenie (Fashion-MNIST)** | [`zalando-datasets/fashion_mnist`](https://huggingface.co/datasets/zalando-datasets/fashion_mnist) | 28×28 grayscale | 10 (tričko, topánka…) | nakresli/priprav obrázok kúska oblečenia |

> *Tip:* ako rozcvičku si môžete najprv skúsiť klasické **MNIST číslice**
> ([`ylecun/mnist`](https://huggingface.co/datasets/ylecun/mnist)) — je to najjednoduchší prípad
> tej istej úlohy.

### Poznámky k variantom

- **Variant 1 (EMNIST):** takmer identický s MNIST, len 26 tried. Pozor — obrázky v EMNIST
  bývajú **otočené/zrkadlené**, overte si orientáciu vizualizáciou pár vzoriek.
- **Variant 2 (HASYv2):** obrázky sú **32×32** → vstupná vrstva má **1024** neurónov. 369 tried
  je veľa; **je povolené vybrať si podmnožinu** ~15–30 najčastejších symbolov a klasifikovať len tie
  (napíšte do správy, ktoré). Toto je najnáročnejší variant.
- **Variant 3 (Fashion-MNIST):** rovnaký formát ako MNIST (28×28, 10 tried), ale ťažší —
  presnosť ~88–90 % je dobrý výsledok.

---

## Povolené nástroje

- **Python** + **NumPy** (maticové operácie sú OK a odporúčané).
- **PyTorch** — **iba** v Časti 2B.
- `datasets` (Hugging Face) na stiahnutie dát, `Pillow` na obrázky, `matplotlib` na vizualizáciu.

---

## Časť 1 — Dáta (spoločné pre obe verzie)

1. Stiahnite dataset svojho variantu z Hugging Face:
   ```python
   from datasets import load_dataset
   # príklad pre variant 3:
   ds = load_dataset("zalando-datasets/fashion_mnist")
   train, test = ds["train"], ds["test"]
   ```
   > *Hint:* pri EMNIST/HASYv2 doplňte príslušný `name=`/konfiguráciu podľa README datasetu.
2. Každý obrázok preveďte na **vektor** (28×28 → 784, resp. 32×32 → 1024) a **normalizujte**
   pixely do rozsahu `<0, 1>` (delenie 255).
3. Cieľové labely preveďte na **one-hot** vektory dĺžky = počet tried (pri PyTorch stačí index).
   > *Hint:* vždy si najprv **vizualizujte pár vzoriek aj s labelmi** (`matplotlib`), aby ste
   > overili orientáciu obrázka a správne priradenie tried.
4. Z trénovacej časti si odkrojte **validačnú množinu** (napr. 10 %, `train` → `train` + `val`).
   Testovacia sada z datasetu zostane **nedotknutá** až do záverečného merania.
   > *Prečo:* v Časti 3 budete porovnávať niekoľko architektúr. Keby ste tú najlepšiu vybrali
   > podľa presnosti na testovacej sade, prestala by byť nezávislá a výsledné číslo by bolo
   > optimistickejšie než realita — viď [zovšeobecnenie](../tutorials/01-prehlad/03-generalizacia-a-preucenie.md).

---

## Časť 2A — Vlastná sieť BEZ PyTorch

Cieľ: implementovať sieť od nuly, aby ste rozumeli, čo sa deje „pod kapotou".

- **Vstupná vrstva:** podľa variantu (784 alebo 1024 neurónov).
- **Skryté vrstvy:** konfigurovateľné — zadané ako zoznam, napr. `[128, 64]`.
- **Výstupná vrstva:** počet neurónov = počet tried (softmax).

### Komponenty na implementáciu

| Komponent | Hint |
|---|---|
| Inicializácia váh | malé náhodné čísla `np.random.randn(...) * 0.01`, alebo **He/Xavier** |
| Aktivácia (skryté) | `ReLU` alebo `sigmoid` |
| Aktivácia (výstup) | `softmax` |
| Chybová funkcia | `cross-entropy` |
| Dopredný priechod | `z = W·a + b`, potom aktivácia |
| Spätné šírenie | reťazové pravidlo, gradienty `dW`, `db` |
| Aktualizácia váh | najprv SGD `W -= lr*dW`, potom **Adam** (viď [01-adam-optimalizator.md](../tutorials/03-ucenie/01-adam-optimalizator.md)) |

> *Hint k backpropu:* pri kombinácii **softmax + cross-entropy** sa gradient na výstupe
> zjednoduší na `(predikcia − skutočnosť)`. Overte si to na papieri.

> *Hint k tréningu:* použite **mini-batch** (napr. 32 alebo 64 vzoriek).

> *Keď sa sieť neučí:* skôr než začnete ladiť hyperparametre, prejdite si
> **[02-problemy-pri-uceni.md](../tutorials/03-ucenie/02-problemy-pri-uceni.md)**. Začnite testom
> **„preuč 8 príkladov"** — ak sieť nedokáže naspamäť zvládnuť osem vzoriek, máte chybu
> v backprope alebo v tvaroch polí, nie problém s učením.

### Optimalizátor Adam (povinné)

Najprv rozbehajte tréning s obyčajným SGD, potom **implementujte Adam**. Postupujte podľa
špecifikácie v [01-adam-optimalizator.md](../tutorials/03-ucenie/01-adam-optimalizator.md) (obsahuje vzorce, matematiku aj referenčný kód).
Do správy porovnajte konvergenciu **SGD vs. Adam** (loss po epochách).

---

## Časť 2B — Tá istá sieť S PyTorch

Cieľ: postaviť ekvivalentnú sieť rýchlo a porovnať, čo za vás framework urobí.

- Model ako `nn.Module` s `nn.Linear` vrstvami a rovnakou architektúrou ako v Časti 2A.
- Aktivácie cez `torch.relu` / `nn.ReLU`.
- Loss: `nn.CrossEntropyLoss` (softmax má v sebe — na výstup nedávajte softmax navyše).
- Optimalizátor: `torch.optim.Adam` (rovnaký ako ste si napísali v Časti 2A).
- Tréningová slučka: `forward → loss → loss.backward() → optimizer.step()`.

> *Hint:* všimnite si, že `loss.backward()` nahrádza celý ručný backprop z Časti 2A a
> `torch.optim.Adam` nahrádza váš vlastný Adam — presne to, čo ste si napísali sami.

> *Hint:* dáta podávajte cez `DataLoader` s `batch_size` a `shuffle=True`.

---

## Časť 3 — Tréning a experimenty (pre obe verzie)

1. Trénujte niekoľko epôch a sledujte **loss** a **presnosť na validačnej množine** po každej
   epoche.
2. **Experimentujte s architektúrou** a zdokumentujte do tabuľky (rovnaké nastavenia pre
   Časť 2A aj 2B, aby bolo porovnanie férové). Porovnávajte podľa **validačnej** presnosti:

   | Skryté vrstvy | LR | Epochy | Val. presnosť 2A (NumPy) | Val. presnosť 2B (PyTorch) |
   |---|---|---|---|---|
   | `[64]` | 0.001 | 10 | ? | ? |
   | `[128, 64]` | 0.001 | 10 | ? | ? |
   | `[256, 128, 64]` | 0.001 | 10 | ? | ? |

   > *Hint:* cieľová presnosť závisí od variantu (MNIST/EMNIST > 95 %, Fashion-MNIST ~88 %,
   > HASYv2 podľa výberu podmnožiny tried). Viac vrstiev nie je vždy lepšie — sledujte aj čas.
3. **Najlepšiu architektúru** z tabuľky zmerajte **raz** na testovacej sade a toto číslo uveďte
   ako výslednú presnosť. Ak je citeľne nižšie než validačná, napíšte do správy prečo.
4. **Uložte natrénované siete** (váhy).
   > *Hint:* 2A → `np.savez` / `pickle`; 2B → `torch.save(model.state_dict(), ...)`.

---

## Časť 4 — Vlastný vstup

1. Vytvorte si **vlastný testovací obrázok** podľa svojho variantu (nakreslite písmeno / symbol /
   kúsok oblečenia v MS Paint alebo inom editore) — konzistentne s datasetom.
2. Napíšte Python script, ktorý obrázok pripraví do formátu datasetu:
   - **zmenšite** na rozmer datasetu (28×28 alebo 32×32),
   - preveďte na **odtiene sivej** (grayscale),
   - **invertujte** farby, ak treba (overte si, či má dataset svetlý objekt na tmavom pozadí),
   - **normalizujte** do `<0, 1>`.
   > *Hint:* `Image.open(...).convert('L').resize((W,W))`, potom `np.array(...)`.
3. Podajte obrázok **obom natrénovaným sieťam** (2A aj 2B) a vypíšte predikciu a
   pravdepodobnosti. Zhodujú sa?

### Nepovinné (bonus)

- **Centrovanie a škálovanie** objektu podľa ťažiska — výrazne zlepší úspešnosť na vlastných
  obrázkoch. *(Postup: bounding box → orez → padding → vycentrovanie cez `scipy.ndimage.center_of_mass`.)*
- Jednoduché **GUI** na kreslenie (`tkinter` canvas), ktoré rovno pošle obrázok sieti.

---

## Diskusná otázka (do správy)

Zamyslite sa a stručne odpovedzte:

- Čo sú **lokálne minimá** a **sedlové body** chybovej funkcie? V čom sa líšia?
- Prečo v sieťach s veľa parametrami (vysoká dimenzia) **nie sú lokálne minimá až taký
  problém**, ako sa bežne intuitívne čaká — a čo býva reálnou prekážkou tréningu?
- Ako pomáhajú **stochastickosť mini-batchov**, **momentum** a **Adam** dostať sa z týchto
  problematických miest? Pozorovali ste rozdiel medzi SGD a Adam pri vašom tréningu?

> *Hint na zamyslenie:* metódy druhého rádu (napr. Levenberg–Marquardt) sú priťahované ku
> **každému** stacionárnemu bodu vrátane sediel — prečo je pri tomto probléme šum v SGD/Adam
> skôr výhodou než nevýhodou?

---

## Odovzdanie

1. **Zdrojový kód** — obe verzie (2A bez PyTorch vrátane vlastného Adam, 2B s PyTorch) + príprava vstupu.
2. **Uložené natrénované siete** (obe).
3. **Krátku správu (1–2 strany)**:
   - zvolený variant a popis architektúry a hyperparametrov,
   - tabuľku experimentov s rôznymi skrytými vrstvami (2A vs. 2B) na **validačnej** množine
     a záverečné meranie najlepšej architektúry na **testovacej** množine,
   - porovnanie konvergencie **SGD vs. Adam**,
   - **porovnanie oboch verzií** — presnosť, čas tréningu, náročnosť implementácie,
   - ukážku rozpoznania vlastného vstupu (screenshot),
   - odpoveď na diskusnú otázku.

## Hodnotenie (orientačne)

| Kritérium | Body |
|---|---|
| Vlastná feed-forward sieť BEZ PyTorch (forward + backprop) | 30 % |
| Vlastná implementácia optimalizátora Adam | 10 % |
| Ekvivalentná sieť S PyTorch | 15 % |
| Tréning, presnosť a experimenty s rôznymi skrytými vrstvami | 15 % |
| Rozpoznanie vlastného vstupu | 15 % |
| Porovnanie verzií + diskusná otázka v správe | 10 % |
| Bonus (centrovanie / GUI) | +10 % |
| Prehľadnosť kódu a správy | 5 % |

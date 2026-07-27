# Fine-tuning v malom: LoRA, QLoRA a rozhodnutie RAG vs. fine-tuning

> **Poradie čítania:** ← [embeddings.md](embeddings.md) · **lekcia 7** → [zadanie 2B](zadania/RAG_Fine_tunning.md) · ďalej [agenti-a-nastroje.md](agenti-a-nastroje.md) →

> **Cieľ dokumentu:** vysvetliť, ako sa dá veľký model prispôsobiť vlastnej úlohe na jednom GPU — čo presne je **LoRA adaptér**, prečo stačí, čo pridáva **QLoRA** — a hlavne vedieť sa **rozhodnúť**, kedy siahnuť po fine-tuningu a kedy po RAG alebo len po dlhšom prompte.

Nadväzuje na [llm-trening.md](llm-trening.md): fine-tuning, ktorý tu robíme, je presne **Fáza 2 (SFT)** z tamojšej pipeline, len na malých dátach a s malým počtom trénovaných parametrov. Tréningová slučka je stále tá istá ako v [lekcii 3](adam-optimalizator.md).

---

## 1. Prečo sa celý model dotrénovať nedá

Plný fine-tuning znamená upravovať **všetky** váhy. Pri modeli so 7 miliardami parametrov treba v pamäti držať:

| Čo | Koľko na parameter | Pre 7B |
|---|---|---|
| váhy (bf16) | 2 B | 14 GB |
| gradienty | 2 B | 14 GB |
| stav Adamu (`m`, `v`, fp32) | 8 B | 56 GB |
| kópia váh vo fp32 | 4 B | 28 GB |
| **spolu** | **~16 B** | **~112 GB** |

K tomu ešte aktivácie. Čiže niekoľko A100/H100 — pre bežnú firmu aj študenta nedostupné. A to všetko preto, aby sme model naučili napríklad odpovedať v našom firemnom tóne.

Kľúčové pozorovanie, z ktorého vychádza LoRA: **prispôsobenie modelu na úzku úlohu je „malá" zmena.** Nemeníme, čo model vie o svete — meníme, ako to podáva. Taká zmena sa nemusí dať zapísať do všetkých miliárd čísel; stačí jej oveľa menší priestor.

---

## 2. LoRA — čo to je

**LoRA** (*Low-Rank Adaptation*) pôvodný model **zmrazí** (žiadna jeho váha sa nemení) a k vybraným váhovým maticiam pripojí **dve malé matice navyše**.

Nech je pôvodná váhová matica `W` rozmeru `[d × k]` (napr. `[4096 × 4096]`). Namiesto toho, aby sme ju upravovali, priráta sa k nej korekcia `ΔW`, ktorá sa rozloží na súčin dvoch úzkych matíc:

```text
výstup = x · (W + ΔW)        kde   ΔW = A · B

  A má rozmer [d × r]          r je „rank" — malé číslo, typicky 8, 16, 32
  B má rozmer [r × k]
```

Trénujú sa **len `A` a `B`**; `W` zostáva zamrznuté. Celý zisk je v počte parametrov:

```text
plná matica W :  4096 × 4096            = 16 777 216 parametrov
LoRA pri r=8  :  4096×8 + 8×4096        =     65 536 parametrov     (0,4 %)
```

Pri typickom 7B modeli sa tak namiesto 7 miliárd trénuje rádovo **10–50 miliónov** parametrov, čiže menej než 1 %. Gradienty a stav Adamu treba držať len pre ne — a práve tie tri riadky tabuľky vyššie boli tie drahé.

**Prečo to funguje:** rozklad `A · B` dokáže vyrobiť len matice **hodnosti (ranku) najviac `r`** — teda „chudobnejšie" zmeny než ľubovoľná matica. Empirický nález článku o LoRA je, že prispôsobenie na úzku úlohu takú bohatú zmenu ani nepotrebuje. `A` sa inicializuje náhodne a `B` nulami, takže na začiatku je `ΔW = 0` a model sa správa presne ako pôvodný — tréning teda štartuje z modelu, ktorý už funguje.

### Čo sa nastavuje

| Parameter | Význam | Bežná voľba |
|---|---|---|
| `r` | hodnosť rozkladu = kapacita adaptéra | 8–16 (štýl, formát), 32–64 (náročnejšia doména) |
| `lora_alpha` | škálovanie príspevku (`ΔW` sa násobí `alpha/r`) | typicky `2 × r` |
| `target_modules` | ktoré matice dostanú adaptér | projekcie attention (`q_proj`, `v_proj`), často aj `k_proj`, `o_proj` a FFN |
| `lora_dropout` | dropout na adaptéri proti preučeniu | 0,05–0,1 |
| `lr` | rýchlosť učenia | `1e-4` až `2e-4` — rádovo **viac** než pri plnom fine-tuningu |

Vyššie `r` neznamená automaticky lepší výsledok: pri malom datasete len urýchli preučenie. Začnite `r = 8`.

### Praktické dôsledky

- **Adaptér je malý súbor** (jednotky až desiatky MB) — dá sa verzovať, posielať, vymieňať za behu. Jeden základný model + päť adaptérov = päť špecializovaných modelov za cenu pamäte jedného.
- **Dá sa zlúčiť** (*merge*) späť do váh, ak chceme jeden samostatný model bez réžie navyše pri inferencii.
- **Nepridáva latenciu**, ak je zlúčený; ak nie, pridáva zanedbateľne.

---

## 3. QLoRA — LoRA na kvantizovanom modeli

**QLoRA** ide o krok ďalej: zamrznutý základný model sa načíta **kvantizovaný na 4 bity** (namiesto 2 bajtov na parameter ~0,5 bajtu), a LoRA adaptéry sa trénujú v bf16 nad ním. Výpočet prebieha tak, že sa váhy pri každom použití „rozbalia" späť do vyššej presnosti.

```text
7B model, plný fine-tuning        ~112 GB   → cloud, viac GPU
7B model, LoRA (bf16 základ)      ~20 GB    → 24 GB karta
7B model, QLoRA (4-bit základ)    ~10 GB    → bežná herná karta, Colab T4
```

Cena je mierna strata presnosti a pomalší tréning (kvôli rozbaľovaniu). Pre naše účely je to výborný obchod — a je to presne konfigurácia, v ktorej pobeží [zadanie 2B](zadania/RAG_Fine_tunning.md). Technicky to zabezpečia knižnice `peft` (adaptéry), `bitsandbytes` (4-bit) a `trl` (`SFTTrainer`); nastavenie prostredia je v [vyvojove-prostredie.md](vyvojove-prostredie.md).

---

## 4. Kedy fine-tuning áno a kedy nie

Toto je najdôležitejšia časť lekcie — mechaniku vám spraví knižnica, rozhodnutie nie.

### Oplatí sa

- **Štýl, tón, formát.** Firemný tón, doménový žargón (právo, medicína), spoľahlivý JSON alebo volanie funkcií presne v požadovanej štruktúre. Lacnejšie a spoľahlivejšie než opakovať dlhé inštrukcie v každom prompte.
- **Distillation.** Natrénovať malý lacný model na výstupoch veľkého pre jednu úzku úlohu. Výsledok: zlomok ceny a latencie pri veľkom objeme. Sem patrí aj prenos „uvažovania" veľkého modelu (*chain-of-thought distillation*).
- **Edge / on-device.** Keď musí model bežať lokálne (súkromie, offline, latencia) a je malý, doučenie na úzku úlohu mu výrazne pomôže.
- **Vyčerpané jednoduchšie možnosti.** Prompt aj RAG sú vyladené a presnosť či formát stále nestačia.

### Neoplatí sa

- **„Model nepozná fakt X."** Toto je najčastejší omyl. Fakty sedia vo váhach z pretrainingu a malý SFT dataset ich spoľahlivo neprepíše — model si skôr osvojí *štýl* vašich viet a fakty domieša. Na fakty patrí **RAG**.
- **Dáta sa často menia.** Fine-tuning treba pri každej zmene zopakovať; aktualizovať retrieval korpus je otázka minút.
- **Potrebujete citovať zdroj.** Fine-tunovaný model odpovedá „z hlavy" a nevie povedať, odkiaľ to má. RAG vracia `source` a `page` (viď metadáta v [embeddings.md](embeddings.md)).
- **Dobrý prompt už úlohu rieši.** Netreba pridávať zložitosť, ktorú niekto musí udržiavať.

### Rozhodovací postup

```text
Potrebujem, aby model poznal MOJE FAKTY?         ──► RAG
   (dokumenty, ktoré sa menia, treba citovať)

Potrebujem, aby model odpovedal INAK?            ──► fine-tuning (LoRA)
   (tón, formát, žargón, kratšie/lacnejšie)

Je dokumentov málo a zmestia sa do kontextu?     ──► len dlhý kontext + dobrý prompt
   (najlacnejšie na implementáciu, najdrahšie na token)

Treba oboje?                                      ──► fine-tuning na štýl + RAG na fakty
```

Tretia možnosť sa často prehliada: dnešné modely majú kontext v státisícoch tokenov, takže pri malej znalostnej báze môže stačiť vložiť **celý dokument do promptu**. Je to najjednoduchšie riešenie; naráža až na cenu za tokeny, latenciu a na to, že kvalita klesá, keď je podstatná informácia utopená v dlhom kontexte.

---

## 5. Ako zmerať, či to pomohlo

Fine-tuning aj RAG sa dajú „urobiť" a pritom nič nezlepšiť. Preto sa vyhodnocuje vždy proti **baseline** — surovému modelu bez úprav:

1. **Testovacia sada otázok so správnymi odpoveďami**, pripravená **pred** tréningom. Časť otázok nesmie byť v tréningových dátach — inak meriate memorovanie, nie schopnosť.
2. **Chytáky** — 2–3 otázky, ktorých odpoveď v dokumente **nie je**. Správna odpoveď znie „v texte to nie je uvedené". Bez nich sa nedá odlíšiť model, ktorý sa naučil obsah, od modelu, ktorý sa naučil sebavedomo tárať.
3. **Rovnaké nastavenie generovania** pre baseline aj upravený model (hlavne teplota — viď [dekódovanie](transformer-siete.md#ako-presne-sa-vyberá-ďalší-token-dekódovanie)), inak porovnávate dve rôzne veci.
4. **Úspešnosť zvlášť pre faktické otázky a zvlášť pre chytáky.**

**Halucinácia** je odpoveď, ktorá znie presvedčivo a nie je pravdivá. Nie je to porucha — je to priamy dôsledok toho, že model generuje **najpravdepodobnejšie pokračovanie**, nie overený fakt. Čo ju obmedzuje: RAG s inštrukciou odpovedať iba z kontextu, nízka teplota, vyžadovanie citácií a explicitné povolenie povedať „neviem". Fine-tuning ju typicky **zhoršuje**, ak sa ním snažíme vložiť fakty: model dostane sebavedomie v doméne bez toho, aby dostal spoľahlivé znalosti.

---

## Kontrolné otázky

1. Prečo plný fine-tuning 7B modelu potrebuje rádovo 100 GB pamäte, keď samotné váhy zaberú 14 GB?
2. Vysvetlite rozklad `ΔW = A · B`. Koľko parametrov sa trénuje pre maticu `4096 × 4096` pri `r = 16` a koľko percent z pôvodného počtu to je?
3. Prečo sa `B` inicializuje nulami? Čo by sa stalo, keby boli obe matice náhodné?
4. Čo presne pridáva QLoRA k LoRA a čo za to platíme?
5. Firma chce, aby model odpovedal na otázky z cenníka, ktorý sa mení každý mesiac, a aby vždy uviedol, z ktorej strany čerpal. RAG, fine-tuning, alebo dlhý kontext? Zdôvodnite.
6. Prečo je pri vyhodnotení nutné mať aj otázky, ktorých odpoveď v dokumente nie je?
7. Kolega tvrdí: „Dotrénoval som model na našej dokumentácii, takže už ju pozná." Ako overíte, či je to pravda, a čo najskôr uvidíte?

---

### Súvisiace dokumenty

- [prehlad-predmetu.md](prehlad-predmetu.md) — prehľad celého predmetu (8 lekcií)
- [llm-trening.md](llm-trening.md) — SFT vo veľkom; toto je tá istá fáza v malom (lekcia 5)
- [embeddings.md](embeddings.md) — **predchádzajúca lekcia**: RAG ako druhá cesta
- [zadania/RAG_Fine_tunning.md](zadania/RAG_Fine_tunning.md) — **zadanie 2**: RAG alebo LoRA na vlastnom dokumente
- [vyvojove-prostredie.md](vyvojove-prostredie.md) — koľko VRAM na to treba a kde to spustiť
- [agenti-a-nastroje.md](agenti-a-nastroje.md) — **nasledujúca lekcia**: agenti a nástroje

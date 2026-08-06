# Kam sa to hýbe — trendy a čo sledovať ďalej

> **Poradie čítania:** ← [Agenti, nástroje a Claude Code](01-agenti-a-nastroje.md) · **záver**

> **Cieľ dokumentu:** krátky záver predmetu — čo je v roku 2026 nové, čo z toho pravdepodobne zostane a ako sa v tejto oblasti orientovať, keď kurz skončí. Detailné mechaniky sú v predchádzajúcich dokumentoch; tu ide o výhľad, nie o učivo na skúšku.

---

## 1. Čo je dnes na hrane

- **Reasoning modely ako predvolená voľba.** Model dostane priestor „premýšľať" pred odpoveďou a kvalita rastie s množstvom výpočtu pri **inferencii**, nie len s veľkosťou modelu. Škálovanie sa tým presunulo z tréningu čiastočne do behu — čo mení aj ekonomiku: viac platíte za odpoveď, menej za tréning.
- **Agenti ako hlavný spôsob nasadenia.** Od „chatbot nad dokumentmi" k systémom, ktoré vykonávajú viackrokové úlohy s nástrojmi (viď [lekcia 8](01-agenti-a-nastroje.md)). S tým prichádza aj hlavné riziko obdobia — **bezpečnosť agentov**.
- **Malé a destilované modely popri veľkých.** Nie všetko potrebuje špičkový model; úzke úlohy sa presúvajú na malé lokálne modely kvôli cene, latencii a súkromiu (viď [lekcia 7](../04-llm/06-fine-tuning-lora.md), distillation).
- **Multimodalita ako samozrejmosť.** Obraz, zvuk a video na vstupe aj výstupe; hlasoví agenti pracujúci v reálnom čase.
- **Dlhý kontext.** Okná v státisícoch až miliónoch tokenov posúvajú hranicu, kedy ešte treba RAG a kedy stačí vložiť celý dokument do promptu.

---

## 2. Čo sa oplatí sledovať po kurze

Nie zoznam nástrojov — tie zastarajú. Skôr miesta, kde sa dá overiť, čo je aktuálne:

- **Modelové karty na [Hugging Face](https://huggingface.co/models)** — primárny zdroj o konkrétnom modeli (veľkosť, licencia, jazyky, benchmarky). Vždy lepší než blogový článok „top 10 modelov".
- **arXiv** pre pôvodné články; väčšina pojmov z tohto kurzu (Attention Is All You Need, LoRA, InfoNCE, RAG) má jeden zakladajúci článok, ktorý sa dá prečítať za hodinu.
- **Dokumentácia poskytovateľov API** — parametre, ceny aj limity sa menia rýchlejšie než čokoľvek iné.
- **Vlastná testovacia sada.** Najspoľahlivejší spôsob, ako zistiť, či je nový model lepší *pre vašu úlohu*, je pustiť naň svojich pätnásť otázok — presne ako v [zadaní 2](../../zadania/RAG_Fine_tunning.md). Verejné benchmarky o vašej doméne nehovoria nič.

---

## 3. Čo z tohto kurzu nezastará

Konkrétne verzie modelov, knižníc aj frameworkov sa vymenia. Mechanika nie:

- **tréningová slučka** — forward → loss → backprop → update je rovnaká pre malú sieť z [lekcie 3](../03-ucenie/01-adam-optimalizator.md) aj pre model s biliónom parametrov,
- **attention** — jadro každého dnešného jazykového modelu ([lekcia 4](../04-llm/01-transformer-siete.md)),
- **dáta a loss určujú, čo sa model naučí** — vysvetľuje rozdiel medzi base a Instruct modelom, embedding a generatívnym modelom, aj to, prečo fine-tuning nefunguje na fakty,
- **typ dát určuje model** — na tabuľky stále XGBoost, na obraz CNN. Táto vec sa za desať rokov nezmenila a pravdepodobne sa ani nezmení,
- **vyhodnotenie proti baseline** — bez merania sa nedá povedať, že niečo pomohlo. Platí to pri sieti zo zadania 1 rovnako ako pri agentovi.

Kto rozumie týmto piatim veciam, vie si nový model, novú knižnicu aj nový buzzword zaradiť sám.

---

### Súvisiace dokumenty

- [prehlad-predmetu.md](../../prehlad-predmetu.md) — prehľad celého predmetu (8 lekcií)
- [01-agenti-a-nastroje.md](01-agenti-a-nastroje.md) — agentová slučka, MCP, bezpečnosť (lekcia 8)
- [03-llm-modely.md](../04-llm/03-llm-modely.md) — ako si vybrať model a na čo si dať pozor právne (lekcia 5)
- [04-embeddings.md](../04-llm/04-embeddings.md) — RAG vrátane pokročilého retrievalu (lekcia 6)
- [06-fine-tuning-lora.md](../04-llm/06-fine-tuning-lora.md) — LoRA, distillation, RAG vs. fine-tuning (lekcia 7)

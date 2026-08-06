# Režimy strojového učenia

> **Poradie čítania:** ← [Čo je umelá inteligencia](01-co-je-ai.md) · **lekcia 1** · [Zovšeobecnenie a preučenie](03-generalizacia-a-preucenie.md) →

Podľa toho, aké dáta máme k dispozícii a čo od modelu chceme, rozlišujeme tri hlavné režimy učenia:

| Režim | Čo máme | Čo sa učí | Typický príklad |
|---|---|---|---|
| **Učenie s učiteľom** (*supervised*) | vstupy **aj správne odpovede** (labely) | mapovanie vstup → výstup | „táto fotka = mačka", predikcia ceny bytu |
| **Učenie bez učiteľa** (*unsupervised*) | len vstupy, **bez labelov** | štruktúra, zhluky, podobnosti | segmentácia zákazníkov, [embeddingy](../04-llm/04-embeddings.md) |
| **Posilňované učenie** (*reinforcement*) | prostredie + **odmena** za akcie | stratégia (politika) maximalizujúca odmenu | hra Go, riadenie robota, [demo s tankom](../../../demo/Readme.md) |

Väčšina modelov v tomto kurze (stromy, XGBoost, klasifikačné siete) sú príklady **učenia s učiteľom**. Spoločná schéma je vždy rovnaká:

```text
  trénovacie dáta ──►  MODEL  ──► predpoveď
                         ▲            │
                         │            ▼
                    úprava parametrov ◄── porovnaj s pravdou (loss)
```

Model urobí predpoveď, porovná ju so správnou odpoveďou (chyba = *loss*), a upraví svoje parametre tak, aby chyba klesala. Toto sa opakuje na tisícoch príkladov. Detailne je tréningová slučka a optimalizátor rozpísaný v [01-adam-optimalizator.md](../03-ucenie/01-adam-optimalizator.md).

Ešte jedno praktické rozdelenie dát, ktoré sa ťahá celým ML:

- **Tabuľkové dáta** — riadky a stĺpce (Excel, databáza): vek, príjem, počet klikov… Tu dnes **kraľujú stromové metódy a XGBoost**.
- **Neštruktúrované dáta** — obraz, zvuk, text, video. Tu **kraľuje hlboké učenie** (CNN pre obraz, transformery pre text).

Toto rozlíšenie je najdôležitejšia intuícia pri výbere modelu, preto sa k nemu budeme vracať pri každej rodine.

---

## Kontrolné otázky

1. Zaraďte do správneho režimu: segmentácia zákazníkov, predikcia ceny bytu, robot učiaci sa chodiť, detekcia podvodov.
2. Popíšte spoločnú schému učenia s učiteľom v štyroch krokoch.
3. Prečo je rozdelenie na tabuľkové vs. neštruktúrované dáta najdôležitejšia intuícia pri výbere modelu?

---

### Súvisiace dokumenty

- [03-generalizacia-a-preucenie.md](03-generalizacia-a-preucenie.md) — **nasleduje**: prečo nestačí uspieť na trénovacích dátach
- [01-adam-optimalizator.md](../03-ucenie/01-adam-optimalizator.md) — tréningová slučka do detailu

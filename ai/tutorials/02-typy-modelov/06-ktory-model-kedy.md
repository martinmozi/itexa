# Ktorý model kedy — rozhodovacia tabuľka

> **Poradie čítania:** ← [Konvolučné siete (CNN)](05-konvolucne-siete.md) · **zhrnutie lekcií 1–3** · [Adam — optimalizátor](../03-ucenie/01-adam-optimalizator.md) →

| Dáta / úloha | Odporúčaný prvý model | Prečo |
|---|---|---|
| **Tabuľkové dáta** (riadky × stĺpce) | **XGBoost** / random forest | najvyššia presnosť, málo dát, netreba GPU |
| Potrebujem **vysvetliteľnosť** | rozhodovací strom (+ SHAP na XGBoost) | čitateľná cesta k rozhodnutiu |
| **Obraz** (klasifikácia, detekcia) | **CNN** | weight sharing, hierarchia príznakov |
| **Text / postupnosti / jazyk** | **transformer** → [01-transformer-siete.md](../04-llm/01-transformer-siete.md) | attention, kontext, dnešné LLM |
| Univerzálny nelineárny vzťah, koncová hlava | **feed-forward (MLP)** | jednoduchý, univerzálny aproximátor |

**Najdôležitejšie pravidlo:** typ dát určuje model viac než čokoľvek iné. Na tabuľky nasadzujte stromy/XGBoost, na obraz CNN, na text transformery — a neurónovú sieť neťahajte tam, kde jednoduchší model spraví rovnakú prácu lacnejšie a vysvetliteľnejšie.

---

## Kontrolné otázky

1. Banka rieši predikciu nesplácania úverov z tabuľky s 50 stĺpcami. Kolega navrhuje hlbokú neurónovú sieť. Aký model navrhnete vy a ako to obhájite?
2. Pre každý riadok tabuľky vyššie povedzte, prečo tam nepatrí ten druhý najbližší model.
3. Kedy sa oplatí siahnuť po jednoduchšom modeli, aj keď zložitejší dáva o niečo vyššiu presnosť?

---

### Súvisiace dokumenty

- [01-adam-optimalizator.md](../03-ucenie/01-adam-optimalizator.md) — **nasleduje**: ako sa neurónové siete trénujú
- [01-transformer-siete.md](../04-llm/01-transformer-siete.md) — modely pre text a postupnosti
- [03-llm-modely.md](../04-llm/03-llm-modely.md) — tá istá otázka o úroveň vyššie: ktorý LLM kedy

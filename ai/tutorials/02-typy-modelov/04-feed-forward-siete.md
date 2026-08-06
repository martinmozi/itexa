# Feed-forward neurónové siete (MLP)

> **Poradie čítania:** ← [XGBoost krok za krokom (ISO 8583)](03-xgboost-priklad-iso8583.md) · **lekcia 3** · [Konvolučné siete (CNN)](05-konvolucne-siete.md) →

**Feed-forward neurónová sieť** (viacvrstvový perceptrón, *MLP*) je najzákladnejší typ neurónovej siete. Skladá sa z vrstiev neurónov; informácia tečie **jedným smerom** — od vstupu cez skryté vrstvy k výstupu, bez cyklov.

![Feed-forward sieť: vstupná vrstva, skrytá vrstva a výstupná vrstva, prepojené váhami](../../images/ff-siet-prehlad.svg)

Každý neurón spočíta vážený súčet svojich vstupov, pripočíta **bias** a prevedie výsledok cez nelineárnu **aktivačnú funkciu** (ReLU, sigmoid…):

![Detail jedného neurónu: vstupy vážené váhami w, pripočítaný bias b, výsledok z prejde aktivačnou funkciou σ na výstup a](../../images/neuron-detail.svg)

## Prečo sú nelineárne aktivácie nevyhnutné

Predstavme si na chvíľu sieť **bez** aktivačných funkcií — každá vrstva by počítala len vážený súčet, teda lineárne zobrazenie y = W·x + b. Čo spraví druhá vrstva s výstupom prvej?

```text
  y = W₂ · (W₁ · x + b₁) + b₂  =  (W₂ · W₁) · x + (W₂ · b₁ + b₂)
```

Výsledok je opäť len vážený súčet pôvodných vstupov — s inou maticou váh a iným biasom. Inak povedané: **zloženie ľubovoľného počtu lineárnych vrstiev je stále jedna lineárna vrstva.** Sieť so sto vrstvami by nedokázala nič viac než obyčajná lineárna regresia — nevedela by oddeliť ani body, ktoré sa nedajú rozdeliť priamkou (klasický príklad je funkcia XOR). Pridávanie ďalších vrstiev by nepomohlo vôbec, len by pribúdali parametre.

Nelineárna aktivácia vložená medzi vrstvy toto „zrútenie" zlomí. Najpoužívanejšia **ReLU** je pritom prekvapivo jednoduchá: záporné hodnoty vynuluje, kladné nechá tak — max(0, z). **Sigmoid** stláča výstup do intervalu (0, 1), preto sa hodí na výstupnú vrstvu, keď má výstup vyjadrovať pravdepodobnosť. Vďaka nelinearite môže každá ďalšia vrstva rozhodovaciu hranicu „ohýbať" — a platí **veta o univerzálnej aproximácii**: už sieť s jednou dostatočne širokou nelineárnou skrytou vrstvou dokáže aproximovať ľubovoľnú spojitú funkciu. V praxi sa namiesto jednej obrovskej vrstvy používa viac menších — hlbšia sieť sa tú istú vec spravidla naučí s menším počtom neurónov.

Ako sa váhy a biasy ladia tréningom (forward pass → loss → backpropagation → update optimalizátorom Adam), podrobne rozoberá [01-adam-optimalizator.md](../03-ucenie/01-adam-optimalizator.md).

**Typické použitie:** univerzálny „lepiaci" model — klasifikácia a regresia na stredne veľkých dátach, koncové vrstvy v zložitejších sieťach (napr. klasifikačná hlava CNN alebo transformera), aproximácia funkcií v simuláciách.

| ✅ Výhody | ❌ Nevýhody |
|---|---|
| **Univerzálny aproximátor** — teoreticky zvládne ľubovoľný vzťah | Ignoruje štruktúru dát (pri obraze nevie, že susedné pixely spolu súvisia) |
| Základný stavebný blok všetkých hlbokých sietí | Veľa parametrov → **potrebuje veľa dát**, ľahko sa preučí |
| Zvláda nelineárne vzťahy, ktoré strom ťažko | Na tabuľkových dátach ho **XGBoost často predbehne** |
| Beží dobre na GPU | Menej vysvetliteľný — „čierna skrinka" |

---

## Kontrolné otázky

1. Čo by sa stalo, keby mala MLP sieť len lineárne aktivácie (žiadne ReLU/sigmoid)? Prečo by potom nepomáhalo pridávať vrstvy?
2. Ručne prepočítajte výstup neurónu s dvoma vstupmi, danými váhami, biasom a ReLU aktiváciou.
3. Prečo sa MLP na tabuľkových dátach zvyčajne neoplatí, hoci je univerzálnym aproximátorom?

---

### Súvisiace dokumenty

- [05-konvolucne-siete.md](05-konvolucne-siete.md) — **nasleduje**: čo pridáva CNN oproti MLP
- [01-adam-optimalizator.md](../03-ucenie/01-adam-optimalizator.md) — ako sa táto sieť trénuje
- [02-problemy-pri-uceni.md](../03-ucenie/02-problemy-pri-uceni.md) — mŕtve ReLU neuróny, miznúce gradienty

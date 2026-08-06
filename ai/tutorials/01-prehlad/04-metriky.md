# Metriky — čím sa meria kvalita modelu

> **Poradie čítania:** ← [Zovšeobecnenie a preučenie](03-generalizacia-a-preucenie.md) · **lekcia 1** · [Rozhodovacie stromy](../02-typy-modelov/01-rozhodovacie-stromy.md) →

**Presnosť** (*accuracy*, podiel správne zaradených príkladov) je najjednoduchšia metrika, ale pri **nevyvážených triedach** klame: ak je 99 % transakcií poctivých, model „nikdy to nie je podvod" má 99 % presnosť a nulovú užitočnosť. Preto sa pri klasifikácii pozeráme na **maticu zámen** (*confusion matrix*) — tabuľku skutočná × predpovedaná trieda — a z nej na tri čísla:

- **Precision** — z tých, ktoré model označil za pozitívne, koľko naozaj pozitívnych je? (koľko poplachov bolo falošných)
- **Recall** (senzitivita) — zo skutočne pozitívnych, koľko ich model našiel? (koľko prípadov prehliadol)
- **F1** — harmonický priemer precision a recall; jedno číslo, keď záleží na oboch naraz.

Ktoré z nich je dôležitejšie, určuje úloha: pri filtri spamu bolí falošný poplach (dôraz na precision), pri skríningu choroby bolí prehliadnutý prípad (dôraz na recall). Pri regresii sa namiesto toho používa **MAE** (priemerná absolútna chyba) alebo **RMSE** (odmocnina z priemernej štvorcovej chyby, tvrdšie trestá veľké omyly).

---

## Kontrolné otázky

1. Detektor podvodov má 99 % presnosť, ale nezachytil ani jeden skutočný podvod. Ako je to možné a ktorými metrikami to odhalíte?
2. Kedy je dôležitejšia precision a kedy recall? Uveďte ku každému prípadu príklad.
3. Prečo sa pri regresii používa MAE alebo RMSE a v čom sa líšia?

---

### Súvisiace dokumenty

- [tutorials/02-typy-modelov](../02-typy-modelov/README.md) — **nasleduje**: konkrétne rodiny modelov
- [03-xgboost-priklad-iso8583.md](../02-typy-modelov/03-xgboost-priklad-iso8583.md) — PR-AUC a nevyvážené triedy na reálnom príklade

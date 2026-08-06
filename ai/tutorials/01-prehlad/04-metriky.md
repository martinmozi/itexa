# Metriky — čím sa meria kvalita modelu

> **Poradie čítania:** ← [Zovšeobecnenie a preučenie](03-generalizacia-a-preucenie.md) · **lekcia 1** · [Rozhodovacie stromy](../02-typy-modelov/01-rozhodovacie-stromy.md) →

**Presnosť** (*accuracy*, podiel správne zaradených príkladov) je najjednoduchšia metrika, ale pri **nevyvážených triedach** klame: ak je 99 % transakcií poctivých, model „nikdy to nie je podvod" má 99 % presnosť a nulovú užitočnosť. Preto sa pri klasifikácii pozeráme na **maticu zámen** (*confusion matrix*) — tabuľku skutočná × predpovedaná trieda:

| | model povedal **áno** | model povedal **nie** |
|---|---|---|
| **v skutočnosti áno** | TP (správny záchyt) | FN (**prehliadnutý** prípad) |
| **v skutočnosti nie** | FP (**falošný poplach**) | TN (správne prepustený) |

Accuracy je `(TP + TN) / všetko` — a práve preto klame: pri 99 % negatívnych ju drží hore samotné veľké TN. Užitočné sú tri čísla, v ktorých TN vôbec nevystupuje:

- **Precision** = `TP / (TP + FP)` — z tých, ktoré model označil za pozitívne, koľko naozaj pozitívnych je? (koľko poplachov bolo falošných)
- **Recall** (senzitivita) = `TP / (TP + FN)` — zo skutočne pozitívnych, koľko ich model našiel? (koľko prípadov prehliadol)
- **F1** — harmonický priemer precision a recall; jedno číslo, keď záleží na oboch naraz.

Ktoré z nich je dôležitejšie, určuje úloha: pri filtri spamu bolí falošný poplach (dôraz na precision), pri skríningu choroby bolí prehliadnutý prípad (dôraz na recall).

## Prah a PR-AUC

Modely spravidla nevracajú „áno / nie", ale **pravdepodobnosť**. Až my z nej rozhodnutie urobíme tým, že zvolíme **prah** — napríklad „nad 0,5 to označ za podvod". Prah je otočný gombík medzi precision a recall: keď ho zdvihneme, model označí menej prípadov, tie sú istejšie (precision rastie), ale viac ich prehliadne (recall klesá). Precision, recall aj F1 teda **nie sú vlastnosti modelu, ale vlastnosti modelu pri konkrétnom prahu**.

Keď chceme porovnať dva modely bez toho, aby výsledok závisel od zvoleného prahu, prejdeme **všetky** prahy naraz a vykreslíme precision proti recallu. Plocha pod touto krivkou je **PR-AUC** (*precision-recall area under curve*, v knižniciach aj `average_precision`) — jedno číslo, ktoré hovorí „ako dobre model vie zoradiť pozitívne prípady nad negatívne". Pri veľmi nevyvážených úlohách je to hlavná metrika: bezcenný model má PR-AUC rovné podielu pozitívnych v dátach (pri 0,1 % podvodov teda 0,001), takže na rozdiel od accuracy sa nedá vylepšiť tým, že model nerobí nič.

Pri regresii sa namiesto toho všetkého používa **MAE** (priemerná absolútna chyba) alebo **RMSE** (odmocnina z priemernej štvorcovej chyby, tvrdšie trestá veľké omyly).

---

## Kontrolné otázky

1. Detektor podvodov má 99 % presnosť, ale nezachytil ani jeden skutočný podvod. Ako je to možné a ktorými metrikami to odhalíte?
2. Kedy je dôležitejšia precision a kedy recall? Uveďte ku každému prípadu príklad.
3. Prečo sa precision a recall nedajú zlepšiť naraz obyčajným posunutím prahu? Čo sa stane s každou z nich, keď prah zdvihnete?
4. Prečo sa pri detekcii podvodov sleduje PR-AUC a nie accuracy? Akú hodnotu má PR-AUC modelu, ktorý háda náhodne?
5. Prečo sa pri regresii používa MAE alebo RMSE a v čom sa líšia?

---

### Súvisiace dokumenty

- [tutorials/02-typy-modelov](../02-typy-modelov/README.md) — **nasleduje**: konkrétne rodiny modelov
- [03-xgboost-priklad-iso8583.md](../02-typy-modelov/03-xgboost-priklad-iso8583.md) — PR-AUC a nevyvážené triedy na reálnom príklade

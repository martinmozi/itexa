# 03 — Tréning modelov: ako sa model naozaj natrénuje

Nie je to prednáška o teórii — je to **remeslo tréningu**. Jedna a tá istá myšlienka
(*zmeraj chybu → zisti, ktorým smerom ju zmenšiť → urob krok*) sa tu prechádza na troch
rôznych typoch modelu: na feed-forward sieti, na gradient boostingu a na transformeri.
Ak porozumiete tomu, čo majú spoločné a v čom sa líšia, viete natrénovať prakticky čokoľvek.

| Dokument | O čom je | Lekcia |
|---|---|---|
| [01-adam-optimalizator.md](01-adam-optimalizator.md) | anatómia siete, aktivačné funkcie, momentum a adaptívny krok, bias correction, referenčná implementácia v NumPy, sedlové body, varianty pre lepšiu konvergenciu (AdamW, AMSGrad, Nadam, RAdam, AdaBelief, clipping, lr rozvrh) | 3 |
| [02-problemy-pri-uceni.md](02-problemy-pri-uceni.md) | **katalóg porúch**: miznúce a explodujúce gradienty, mŕtve ReLU, inicializácia, learning rate, normalizácia, `NaN`, chyby v dátach, fp16/bf16, nedeterminizmus, hardvér a ECC | 3 |
| [03-xgboost-trening-a-inferencia.md](03-xgboost-trening-a-inferencia.md) | **celý cyklus na tabuľkových dátach**: príprava dát, dve API XGBoostu, **algoritmus pod kapotou** (`g` a `h`, `w* = −G/(H+λ)`, zisk zo splitu, histogramy, chýbajúce hodnoty) s referenčnou implementáciou v NumPy, tréning s early stoppingom, ladenie hyperparametrov, krížová validácia, voľba prahu, uloženie modelu, dávková aj online inferencia, SHAP, katalóg chýb | 3 |
| [04-trening-transformera.md](04-trening-transformera.md) | **čo do tréningu vnáša attention**: škálovanie `1/√d_head` a saturovaný softmax, prečo je AdamW povinný a `β₂ = 0,95`, warmup, clipping, pre-LN, pamäťový rozpočet a FlashAttention, sanity checky a poruchy špecifické pre transformer | 4 |

**Ako to čítať.** Prvý dokument je špecifikácia, podľa ktorej sa dá Adam naprogramovať (zadanie 1).
Druhý je referenčná príručka — otvorte ju vždy, keď sa sieť neučí tak, ako má.
Tretí ukazuje to isté remeslo na druhej strane sveta modelov: XGBoost od CSV až po službu,
ktorá skóruje transakcie — a rovnako ako prvý obsahuje **referenčnú implementáciu**, podľa ktorej
sa dá celý algoritmus naprogramovať a overiť proti knižnici.
Štvrtý vyžaduje attention, preto sa číta **až po** [Vnútre transformera](../04-llm/02-transformer-vnutro.md)
(lekcia 4); v súbornom poradí ho nájdete tu, lebo patrí k tréningu, nie k architektúre.

| | sieť (01, 02) | XGBoost (03) | transformer (04) |
|---|---|---|---|
| čo je jeden krok | update váh podľa gradientu | pridanie stromu na rezíduá | update váh podľa gradientu |
| kedy prestať | validačná loss prestane klesať | `early_stopping_rounds` | rozvrh krokov / vyčerpané tokeny |
| hlavné riziko | miznúci gradient, `NaN` | preučenie a data leakage | saturovaná attention, loss spike, pamäť |

← [02 — Typy modelov](../02-typy-modelov/README.md) · ďalej [04 — LLM](../04-llm/README.md) →

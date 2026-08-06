# Zovšeobecnenie a preučenie

> **Poradie čítania:** ← [Režimy strojového učenia](02-rezimy-strojoveho-ucenia.md) · **lekcia 1** · [Metriky](04-metriky.md) →

Cieľom učenia nie je, aby model bezchybne zvládol trénovacie príklady — ich správne odpovede predsa už poznáme. Cieľom je **zovšeobecnenie (generalizácia)**: správne predpovedať aj na dátach, ktoré model počas tréningu nikdy nevidel. Preto sa dostupné dáta pred trénovaním rozdelia: väčšina (typicky okolo 80 %) tvorí **trénovaciu množinu**, na ktorej sa model učí, a zvyšok sa odloží bokom ako **testovacia množina**, na ktorej sa až na záver zmeria, ako model obstojí na neznámych príkladoch. Chyba na testovacej množine je jediný poctivý odhad kvality modelu — chyba na trénovacej množine sa dá „vylepšiť" obyčajným memorovaním.

Len čo však začneme **ladiť nastavenia** (koľko vrstiev, aká hĺbka stromu, aký learning rate), potrebujeme ešte tretiu množinu. Keby sme si najlepšie nastavenie vybrali podľa testovacej množiny, nepriamo by sme sa na ňu „preučili" — vybrali by sme to, čo náhodou vyhovuje práve jej, a odhad kvality by prestal byť poctivý. Dáta sa preto delia na tri časti:

| Množina | Podiel | Na čo slúži |
|---|---|---|
| **trénovacia** | ~60–80 % | učia sa na nej parametre (váhy, splity) |
| **validačná** | ~10–20 % | porovnávajú sa na nej nastavenia a rozhoduje sa, kedy tréning zastaviť |
| **testovacia** | ~10–20 % | siahne sa na ňu **raz, na úplný záver** |

Ak je dát málo na to, aby sa dala odkrojiť samostatná validačná časť, nahrádza ju **krížová validácia** (*k-fold*): dáta sa rozdelia na `k` dielov, model sa `k`-krát natrénuje vždy na `k−1` dieloch a overí na tom zvyšnom, a výsledky sa spriemerujú.

Pri učení hrozia dva opačné neduhy:

- **Preučenie (overfitting):** model je príliš pružný a naučí sa trénovacie dáta doslova naspamäť — vrátane náhodného šumu a výnimiek, ktoré sa už nikdy nezopakujú. Poznávacie znamenie: na trénovacej množine takmer nulová chyba, na testovacej výrazne horšia. Model si nezapamätal vzor, ale konkrétne príklady.
- **Nedoučenie (underfitting):** model je naopak príliš jednoduchý na to, aby vzor v dátach vôbec zachytil — chybuje na trénovacej aj testovacej množine. Typický obraz: zjavne zakrivený vzťah sa snažíme preložiť priamkou.

S tým súvisí užitočný rozklad chyby na dve zložky, ktorý budeme potrebovať pri ansámbloch stromov:

- **Skreslenie (bias)** je systematická chyba príliš jednoduchého modelu. Nech ho trénujeme na akejkoľvek vzorke, na skutočný vzor jednoducho „nedosiahne" — vždy sa mýli podobným smerom.
- **Rozptyl (variance)** je nestálosť príliš pružného modelu. Na každej trénovacej vzorke sa naučí niečo trochu iné; jeho predpovede „lietajú" podľa toho, aké dáta náhodou dostal.

Jednoduché modely mávajú vysoké skreslenie a nízky rozptyl, zložité modely naopak. Umenie strojového učenia spočíva v hľadaní rovnováhy medzi nimi — alebo, ako uvidíme pri random foreste a boostingu, v šikovnom zložení viacerých modelov tak, aby sa jedna zo zložiek chyby potlačila.

## Ako sa preučenie brzdí

Okrem voľby jednoduchšieho modelu má proti preučeniu každá rodina svoje nástroje — spoločne sa im hovorí **regularizácia**:

- **obmedzenie zložitosti** — maximálna hĺbka stromu, minimálny počet vzoriek v liste, menej neurónov a vrstiev,
- **pokuta za veľké váhy** (*weight decay*, L2) — k chybovej funkcii sa pripočíta trest úmerný veľkosti váh, takže model uprednostní „hladšie" riešenie pred divoko kmitajúcim,
- **dropout** (len neurónové siete) — počas tréningu sa v každom kroku náhodne „vypne" časť neurónov (typicky 10–50 %), takže sa sieť nemôže spoľahnúť na jeden konkrétny neurón a musí si vzor uložiť redundantne; pri predikcii sú zapnuté všetky,
- **skoré zastavenie** (*early stopping*) — tréning sa ukončí vo chvíli, keď chyba na **validačnej** množine prestane klesať, hoci na trénovacej ešte klesá; presne to robí [XGBoost](../02-typy-modelov/02-random-forest-a-xgboost.md),
- **viac dát** — najúčinnejší liek; ak sa nedajú získať, pomôže **umelé rozšírenie** (*augmentácia*): pri obrázkoch posun, otočenie, orezanie či zmena jasu.

---

## Kontrolné otázky

1. Načo je popri trénovacej a testovacej množine ešte tretia, validačná? Čo presne sa pokazí, ak si architektúru vyberiete podľa presnosti na testovacej množine?
2. Rozlíšte preučenie a nedoučenie podľa toho, ako vyzerá chyba na trénovacej a testovacej množine.
3. Vysvetlite skreslenie (bias) a rozptyl (variance) vlastnými slovami a povedzte, ktorý z nich má jednoduchý model vysoký.
4. Vymenujte tri spôsoby, ako brzdiť preučenie, a povedzte, čo každý z nich robí.

---

### Súvisiace dokumenty

- [04-metriky.md](04-metriky.md) — **nasleduje**: čím sa kvalita modelu meria
- [02-problemy-pri-uceni.md](../03-ucenie/02-problemy-pri-uceni.md) — preučenie v praxi: ako ho spoznať z krivky lossu
- [02-random-forest-a-xgboost.md](../02-typy-modelov/02-random-forest-a-xgboost.md) — ansámble ako spôsob, ako potlačiť rozptyl či skreslenie

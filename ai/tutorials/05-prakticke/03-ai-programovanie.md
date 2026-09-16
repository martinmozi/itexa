# AI pri programovaní — nástroje, kontext a čo je dnes trend

> **Poradie čítania:** ← [Agenti, nástroje a MCP](02-agenti-a-nastroje.md) · **lekcia 8** · [Trendy — čo sledovať ďalej](05-llm-trendy.md) →

> **Cieľ dokumentu:** zorientovať sa v tom, ako dnes vyzerá práca programátora s AI. Čo sú Claude Code, Codex a GitHub Copilot a v čom sa líšia; ako sa agentovi dáva kontext (`CLAUDE.md` / `AGENTS.md`, skills, MCP, hooks); ako s tým reálne pracovať, aby z toho bol použiteľný kód — a hlavne **čo je dnešný trend a čo sa už nepoužíva**.
>
> *Stav: september 2026. Nástroje v tejto oblasti sa menia každé dva-tri mesiace; kategórie a pracovné postupy sú trvácnejšie než konkrétne názvy.*

Nadväzuje na [02-agenti-a-nastroje.md](02-agenti-a-nastroje.md) — všetko nižšie je tá istá agentová slučka, len s nástrojmi na prácu s repozitárom. Ak ste ju nečítali, prečítajte najprv ju; inak sú tieto nástroje čierne skrinky.

---

## 1. Prečo je práve programovanie pre AI výnimočné

Nie preto, že by bol kód „ľahší" než iný text. Ale preto, že pri kóde ako v jedinom obore platí všetko naraz:

- **kód je text** — presne to, čo model generuje,
- **tréningových dát je obrovské množstvo** a sú verejné (GitHub),
- **výsledok sa dá overiť strojom** — kompilátor, testy, linter, typová kontrola. Model dostane **spätnú väzbu bez človeka** a môže iterovať, kým to nesedí.

Práve tá tretia vec je dôvod, prečo AI v programovaní bežala rýchlejšie než inde. Uzavretá slučka *sprav → spusti testy → oprav* je presne agentová slučka z [predchádzajúceho dokumentu](02-agenti-a-nastroje.md), len s obzvlášť dobrým signálom o úspechu. Tam, kde taký signál chýba (dizajn, texty, rozhodnutia), je pokrok pomalší — a to vysvetľuje aj hranice, o ktorých je [sekcia 7](#7-kde-to-zlyháva).

### Štyri generácie za päť rokov

```text
2021  doplňovanie riadku      Copilot v editore: šedý text, Tab
2023  chat v editore          „vysvetli mi tento súbor", copy-paste
2024  agent nad repozitárom   číta, píše, spúšťa testy, commituje
2026  agent v pozadí / v cloude  dostane issue, vráti pull request
```

Každá generácia tú predchádzajúcu **nezrušila** — doplňovanie riadku sa stále používa a je na svoju úlohu ideálne. Zmenilo sa, kde sa odohráva ťažisko práce a čo je dnes „bežná" úloha pre AI.

---

## 2. Nástroje 2026 — čo je čo

| Nástroj | Typ | Kde beží | Silné stránky |
|---|---|---|---|
| **Claude Code** (Anthropic) | agent | terminál, rozšírenie do VS Code/JetBrains, desktop, web | dlhé viackrokové úlohy nad celým repozitárom, práca s gitom, rozšíriteľnosť (MCP, skills, hooks, podagenti) |
| **Codex** (OpenAI) | agent | terminál (CLI), IDE, cloud | lokálna aj cloudová podoba — úlohu delegujete a vráti sa s hotovou zmenou / pull requestom |
| **GitHub Copilot** | doplňovanie + chat + agent | editor a GitHub | najtesnejšia integrácia s GitHubom (issues, PR, review), najnižší prah vstupu, firemné nasadenie |
| **Cursor**, **Windsurf** | AI-first IDE | vlastný editor (fork VS Code) | doplňovanie „na steroidoch", agent priamo v editore, veľmi dobrá práca s viacerými súbormi |
| **Cline**, **Aider**, **OpenHands**, **Continue** | agenti, open source | rozšírenie / terminál | vlastná voľba modelu (aj lokálneho), transparentnosť, žiadna viazanosť na poskytovateľa |
| **Jules**, **Amazon Q Developer**, **Devin** a ďalší | agenti v cloude | webové rozhranie / CI | asynchrónne delegovanie úloh, hromadné zmeny naprieč repozitármi |

**Tri veci, ktoré z tejto tabuľky stoja za zapamätanie:**

1. **Hranice medzi nástrojmi sa stierajú.** Copilot dnes tiež vie agenta, Cursor tiež vie doplňovať riadky, Claude Code aj Codex majú lokálnu aj cloudovú podobu. Vyberať podľa marketingových kategórií je zbytočné — rozhoduje, kde máte repozitár, aký model chcete používať a či vám vyhovuje terminál alebo editor.
2. **Nástroj nie je model.** Väčšina nástrojov vie bežať nad viacerými modelmi. Kvalita výsledku je z veľkej časti vlastnosť modelu, kvalita *práce* je vlastnosť nástroja (ako spravuje kontext, ako vie spúšťať príkazy, ako sa integruje).
3. **Ekosystém je zámerne kompatibilný.** MCP servery, `AGENTS.md` aj skills sú otvorené formáty, ktoré prijalo viacero nástrojov naraz — to, čo si raz pripravíte pre projekt, prežije výmenu nástroja.

### Claude Code ako reprezentatívna ukážka

Stojí za to pozrieť sa na jeden nástroj podrobnejšie, lebo je to **priama ukážka agentovej slučky** z predchádzajúcej lekcie, nie čierna skrinka. Jeho nástroje sú presne tie, ktoré potrebuje vývojár: čítanie a zápis súborov, hľadanie v projekte (`grep`, `glob`), spúšťanie príkazov v termináli, práca s gitom, prehliadanie webu — a čokoľvek doplníte cez MCP.

Kde reálne pomáha:

- zorientovať sa v cudzom repozitári („kde sa spracúva prihlásenie?"),
- mechanická, ale rozsiahla práca — premenovanie naprieč projektom, doplnenie testov, migrácia knižnice,
- prvý návrh riešenia, ktorý potom upravíte,
- rutina okolo kódu: commit správy, changelog, reprodukcia chyby z logu.

**Kedy mu neveriť:** agent má tendenciu tvrdiť, že je hotový. Overujte tri veci — či testy naozaj prešli (pozrite výstup, nie zhrnutie), či nezmenil viac, než mal (`git diff`), a či navrhnutá knižnica či API vôbec existuje. Platí to isté, čo v [lekcii 7](../04-llm/07-fine-tuning-lora.md) pri halucináciách: model generuje najpravdepodobnejšie pokračovanie, nie overený fakt.

> **A jedna vec z pohľadu tohto predmetu:** pri zadaniach je cieľom pochopiť mechaniku vlastnými rukami. Agentom si dajte **vysvetľovať, nie riešiť** — inak odovzdáte kód, ktorý neviete obhájiť, a to je na skúške poznať okamžite.

---

## 3. Ako sa agentovi dáva kontext

Toto je dnes to skutočné remeslo. Nie formulácia jednej otázky, ale **príprava prostredia, v ktorom agent pracuje**. Štyri vrstvy, od najlacnejšej po najmocnejšiu.

### 3.1 Súbor s pravidlami projektu — `CLAUDE.md` / `AGENTS.md`

Textový súbor v koreni repozitára, ktorý si agent načíta na začiatku každého sedenia. Claude Code ho hľadá ako `CLAUDE.md`, časť ekosystému sa zjednotila na názve `AGENTS.md`, Copilot má `.github/copilot-instructions.md`, Cursor `.cursor/rules`. Obsah je v podstate ten istý a väčšina nástrojov dnes vie čítať viacero z nich.

**Čo tam patrí** — veci, ktoré model nemá odkiaľ vedieť:

```markdown
# Projekt XY

## Príkazy
- testy:        `make test` (NIE `pytest` priamo — potrebuje docker compose)
- lint+format:  `make fmt` — pusti pred každým commitom
- lokálny beh:  `make dev`, beží na :8080

## Konvencie
- Migrácie DB sa píšu ručne do `db/migrations/`, ORM ich negeneruje.
- Chyby domény: vlastné výnimky z `app/errors.py`, nikdy holý `Exception`.
- Do `app/legacy/` sa nesiaha — prepisuje sa to inde, zmeny by sa stratili.

## Architektúra — čo nie je vidieť z kódu
- `sync_service` beží ako cron v inom repozitári; naše API mu len pripravuje dáta.
- Tabuľka `orders_v2` je tá živá, `orders` je zvyšok po migrácii (zatiaľ nemažeme).
```

**Čo tam nepatrí:** vysvetľovanie Pythonu, „píš čistý kód", „buď dôsledný", prepis dokumentácie knižnice. To všetko model vie a v súbore to len zaberá miesto v kontexte každého sedenia. Platí presne pravidlo zo [sekcie 3.1 predchádzajúceho dokumentu](01-ako-pouzivat-llm.md#31-kontext-ktorý-model-nemá-odkiaľ-vedieť): **čo viete len vy, to napíšte; čo vie model sám, vynechajte.**

Praktická poučka z prevádzky: súbor rastie sám od seba a po pol roku má tristo riadkov pravidiel, z ktorých polovicu už nikto nevie zdôvodniť. Oplatí sa ho občas prejsť a vyhádzať všetko, čo nerieši konkrétny, reálne pozorovaný problém.

### 3.2 Skills — zabalený postup, ktorý sa načíta, až keď treba

Súbor s pravidlami má strop: všetko v ňom sa načítava vždy. **Skill** je priečinok s inštrukciami (`SKILL.md`), prípadne aj pomocnými skriptami a šablónami, ktorý má krátky popis „na čo som" — a agent si jeho **plný obsah načíta, až keď naň príde rad**.

```text
.claude/skills/
└── nasadenie-produkcia/
    ├── SKILL.md            ← popis + postup (načíta sa pri zmienke o nasadení)
    ├── kontrola.sh         ← skript, ktorý agent môže spustiť
    └── sablona-changelog.md
```

Je to **postupné odkrývanie** (*progressive disclosure*) — tá istá myšlienka ako v RAG: do kontextu púšťaj len to, čo je práve potrebné. Typické skills: postup nasadenia, firemný checklist pri code review, ako sa v tomto projekte píšu migrácie, ako sa generuje report pre zákazníka. Formát je otvorený, takže sa dá použiť aj mimo jedného nástroja.

Rozhodovacie pravidlo: **čo platí vždy → súbor s pravidlami. Čo platí pri jednom type úlohy → skill.**

### 3.3 MCP — pripojenie na svet mimo repozitára

[MCP zo sekcie 3 predchádzajúceho dokumentu](02-agenti-a-nastroje.md#3-mcp--štandard-na-pripájanie-nástrojov) je presne tá vec, ktorá z agenta nad kódom spraví agenta nad vašou firmou:

```text
                  ┌── Jira / Linear     (o čom je ten ticket vlastne?)
  agent ── MCP ───┼── Postgres          (aká je reálna schéma v staging?)
                  ├── Sentry            (aký stacktrace to hádže v produkcii?)
                  └── prehliadač        (naozaj sa to v UI zobrazí?)
```

Bez toho agent pracuje len s tým, čo je v repozitári. S tým vie opraviť chybu tak, že si najprv pozrie skutočné hlásenie zo Sentry a overí schému v databáze.

**Praktická brzda:** každý pripojený MCP server pridá definície svojich nástrojov do každej požiadavky — teda tokeny a viac možností, medzi ktorými sa model môže pomýliť. Pripájajte to, čo naozaj používate, nie všetko, čo existuje.

### 3.4 Hooks a CI — čo sa nemá prosiť, ale vynútiť

Inštrukcia v prompte je **prosba**; model ju splní skoro vždy, a to „skoro" je ten problém. Čo sa dá skontrolovať strojom, nech kontroluje stroj:

| Chcem | Zlé riešenie | Dobré riešenie |
|---|---|---|
| Kód je naformátovaný | `„vždy spusti formátovač"` v pravidlách | **hook** po zápise súboru spustí formátovač |
| Nemení sa `app/legacy/` | veta v `CLAUDE.md` | hook / oprávnenia zakážu zápis do priečinka |
| Testy prechádzajú | „nezabudni pustiť testy" | CI, ktorá PR neprepustí |
| Tajomstvá sa nedostanú do commitu | nič | pre-commit hook so scanerom |

Hook je obyčajný príkaz, ktorý spustí **nástroj, nie model** — pri určitej udalosti (pred volaním nástroja, po zápise súboru, na konci sedenia). Je to deterministické a nedá sa to prehovoriť. Rovnaké pravidlo ako pri bezpečnosti agentov: **obrana musí byť mimo modelu.**

---

## 4. Ako s tým reálne pracovať

Nástroj sám o sebe nespraví dobrý výsledok. Toto je postup, ktorý sa osvedčuje naprieč nástrojmi.

### 4.1 Uzavrite slučku spätnej väzby

Najväčší rozdiel medzi „vygeneroval mi nezmysel" a „za dvadsať minút to fungovalo" nie je prompt, ale **či agent vedel, či to funguje**. Dajte mu spôsob, ako si to overiť: príkaz na testy, reprodukciu chyby, skript, ktorý to spustí.

```text
❌ „Oprav mi ten bug s duplicitnými faktúrami."

✅ „V `tests/test_invoices.py::test_no_duplicates` je padajúci test, ktorý ten bug
    reprodukuje. Spusti `make test`, nájdi príčinu a oprav ju.
    Hotové je, keď prejde celá sada, nie len ten jeden test."
```

Čo sa nedá overiť strojom, overte sami — a počítajte s tým, že tam bude agent slabší.

### 4.2 Najprv sa zorientovať, potom plánovať, až potom písať

Pri čomkoľvek väčšom než jednosúborová zmena sa oplatí rozdeliť to na fázy a **plán si prečítať skôr, než sa začne písať**. Väčšina nástrojov na to má režim, v ktorom agent len číta a navrhuje.

```text
1. prieskum  „Nájdi, kde sa spracúva import objednávok, a vysvetli mi tok dát.
              Nič zatiaľ nemeň."
2. plán      „Navrhni, ako pridať podporu pre čiastočné dodávky. Daj dve varianty
              a povedz, čo ktorá rozbije."
3. práca     „Urob variantu B. Po každom kroku pusti testy."
4. kontrola  git diff, review, testy — vaše oči
```

Chyba v pláne stojí minútu, tá istá chyba v desiatich súboroch stojí hodinu.

### 4.3 Malé kroky a git ako záchranná sieť

Commitujte často a pred väčšou zmenou majte čistý pracovný strom. Agent, ktorý sa pokazí, sa potom vráti jedným príkazom. Úloha na dva dni práce zadaná jedným promptom skončí zle takmer vždy; tá istá úloha po krokoch, každý overený, skončí dobre prekvapivo často.

### 4.4 Čítajte diff, nie zhrnutie

Toto je nová a nepríjemná zručnosť: **kontrolovať kód, ktorý ste nepísali, a ktorý vyzerá dobre.** Zhrnutie od agenta („pridal som validáciu a testy") je jeho vlastný text, nie výpis toho, čo spravil. Pravda je v `git diff`.

Na čo sa pozerať prednostne:

- **rozsah** — zmenil viac súborov, než bolo treba? Neupratal „pri tom" niečo, o čo nikto nežiadal?
- **testy** — testujú niečo, alebo len prejdú? (`assert True`, zamockované presne to, čo sa malo overiť.)
- **závislosti** — pribudla knižnica? Existuje vôbec? Aká má licenciu?
- **tiché zmeny správania** — zmenená predvolená hodnota, potichu odchytená výnimka, upravená migrácia.

### 4.5 Viac agentov naraz — áno, ale na čítanie

To isté pravidlo ako v [sekcii 5.2 predchádzajúceho dokumentu](02-agenti-a-nastroje.md#52-kedy-sa-oplatí-viac-agentov): **paralelné čítanie funguje, paralelný zápis do tej istej kódovej bázy takmer nikdy.** Tri agenty, ktorí súčasne skúmajú tri časti systému a vrátia zhrnutia, sú výborný nápad. Traja agenti píšuci do toho istého modulu si navzájom rozbijú predpoklady. Ak už paralelný zápis, tak do oddelených vetiev alebo pracovných stromov (`git worktree`) a s poctivým zlučovaním.

### 4.6 Kontext je vyčerpateľný zdroj

Dlhé sedenie zaplní kontextové okno výpismi z testov a obsahom súborov, model stratí prehľad a začne opakovať chyby. Praktické opatrenia: **začínať nové sedenie pre novú úlohu**, veľké výpisy nechať zhrnúť, a dlhodobé poznatky presunúť do súboru s pravidlami — nie ich znovu a znovu vysvetľovať v chate.

---

## 5. Čo je trend a čo sa už tak nepoužíva

Toto je tabuľka, kvôli ktorej sa celý dokument oplatí prečítať. Vľavo postupy, ktoré boli pred pár rokmi štandard a dnes sú prežitok; vpravo to, čo ich nahradilo.

| Ustupuje | Nastupuje |
|---|---|
| **Prompt engineering ako hľadanie správnych zaklínadiel** („rozmýšľaj krok po kroku", „si expert svetovej triedy") | **Context engineering** — čo presne má model v okne: súbor s pravidlami, skills, správne nástroje. Formulácia otázky je dnes menšia časť výsledku než obsah kontextu ([podrobne](01-ako-pouzivat-llm.md#4-čo-sa-už-nepoužíva-a-čo-dnes-škodí)) |
| **Copy-paste do chatového okna** — vybrať súbor, vložiť, skopírovať odpoveď späť | **Agent s prístupom k repozitáru**, ktorý si súbory nájde sám, spustí testy a ukáže diff |
| **Doplňovanie riadkov ako hlavný spôsob použitia** | Doplňovanie zostáva (a je na svoju úlohu ideálne), ale ťažisko sa presunulo na **úlohy veľkosti celého ticketu** |
| **Doladenie (fine-tuning) modelu na vlastnú kódovú bázu** | **Kontext, nástroje a vyhľadávanie v repozitári.** Fine-tuning na fakty nefunguje ([lekcia 7](../04-llm/07-fine-tuning-lora.md)) a kódová báza sa mení rýchlejšie, než sa stihne trénovať |
| **Vektorový index nad celým repozitárom** ako spôsob, ako agentovi ukázať kód | **Agentické hľadanie** — `grep`, `glob`, čítanie súborov, sledovanie importov. Kód má presné identifikátory a štruktúru; doslovné hľadanie ich využije lepšie než podobnosť embeddingov (RAG zostáva kráľom nad **dokumentáciou a prózou**, viď [lekcia 6](../04-llm/06-rag.md)) |
| **Vlastná integrácia pre každý nástroj a každú aplikáciu zvlášť** | **MCP** — jeden protokol, integráciu napíšete raz |
| **Jeden obrovský systémový prompt so všetkým** | **Vrstvenie a postupné odkrývanie** — pravidlá projektu + skills, ktoré sa načítajú, až keď treba |
| **Pravidlá v prompte na veci, ktoré sa dajú skontrolovať strojom** | **Hooks, oprávnenia a CI** — deterministické vynútenie mimo modelu |
| **Synchrónne sedenie ako jediný režim práce** | **Agenti na pozadí a v cloude** — úloha sa zadá, vráti sa pull request; človek vstupuje až pri review |
| **Ladenie `temperature` a predvyplnenie odpovede na vynútenie formátu** | **Parametre hĺbky premýšľania a vynútená schéma výstupu**; staré triky na nových modeloch nefungujú alebo rovno vracajú chybu |
| **„AI to napísala, tak to je asi dobré"** | **Review a testy ako úzke hrdlo.** Písanie kódu zlacnelo; jeho overenie nie — a tam sa presunula hodnota seniora |

**Čo naopak nezostarlo ani trochu:** vedieť čítať cudzí kód, vedieť napísať test, ktorý niečo skutočne overuje, rozumieť architektúre systému a vedieť povedať, prečo je návrh zlý. Agent tieto veci zrýchli, ale nenahradí — a kto ich nemá, ten pri kontrole jeho výstupu nemá čo kontrolovať.

### Poznámka k „vibe codingu"

Pojmom *vibe coding* sa označuje práca, pri ktorej sa výsledný kód ani nečíta — zadá sa, čo má vzniknúť, a hodnotí sa len to, či to navonok funguje. Je to legitímny režim pre **prototyp, jednorazový skript alebo skúšku nápadu**, kde je cena chyby nulová a kód sa aj tak zahodí. Pre čokoľvek, čo má bežať v produkcii, spracúvať cudzie dáta alebo to bude niekto o rok udržiavať, je to spôsob, ako si vyrobiť systém, ktorému nikto nerozumie. Rozdiel nie je v nástroji, ale v tom, či niekto zodpovedá za výsledok.

---

## 6. Bezpečnosť špecificky pri práci s kódom

Všetko zo [sekcie 6 predchádzajúceho dokumentu](02-agenti-a-nastroje.md#6-bezpečnosť-agentov) platí — agent nad repozitárom má navyše pár vlastných rizík:

- **Prompt injection cez obsah repozitára.** Agent číta issue, komentár v PR, `README` závislosti alebo webovú stránku z dokumentácie. Ktokoľvek, kto vie umiestniť text tam, kam sa agent pozrie, mu vie adresovať inštrukcie. Pri verejnom repozitári to znamená ktokoľvek.
- **Tajomstvá v kontexte.** `.env`, konfigurácia, produkčné pripojenie k databáze. Čo agent prečíta, môže skončiť vo výstupe, v logu poskytovateľa alebo v commite. Tajomstvá patria mimo dosah — a agent by mal bežať v prostredí, ktoré ich nevidí.
- **Vymyslené balíky (*slopsquatting*).** Model občas navrhne knižnicu, ktorá neexistuje. Útočníci tieto vymyslené názvy registrujú v balíčkovacích registroch a čakajú. **Každú novú závislosť overte** — či existuje, kto ju vydáva a odkedy.
- **Nezvratné operácie.** `git push --force`, mazanie vetiev, migrácia produkčnej databázy, nasadenie. Čítanie môže bežať automaticky; zápis do sveta mimo pracovnej kópie nech potvrdzuje človek.
- **Licencie a únik kódu.** Čo sa posiela poskytovateľovi modelu a za akých podmienok, je vec firemnej politiky a zmluvy — nie vec vkusu ([lekcia 5](../04-llm/04-llm-modely.md)).

---

## 7. Kde to zlyháva

Poctivý zoznam, lebo marketingové video ho neukáže:

- **Veľký, neusporiadaný, starý systém.** Kde je logika rozsypaná v šiestich vrstvách a chýbajú testy, agent nemá o čo oprieť ani kontext, ani spätnú väzbu.
- **Úlohy bez strojovej kontroly** — návrh API, výkonnostná optimalizácia, dizajnové rozhodnutia. Model dá pravdepodobný návrh, nie overený.
- **Kód, ktorý vyzerá správne.** Najdrahšia trieda chýb — kompiluje sa, testy prejdú, a sémanticky je to zle. Odhaliť to je práca človeka.
- **Testy, ktoré netestujú.** Ak sa agentovi zadá „doplň testy", vyrobí ich; či niečo overujú, je iná otázka.
- **Posun pri dlhom sedení.** Po hodine v jednom kontexte začne agent opakovať chyby a odbiehať od zadania.
- **Presun úzkeho hrdla.** Keď sa písanie kódu desaťnásobne zrýchli a review nie, hrdlom je review. Tímy, ktoré si to nepriznajú, merajú produktivitu počtom riadkov a platia to v údržbe.
- **Atrofia zručností u začiatočníkov.** Kto sa naučil zadávať a nie riešiť, nevie posúdiť výstup — a presne to je práca, ktorá zostala ľuďom. Preto: pri učení sa nechajte agentom **vysvetľovať**, a riešte sami.

---

## Kontrolné otázky

1. Prečo sa AI presadila v programovaní rýchlejšie než v iných oboroch? Ktorá z tých vlastností je najdôležitejšia a prečo?
2. V čom sa líši doplňovanie kódu, chat v editore a agent nad repozitárom? Ktoré z toho sa prestalo používať?
3. Čo patrí do `CLAUDE.md` / `AGENTS.md` a čo tam naopak nemá čo robiť? Uveďte po dvoch príkladoch.
4. Kedy použijete skill a kedy súbor s pravidlami projektu? Čo je *progressive disclosure* a prečo na ňom záleží?
5. Chcete mať istotu, že agent nikdy nezmení priečinok `legacy/`. Prečo je veta v pravidlách zlé riešenie a čo je dobré?
6. Popíšte štyri fázy práce na väčšej zmene s agentom. Prečo sa oplatí čítať plán skôr než kód?
7. Agent oznámi „hotovo, testy prechádzajú". Čo konkrétne skontrolujete, než tomu uveríte?
8. Prečo sa nad kódovou bázou dnes používa skôr agentické hľadanie (`grep`) než vektorový index, hoci pri dokumentácii je to naopak?
9. Vymenujte tri postupy z éry 2022–2023, ktoré sa dnes už nepoužívajú, a pri každom povedzte, čo ich nahradilo a prečo.
10. Čo je slopsquatting a aké jednoduché opatrenie proti nemu funguje?

---

### Súvisiace dokumenty

- [prehlad-predmetu.md](../../prehlad-predmetu.md) — prehľad celého predmetu (8 lekcií)
- [01-ako-pouzivat-llm.md](01-ako-pouzivat-llm.md) — prompting, API a šetrenie tokenov (rovnaké remeslo mimo kódu)
- [02-agenti-a-nastroje.md](02-agenti-a-nastroje.md) — **predchádzajúci dokument**: agentová slučka, MCP, viac agentov, bezpečnosť
- [05-llm-trendy.md](05-llm-trendy.md) — **nasledujúci dokument**: kam sa to celé hýbe
- [06-rag.md](../04-llm/06-rag.md) — prečo RAG nad dokumentáciou áno a nad kódom skôr nie
- [04-llm-modely.md](../04-llm/04-llm-modely.md) — výber modelu a právne mantinely

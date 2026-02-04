# Zadanie projektu: Simulátor šikmého vrhu - Hod kameňom

## Úvod
Vytvorte webovú aplikáciu pre real-time simuláciu šikmého vrhu kameňa (hmotný bod bez odporu vzduchu). Aplikácia musí využívať backend v Pythone pre fyzikálne výpočty balistickej trajektórie, frontend v JavaScripte pre vizualizáciu a WebSocket pre jednosmernú komunikáciu v reálnom čase.

---

## Architektúra komunikácie

### Communication Flow:
```
FÁZA 1: VYTVORENIE KONFIGURÁCIE
1. Používateľ vyplní formulár (hmotnosť, počiatočná rýchlosť, uhol vrhu, krok, metóda)
   ↓
2. Frontend → REST API: POST /api/projectile/configurations
   ↓
3. Backend: Validácia parametrov → Uloženie konfigurácie → Výpočet analytických výsledkov
   ↓
4. Backend → Frontend: JSON response (config_id, analytické výsledky, validácia)
   ↓
5. Frontend: Zobrazenie uloženej konfigurácie v zozname

FÁZA 2: SPRÁVA KONFIGURÁCIÍ
6. Frontend → REST API: GET /api/projectile/configurations
   ↓
7. Backend → Frontend: Zoznam všetkých uložených konfigurácií
   ↓
8. Frontend: Zobrazenie zoznamu s možnosťou výberu

FÁZA 3: SPUSTENIE SIMULÁCIE
9. Používateľ vyberie konfiguráciu zo zoznamu a klikne "Spustiť simuláciu"
   ↓
10. Frontend → REST API: POST /api/projectile/simulations/start
    Body: {"config_id": "conf_xyz789"}
   ↓
11. Backend: Načítanie konfigurácie → Vytvorenie simulácie → Spustenie na pozadí
   ↓
12. Backend → Frontend: JSON response (simulation_id, websocket_url)
   ↓
13. Frontend: Otvorenie WebSocket spojenia (ws://localhost:8000/ws/{simulation_id})
   ↓
14. Backend → Frontend: Prvá správa (type: "setup")
   ↓
15. Backend → Frontend: Pravidelné správy (type: "data") každých 50ms
   ↓
16. Backend → Frontend: Posledná správa (type: "completed") keď kameň dopadne
   ↓
17. Frontend: Zatvorenie WebSocket spojenia
```

**Kľúčové charakteristiky:**
- **Oddelenie konfigurácie a spustenia** - konfigurácie sa vytvárajú a ukladajú samostatne
- Frontend najprv vytvorí konfiguráciu, potom ju môže spustiť
- Rovnakú konfiguráciu možno spustiť viackrát
- REST API slúži na správu konfigurácií a riadenie simulácií
- WebSocket je **jednosmerný** (server → klient) - slúži iba na prenos simulačných dát
- Backend riadi celý beh simulácie nezávisle po spustení cez REST API
- Frontend iba vizualizuje prijímané dáta, nemení beh simulácie cez WebSocket
- Simulácia sa končí automaticky keď kameň dopadne na zem (y ≤ 0)

---

## 1. Backend (Python)

### 1.1 Fyzikálny model
Implementujte simulátor šikmého vrhu hmotného bodu bez odporu vzduchu podľa rovníc pohybu:
```
x(t) = v₀·cos(α)·t
y(t) = v₀·sin(α)·t - (1/2)·g·t²

vx(t) = v₀·cos(α)
vy(t) = v₀·sin(α) - g·t

ax = 0
ay = -g
```
kde:
- `v₀` = počiatočná rýchlosť [m/s]
- `α` = uhol vrhu od horizontály [stupne]
- `g` = gravitačné zrýchlenie (9.81 m/s²)
- `x, y` = horizontálna a vertikálna poloha [m]
- `vx, vy` = zložky rýchlosti [m/s]
- `m` = hmotnosť kameňa [kg] (pre úplnosť, aj keď v tomto modeli neovplyvňuje trajektóriu)

**Analytické vzorce (pre porovnanie s numerickou simuláciou):**
```
Dolet: D = (v₀²·sin(2α))/g
Maximálna výška: H = (v₀²·sin²(α))/(2g)
Čas letu: T = (2·v₀·sin(α))/g
Optimálny uhol pre maximálny dolet: 45°
```

**Požiadavky:**
- Implementujte minimálne **2 rôzne numerické metódy**:
  1. **Eulerova metóda** - jednoduchá, 1. rád presnosti
  2. **Runge-Kutta 4. rádu** - presnejšia, 4. rád presnosti
- Časový krok simulácie: konfigurovateľný používateľom (odporúčané: 0.001 - 0.1s)
- Validácia vstupných parametrov:
  - `m > 0` (hmotnosť musí byť kladná, minimum 0.01 kg)
  - `0 < v₀ ≤ 100` (počiatočná rýchlosť v m/s)
  - `0 < α < 90` (uhol vrhu v stupňoch, vylúčený vodorovný a zvislý hod)
  - `0.001 ≤ time_step ≤ 0.1` (stabilita numeriky)
- Výpočet charakteristík vrhu:
  - **Teoretický dolet** (analytický výpočet)
  - **Teoretická maximálna výška** (analytický výpočet)
  - **Teoretický čas letu** (analytický výpočet)
  - **Skutočné hodnoty z numerickej simulácie** (pre porovnanie presnosti metód)
- Ukončovacie podmienky:
  - y ≤ 0 (kameň dopadol na zem)
  - t > 120s (maximálny čas simulácie - bezpečnostná podmienka)
- Asynchrónny beh simulácie (background task po spustení cez REST API)

### 1.2 REST API - Správa konfigurácií

#### `POST /api/projectile/configurations`
Vytvorí a uloží novú konfiguráciu simulácie. **Nespúšťa simuláciu**, len validuje a ukladá parametre.

**Request:**
```json
{
  "name": "Môj hod 45°",
  "description": "Optimálny uhol pre maximálny dolet",
  "mass": 0.5,
  "initial_velocity": 20.0,
  "launch_angle": 45.0,
  "time_step": 0.01,
  "numerical_method": "runge_kutta_4"
}
```

**Podporované numerické metódy:**
- `"euler"` - Eulerova metóda (1. rád presnosti)
- `"runge_kutta_2"` - Runge-Kutta 2. rádu
- `"runge_kutta_4"` - Runge-Kutta 4. rádu (predvolená)

**Odpoveď:**
```json
{
  "config_id": "conf_a1b2c3",
  "name": "Môj hod 45°",
  "description": "Optimálny uhol pre maximálny dolet",
  "parameters": {
    "mass": 0.5,
    "initial_velocity": 20.0,
    "launch_angle": 45.0,
    "time_step": 0.01,
    "numerical_method": "runge_kutta_4"
  },
  "analytical_results": {
    "max_range": 40.82,
    "max_height": 10.20,
    "flight_time": 2.88,
    "optimal_angle": 45.0
  },
  "launch_parameters": {
    "velocity": 20.0,
    "angle": 45.0,
    "vx": 14.14,
    "vy": 14.14
  },
  "created_at": "2025-11-08T10:30:00Z"
}
```

#### `GET /api/projectile/configurations`
Vráti zoznam všetkých uložených konfigurácií.

**Odpoveď:**
```json
{
  "configurations": [
    {
      "config_id": "conf_a1b2c3",
      "name": "Môj hod 45°",
      "description": "Optimálny uhol pre maximálny dolet",
      "parameters": {
        "mass": 0.5,
        "initial_velocity": 20.0,
        "launch_angle": 45.0,
        "time_step": 0.01,
        "numerical_method": "runge_kutta_4"
      },
      "analytical_results": {
        "max_range": 40.82,
        "max_height": 10.20,
        "flight_time": 2.88
      },
      "created_at": "2025-11-08T10:30:00Z",
      "times_simulated": 5
    },
    {
      "config_id": "conf_d4e5f6",
      "name": "Vysoký hod 60°",
      "parameters": {...},
      "analytical_results": {...},
      "created_at": "2025-11-08T11:00:00Z",
      "times_simulated": 2
    }
  ]
}
```

#### `GET /api/projectile/configurations/{config_id}`
Vráti detail konkrétnej konfigurácie.

#### `PUT /api/projectile/configurations/{config_id}`
Aktualizuje existujúcu konfiguráciu.

#### `DELETE /api/projectile/configurations/{config_id}`
Zmaže konfiguráciu (iba ak nie je práve spustená žiadna simulácia s touto konfiguráciou).

### 1.3 REST API - Správa simulácií

#### `POST /api/projectile/simulations/start`
Spustí simuláciu na základe existujúcej konfigurácie.

**Request:**
```json
{
  "config_id": "conf_a1b2c3"
}
```

**Odpoveď:**
```json
{
  "simulation_id": "sim_xyz789",
  "config_id": "conf_a1b2c3",
  "websocket_url": "ws://localhost:8000/ws/sim_xyz789",
  "status": "running",
  "started_at": "2025-11-08T10:35:00Z"
}
```

#### `GET /api/projectile/simulations`
Vráti zoznam všetkých simulácií (aktívnych aj dokončených).

**Query parametre:**
- `status` - filter podľa stavu: "running", "completed", "stopped"
- `config_id` - filter podľa konfigurácie
- `limit` - maximálny počet výsledkov (default 50)

**Odpoveď:**
```json
{
  "simulations": [
    {
      "simulation_id": "sim_xyz789",
      "config_id": "conf_a1b2c3",
      "config_name": "Môj hod 45°",
      "status": "running",
      "started_at": "2025-11-08T10:35:00Z",
      "completed_at": null,
      "actual_results": null
    },
    {
      "simulation_id": "sim_abc123",
      "config_id": "conf_a1b2c3",
      "config_name": "Môj hod 45°",
      "status": "completed",
      "started_at": "2025-11-08T10:30:00Z",
      "completed_at": "2025-11-08T10:30:03Z",
      "actual_results": {
        "actual_range": 40.81,
        "actual_max_height": 10.20,
        "numerical_error": 0.02
      }
    }
  ]
}
```

#### `GET /api/projectile/simulations/{simulation_id}`
Vráti detail konkrétnej simulácie.

#### `DELETE /api/projectile/simulations/{simulation_id}`
Zastaví bežiacu simuláciu.

**Odpoveď:**
```json
{
  "status": "stopped",
  "simulation_id": "sim_xyz789",
  "time_elapsed": 1.25
}
```

#### Ďalšie endpointy:
- `POST /api/projectile/configurations/validate` - validácia parametrov pred vytvorením konfigurácie
- `GET /api/projectile/configurations/{config_id}/history` - história simulácií pre danú konfiguráciu
- `GET /api/info` - vráti server info (verzia, podporované rozsahy parametrov, dostupné metódy)

### 1.4 WebSocket server (jednosmerný)
- Framework: `websockets` alebo `socket.io` pre Python
- Endpoint: `ws://localhost:8000/ws/{simulation_id}`
- **Iba jednostranná komunikácia:** server → klient
- Klient sa pripojí pomocou `simulation_id` z REST API odpovede

**Prvá správa (setup):**
```json
{
  "type": "setup",
  "simulation_id": "sim_xyz789",
  "config_id": "conf_a1b2c3",
  "parameters": {
    "mass": 0.5,
    "initial_velocity": 20.0,
    "launch_angle": 45.0,
    "time_step": 0.01,
    "method": "runge_kutta_4"
  },
  "analytical_results": {
    "max_range": 40.82,
    "max_height": 10.20,
    "flight_time": 2.88
  }
}
```

**Následné správy (dáta simulácie)** - každých 50ms:
```json
{
  "type": "data",
  "time": 1.25,
  "position": {
    "x": 17.68,
    "y": 9.95
  },
  "velocity": {
    "vx": 14.14,
    "vy": 1.89
  },
  "speed": 14.27,
  "height": 9.95
}
```

**Posledná správa (kameň dopadol):**
```json
{
  "type": "completed",
  "total_time": 2.88,
  "final_position": {
    "x": 40.81,
    "y": 0.0
  },
  "actual_range": 40.81,
  "actual_max_height": 10.20,
  "numerical_error": 0.02
}
```

---

## 2. Frontend (JavaScript/HTML/CSS)

### 2.1 Používateľské rozhranie

**Sekcia 1: Správa konfigurácií**
- Formulár na vytvorenie novej konfigurácie:
  - Názov konfigurácie (string)
  - Popis (voliteľné)
  - Hmotnosť kameňa (kg) - default 0.5
  - Počiatočná rýchlosť (m/s) - default 20
  - Uhol vrhu (stupne) - default 45
  - **Časový krok simulácie** (s) - default 0.01
  - **Numerická metóda** - dropdown (Euler, RK2, RK4)
- Tlačidlo "Uložiť konfiguráciu"
- Zobrazenie analytických výsledkov po uložení
- Validácia na strane klienta (rozsahy hodnôt, numerické vstupy)

**Sekcia 2: Zoznam konfigurácií**
- Tabuľka/karty zobrazujúce všetky uložené konfigurácie
- Pre každú konfiguráciu:
  - Názov a popis
  - Základné parametre
  - Analytické výsledky
  - Počet spustení
  - Tlačidlá: "Spustiť simuláciu", "Upraviť", "Zmazať"

**Sekcia 3: Vizualizácia simulácie**
- Canvas/SVG zobrazenie trajektórie
- Zobrazenie aktuálnych hodnôt:
  - Čas (s)
  - Pozícia (x, y) v metroch
  - Rýchlosť (vx, vy, celková) v m/s
  - Výška nad zemou
- Graf výšky v čase
- Tlačidlá: Stop, Reset
- Porovnanie numerických a analytických výsledkov

**Sekcia 4: História simulácií**
- Zoznam posledných spustených simulácií
- Filter podľa konfigurácie
- Možnosť replay

### 2.2 Workflow používateľa

**Vytvorenie konfigurácie:**
1. Používateľ vyplní formulár s parametrami
2. Klikne "Uložiť konfiguráciu"
3. Frontend: POST `/api/projectile/configurations`
4. Zobrazenie analytických výsledkov
5. Konfigurácia sa objaví v zozname konfigurácií

**Spustenie simulácie:**
1. Používateľ vyberie konfiguráciu zo zoznamu
2. Klikne "Spustiť simuláciu"
3. Frontend: POST `/api/projectile/simulations/start` s `config_id`
4. Získanie `simulation_id` a `websocket_url`
5. Otvorenie WebSocket spojenia
6. Real-time vizualizácia

**Správa konfigurácií:**
1. Načítanie všetkých konfigurácií: GET `/api/projectile/configurations`
2. Zobrazenie v tabuľke/kartách
3. Možnosť úpravy, mazania, spúšťania

### 2.3 Príklad implementácie

```javascript
// Vytvorenie novej konfigurácie
async function saveConfiguration() {
    const configData = {
        name: document.getElementById('configName').value,
        description: document.getElementById('configDesc').value,
        mass: parseFloat(document.getElementById('mass').value),
        initial_velocity: parseFloat(document.getElementById('velocity').value),
        launch_angle: parseFloat(document.getElementById('angle').value),
        time_step: parseFloat(document.getElementById('timeStep').value),
        numerical_method: document.getElementById('method').value
    };
    
    const response = await fetch('/api/projectile/configurations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configData)
    });
    
    const config = await response.json();
    displayAnalyticalResults(config.analytical_results);
    addConfigToList(config);
}

// Načítanie všetkých konfigurácií
async function loadConfigurations() {
    const response = await fetch('/api/projectile/configurations');
    const data = await response.json();
    
    displayConfigurations(data.configurations);
}

// Spustenie simulácie z vybranej konfigurácie
async function startSimulation(configId) {
    const response = await fetch('/api/projectile/simulations/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_id: configId })
    });
    
    const simulation = await response.json();
    connectWebSocket(simulation.websocket_url);
}

// WebSocket pripojenie
function connectWebSocket(wsUrl) {
    const ws = new WebSocket(wsUrl);
    
    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        switch(message.type) {
            case 'setup':
                initializeVisualization(message);
                break;
            case 'data':
                updateVisualization(message);
                updateChart(message.time, message.height);
                break;
            case 'completed':
                displayFinalResults(message);
                ws.close();
                break;
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}
```

### 2.4 Vizualizácia
Vytvorte animáciu pomocou **Canvas API** alebo **SVG**:
- 2D pohľad zboku na trajektóriu
- Real-time pohyb kameňa
- Stopa trajektórie za kameňom
- Zobrazenie analytickej trajektórie (transparentná čiara pre porovnanie)
- Graf výšky v čase (Chart.js/D3.js)
- Plynulá animácia (60 FPS)
- Zobrazenie aktuálneho času, pozície, rýchlosti

---

## 3. Minimálne požiadavky na funkcionalitu

### Backend musí:
1. Správne implementovať fyzikálny model šikmého vrhu s minimálne 2 numerickými metódami
2. **Poskytnúť API pre správu konfigurácií** (CRUD operácie)
3. **Poskytnúť API pre správu simulácií** (spustenie, zastavenie, história)
4. Ukladať konfigurácie a históriu simulácií do databázy (SQLite/PostgreSQL)
5. Generovať unikátne `config_id` a `simulation_id`
6. Bežať simuláciu asynchrónne (background task) po spustení cez REST API
7. Posielať dáta cez WebSocket v správnom formáte (setup → data → completed)
8. Podporovať viacero súčasných simulácií
9. Validovať všetky vstupy a odmietnuť neplatné hodnoty
10. Počítať analytické výsledky pri vytvorení konfigurácie

### Frontend musí:
1. Poskytovať formulár pre vytvorenie novej konfigurácie
2. **Zobrazovať zoznam všetkých uložených konfigurácií**
3. Umožňovať výber konfigurácie a spustenie simulácie
4. **Volať správne API endpointy** (konfigurácie vs simulácie)
5. Pripojiť sa na WebSocket pomocou `simulation_id`
6. Spracovať správy typu "setup", "data" a "completed"
7. Vizualizovať trajektóriu kameňa v reálnom čase
8. Vykresliť graf výšky v čase
9. Zobrazovať porovnanie numerických a analytických výsledkov
10. Správne spracovať výpadky spojenia
11. Umožniť manuálne zastavenie simulácie
12. Umožniť úpravu a mazanie konfigurácií
13. Byť responzívny (mobile-friendly)

### Security musí:
1. Zabrániť základným typom útokov (XSS, SQL injection, CSRF)
2. Implementovať rate limiting na REST API
3. Validovať `config_id` a `simulation_id` pred použitím
4. Používať bezpečné protokoly (HTTPS/WSS) v produkcii
5. Správne uchovávať citlivé údaje v `.env` súbore

### DevOps musí:
1. Mať čistú Git históriu s logickými commitmi (min 15 zmysluplných commitov)
2. Obsahovať kompletnú dokumentáciu (README, ARCHITECTURE, API dokumentácia)
3. Mať funkčný CI/CD pipeline (GitHub Actions)
4. Byť jednoducho nasaditeľný pomocou Docker (docker-compose)
5. Obsahovať príklady použitia API (curl/Postman príklady)

---

## 4. Príklady použitia API

### 1. Vytvorenie konfigurácie
```bash
curl -X POST http://localhost:8000/api/projectile/configurations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Optimálny hod",
    "description": "45° uhol pre maximálny dolet",
    "mass": 0.5,
    "initial_velocity": 25.0,
    "launch_angle": 45.0,
    "time_step": 0.01,
    "numerical_method": "runge_kutta_4"
  }'
```

**Odpoveď:**
```json
{
  "config_id": "conf_a1b2c3",
  "name": "Optimálny hod",
  "parameters": {...},
  "analytical_results": {
    "max_range": 63.75,
    "max_height": 15.94,
    "flight_time": 3.60
  },
  "created_at": "2025-11-08T10:30:00Z"
}
```

### 2. Získanie všetkých konfigurácií
```bash
curl -X GET http://localhost:8000/api/projectile/configurations
```

### 3. Spustenie simulácie
```bash
curl -X POST http://localhost:8000/api/projectile/simulations/start \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "conf_a1b2c3"
  }'
```

**Odpoveď:**
```json
{
  "simulation_id": "sim_xyz789",
  "config_id": "conf_a1b2c3",
  "websocket_url": "ws://localhost:8000/ws/sim_xyz789",
  "status": "running",
  "started_at": "2025-11-08T10:35:00Z"
}
```

### 4. Získanie všetkých simulácií
```bash
curl -X GET "http://localhost:8000/api/projectile/simulations?status=completed&limit=10"
```

### 5. Zastavenie simulácie
```bash
curl -X DELETE http://localhost:8000/api/projectile/simulations/sim_xyz789
```

---

## 5. Dátový model

### Tabuľka: configurations
```sql
CREATE TABLE configurations (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    mass FLOAT NOT NULL,
    initial_velocity FLOAT NOT NULL,
    launch_angle FLOAT NOT NULL,
    time_step FLOAT NOT NULL,
    numerical_method VARCHAR(50) NOT NULL,
    analytical_max_range FLOAT,
    analytical_max_height FLOAT,
    analytical_flight_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Tabuľka: simulations
```sql
CREATE TABLE simulations (
    id VARCHAR(50) PRIMARY KEY,
    config_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL, -- 'running', 'completed', 'stopped'
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    actual_range FLOAT,
    actual_max_height FLOAT,
    actual_flight_time FLOAT,
    numerical_error FLOAT,
    FOREIGN KEY (config_id) REFERENCES configurations(id)
);
```

---

## Tipy a odporúčania

### Pre backend:
- Použite ORM (SQLAlchemy pre Python) pre prácu s databázou
- Implementujte transakcional handling pri vytváraní konfigurácií
- Pri mazaní konfigurácie skontrolujte, či nemá aktívne simulácie
- Cachujte analytické výsledky v konfigurácii
- Pre UUID použite `import uuid`

### Pre frontend:
- Použite state management (React Context/Redux alebo jednoduchý objekt)
- Implementujte debouncing pri validácii formulára
- Zobrazujte loading states pri API calloch
- Použite optimistic UI updates pre lepší UX
- Implementujte pagination pre veľké zoznamy konfigurácií

### Pre DevOps:
- Databázové migrácie (Alembic pre SQLAlchemy)
- Backup stratégia pre databázu
- Monitoring aktívnych simulácií
- Health check endpointy

---

**Veľa úspechov pri realizácii projektu! **
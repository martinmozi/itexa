# Zadanie projektu: Simulátor križovatky so semaformi

## Úvod
Vytvorte webovú aplikáciu pre real-time simuláciu križovatky v tvare plus so semaformi a premávkou. Aplikácia musí využívať backend v Pythone pre simuláciu pohybu áut a riadenie semaforov, frontend v JavaScripte pre vizualizáciu a WebSocket pre jednosmernú komunikáciu v reálnom čase.

---

## Architektúra komunikácie

### Communication Flow:
```
FÁZA 1: VYTVORENIE KONFIGURÁCIE SEMAFOROV
1. Používateľ nastaví časovanie semaforov v setup formulári/editore
   ↓
2. Frontend → REST API: POST /api/intersection/configurations
   ↓
3. Backend: Validácia parametrov → Kontrola konfliktov → Uloženie konfigurácie
   ↓
4. Backend → Frontend: JSON response (config_id, validácia konfliktov, warnings)
   ↓
5. Frontend: Zobrazenie uloženej konfigurácie v zozname

FÁZA 2: SPRÁVA KONFIGURÁCIÍ
6. Frontend → REST API: GET /api/intersection/configurations
   ↓
7. Backend → Frontend: Zoznam všetkých uložených konfigurácií (vrátane prednastavených)
   ↓
8. Frontend: Zobrazenie zoznamu s možnosťou výberu

FÁZA 3: SPUSTENIE SIMULÁCIE
9. Používateľ vyberie konfiguráciu a nastaví intenzitu premávky
   ↓
10. Frontend → REST API: POST /api/intersection/simulations/start
    Body: {"config_id": "conf_abc123", "traffic_intensity": {...}, "duration": 300}
   ↓
11. Backend: Načítanie konfigurácie → Vytvorenie simulácie → Spustenie na pozadí
   ↓
12. Backend → Frontend: JSON response (simulation_id, websocket_url)
   ↓
13. Frontend: Otvorenie WebSocket spojenia (ws://localhost:8000/ws/{simulation_id})
   ↓
14. Backend → Frontend: Prvá správa (type: "setup")
   ↓
15. Backend → Frontend: Pravidelné správy (type: "state") každých 100ms
   ↓
16. Backend → Frontend: Správa (type: "completed") po ukončení simulácie
   ↓
17. Frontend: Zatvorenie WebSocket spojenia
```

**Kľúčové charakteristiky:**
- **Oddelenie konfigurácie časovania semaforov a spustenia simulácie**
- Konfigurácie časovania sú nezávislé od parametrov premávky
- Rovnakú konfiguráciu možno použiť s rôznymi intenzitami premávky
- REST API slúži na správu konfigurácií a riadenie simulácií
- WebSocket je **jednosmerný** (server → klient) - slúži na prenos stavov semaforov a pozícií áut
- Backend riadi celú simuláciu (generovanie áut, pohyb, riadenie semaforov)
- Frontend iba vizualizuje prijímané dáta
- Simulácia beží zadaný čas alebo donekonečna až do manuálneho zastavenia

---

## 1. Backend (Python)

### 1.1 Model križovatky

**Geometria:**
- Križovatka v tvare plus (+) so 4 smermi: **Sever, Juh, Východ, Západ**
- Každý smer má 3 jazdné pruhy pre:
  - **Rovno** (straight)
  - **Doľava** (left)
  - **Doprava** (right)
- Celkovo **12 semaforov** (4 smery × 3 pohyby)

**Semafory:**
- Iba 2 stavy: **ZELENÁ** (green) a **ČERVENÁ** (red)
- Každý semafor má **šípku** (nie guľu) - riadi špecifický pohyb
- Konfigurovateľné časovanie v rámci cyklu

**Identifikácia semaforov:**
```
N_S  - Sever → Juh (rovno)
N_L  - Sever → Západ (doľava)
N_R  - Sever → Východ (doprava)

S_S  - Juh → Sever (rovno)
S_L  - Juh → Východ (doľava)
S_R  - Juh → Západ (doprava)

E_S  - Východ → Západ (rovno)
E_L  - Východ → Juh (doľava)
E_R  - Východ → Sever (doprava)

W_S  - Západ → Východ (rovno)
W_L  - Západ → Sever (doľava)
W_R  - Západ → Juh (doprava)
```

### 1.2 Pravidlá bezpečnosti

Backend musí kontrolovať **konflikty** medzi semaformi. Nemôžu mať zelenú súčasne:

**Základné konflikty:**
1. **Protiidúce smery rovno:** `N_S` vs `S_S`, `E_S` vs `W_S`
2. **Odbočenie doľava vs protiidúci rovno:**
   - `N_L` (Sever doľava) vs `S_S` (Juh rovno)
   - `S_L` (Juh doľava) vs `N_S` (Sever rovno)
   - `E_L` (Východ doľava) vs `W_S` (Západ rovno)
   - `W_L` (Západ doľava) vs `E_S` (Východ rovno)
3. **Kolízne odbočenia:**
   - `N_L` vs `E_L` (obidva odbočujú do rovnakého priestoru)
   - `N_L` vs `W_R` (križujú sa)
   - A podobne pre všetky kombinácie

**Požiadavky:**
- Backend validuje konfliktné nastavenia pri vytvorení konfigurácie
- Ak používateľ nastaví konfliktné zelené fázy, backend vráti chybu
- Backend generuje warning ak sú intervaly neefektívne (malé využitie)

### 1.3 Generovanie a pohyb áut

**Generovanie áut:**
- Autá prichádzajú na križovatku náhodne alebo podľa Poissonovho procesu
- Konfigurovateľná **intenzita premávky** (autá/minútu) pre každý smer
- Každé auto má náhodne zvolený cieľový pohyb (rovno, doľava, doprava)

**Pohyb áut:**
- Auto prichádza k semaforu
- Ak je červená, auto čaká
- Ak je zelená, auto pokračuje cez križovatku
- Rýchlosť áut: konfigurovateľná (default 10 m/s)
- Čas prechodu križovatkou: cca 3-5 sekúnd
- Auto opúšťa simuláciu po prejdení križovatkou

**Štatistiky:**
- Počet prejdených áut
- Priemerná čakacia doba
- Maximálna dĺžka radu
- Využitie križovatky (% času kedy prejdú autá)

### 1.4 REST API - Správa konfigurácií

#### `POST /api/intersection/configurations`
Vytvorí a uloží novú konfiguráciu časovania semaforov. **Nespúšťa simuláciu**, len validuje a ukladá nastavenia.

**Request:**
```json
{
  "name": "Vyvážená konfigurácia",
  "description": "Rovnaký čas pre Sever-Juh a Východ-Západ",
  "cycle_duration": 120,
  "signal_timings": {
    "N_S": {"start": 0, "duration": 50},
    "N_L": {"start": 50, "duration": 10},
    "N_R": {"start": 0, "duration": 50},
    "S_S": {"start": 0, "duration": 50},
    "S_L": {"start": 0, "duration": 0},
    "S_R": {"start": 0, "duration": 50},
    "E_S": {"start": 60, "duration": 50},
    "E_L": {"start": 110, "duration": 10},
    "E_R": {"start": 60, "duration": 50},
    "W_S": {"start": 60, "duration": 50},
    "W_L": {"start": 60, "duration": 0},
    "W_R": {"start": 60, "duration": 50}
  }
}
```

**Validácia:**
- `cycle_duration > 0` (min 30s, max 300s)
- Pre každý semafor: `0 <= start < cycle_duration`, `0 <= duration <= cycle_duration`
- **Kontrola konfliktov:** žiadne dva konfliktné semafory nemôžu mať zelenú v rovnakom čase

**Odpoveď:**
```json
{
  "config_id": "conf_abc123",
  "name": "Vyvážená konfigurácia",
  "description": "Rovnaký čas pre Sever-Juh a Východ-Západ",
  "cycle_duration": 120,
  "signal_timings": {...},
  "total_phases": 4,
  "conflicts_detected": [],
  "warnings": [
    "Semafor S_L má nulovú dĺžku zelenej fázy",
    "Celkové využitie cyklu je 91.6% - odporúčané je 80-95%"
  ],
  "cycle_utilization": 0.916,
  "created_at": "2025-11-08T10:30:00Z"
}
```

#### `GET /api/intersection/configurations`
Vráti zoznam všetkých uložených konfigurácií.

**Query parametre:**
- `include_presets` - zahrnúť prednastavené konfigurácie (default: true)

**Odpoveď:**
```json
{
  "configurations": [
    {
      "config_id": "conf_abc123",
      "name": "Vyvážená konfigurácia",
      "description": "Rovnaký čas pre Sever-Juh a Východ-Západ",
      "cycle_duration": 120,
      "is_preset": false,
      "times_simulated": 8,
      "created_at": "2025-11-08T10:30:00Z"
    },
    {
      "config_id": "preset_balanced",
      "name": "Prednastavená: Vyvážená",
      "description": "Prednastavená vyvážená konfigurácia",
      "cycle_duration": 120,
      "is_preset": true,
      "times_simulated": 0,
      "created_at": "2025-01-01T00:00:00Z"
    }
  ]
}
```

#### `GET /api/intersection/configurations/{config_id}`
Vráti detail konkrétnej konfigurácie vrátane kompletného časovania semaforov.

#### `PUT /api/intersection/configurations/{config_id}`
Aktualizuje existujúcu konfiguráciu (iba ak nie je preset).

#### `DELETE /api/intersection/configurations/{config_id}`
Zmaže konfiguráciu (iba používateľské, nie presety).

#### `POST /api/intersection/configurations/validate`
Validácia nastavení bez uloženia (pre real-time feedback v editore).

### 1.5 REST API - Správa simulácií

#### `POST /api/intersection/simulations/start`
Spustí simuláciu na základe existujúcej konfigurácie.

**Request:**
```json
{
  "config_id": "conf_abc123",
  "simulation_duration": 300,
  "traffic_intensity": {
    "north": 20,
    "south": 20,
    "east": 15,
    "west": 15
  },
  "vehicle_speed": 10
}
```

**Validácia:**
- `simulation_duration > 0` (max 3600s = 1 hodina)
- `traffic_intensity >= 0` (autá/minútu, max 100)
- `vehicle_speed > 0` (m/s, rozsah 5-20)

**Odpoveď:**
```json
{
  "simulation_id": "sim_xyz789",
  "config_id": "conf_abc123",
  "config_name": "Vyvážená konfigurácia",
  "websocket_url": "ws://localhost:8000/ws/sim_xyz789",
  "status": "running",
  "parameters": {
    "cycle_duration": 120,
    "simulation_duration": 300,
    "traffic_intensity": {...},
    "vehicle_speed": 10
  },
  "expected_throughput": {
    "north": 18,
    "south": 18,
    "east": 14,
    "west": 14
  },
  "started_at": "2025-11-08T10:35:00Z"
}
```

#### `GET /api/intersection/simulations`
Vráti zoznam všetkých simulácií.

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
      "config_id": "conf_abc123",
      "config_name": "Vyvážená konfigurácia",
      "status": "running",
      "started_at": "2025-11-08T10:35:00Z",
      "elapsed_time": 45.2,
      "current_statistics": {
        "total_vehicles_passed": 35,
        "average_wait_time": 12.5
      }
    },
    {
      "simulation_id": "sim_abc456",
      "config_id": "conf_abc123",
      "config_name": "Vyvážená konfigurácia",
      "status": "completed",
      "started_at": "2025-11-08T10:00:00Z",
      "completed_at": "2025-11-08T10:05:00Z",
      "final_statistics": {
        "total_vehicles_generated": 125,
        "total_vehicles_passed": 125,
        "average_wait_time": 14.2,
        "max_wait_time": 58.3,
        "intersection_utilization": 0.75
      }
    }
  ]
}
```

#### `GET /api/intersection/simulations/{simulation_id}`
Vráti detail konkrétnej simulácie.

#### `GET /api/intersection/simulations/{simulation_id}/stats`
Získanie aktuálnych štatistík bežiacej simulácie.

#### `DELETE /api/intersection/simulations/{simulation_id}`
Zastaví bežiacu simuláciu.

**Odpoveď:**
```json
{
  "status": "stopped",
  "simulation_id": "sim_xyz789",
  "elapsed_time": 150.5,
  "final_statistics": {...}
}
```

#### Ďalšie endpointy:
- `GET /api/intersection/configurations/{config_id}/history` - história simulácií pre danú konfiguráciu
- `GET /api/intersection/presets` - vráti prednastavené konfigurácie
- `GET /api/info` - server info

### 1.6 WebSocket server (jednosmerný)
- Framework: `websockets` alebo `socket.io` pre Python
- Endpoint: `ws://localhost:8000/ws/{simulation_id}`
- **Iba jednostranná komunikácia:** server → klient

**Prvá správa (setup):**
```json
{
  "type": "setup",
  "simulation_id": "sim_xyz789",
  "config_id": "conf_abc123",
  "cycle_duration": 120,
  "signal_timings": {...},
  "intersection_layout": {
    "width": 20,
    "height": 20,
    "lane_width": 3
  }
}
```

**Následné správy (state updates)** - každých 100ms:
```json
{
  "type": "state",
  "time": 45.3,
  "cycle_time": 45.3,
  "signals": {
    "N_S": "green",
    "N_L": "red",
    "N_R": "green",
    "S_S": "green",
    ...
  },
  "vehicles": [
    {
      "id": "v_001",
      "from": "north",
      "to": "south",
      "position": {"x": 0, "y": -5},
      "state": "approaching"
    },
    {
      "id": "v_002",
      "from": "east",
      "to": "north",
      "position": {"x": 15, "y": 2},
      "state": "crossing"
    }
  ],
  "queue_lengths": {
    "north": 2,
    "south": 1,
    "east": 0,
    "west": 3
  },
  "statistics": {
    "total_vehicles_generated": 45,
    "total_vehicles_passed": 35,
    "total_vehicles_waiting": 10,
    "average_wait_time": 12.3
  }
}
```

**Posledná správa (completed):**
```json
{
  "type": "completed",
  "total_time": 300.0,
  "final_statistics": {
    "total_vehicles_generated": 125,
    "total_vehicles_passed": 120,
    "average_wait_time": 14.2,
    "max_wait_time": 58.3,
    "average_queue_length": 2.1,
    "max_queue_length": 7,
    "intersection_utilization": 0.75
  }
}
```

---

## 2. Frontend (JavaScript/HTML/CSS)

### 2.1 Používateľské rozhranie

**Sekcia 1: Editor konfigurácií časovania**
- Vizuálny editor pre nastavenie časovania semaforov
- Timeline zobrazujúci celý cyklus
- Možnosť drag & drop pre nastavenie intervalov
- Real-time detekcia konfliktov (zvýraznenie červenou)
- Formulár s parametrami:
  - Názov konfigurácie
  - Popis
  - Dĺžka cyklu (s)
  - Pre každý semafor: start_time, duration
- Tlačidlo "Uložiť konfiguráciu"
- Zobrazenie warnings a konfliktov po uložení

**Sekcia 2: Zoznam konfigurácií**
- Tabuľka/karty zobrazujúce všetky konfigurácie
- Filter: Moje konfigurácie / Prednastavené / Všetky
- Pre každú konfiguráciu:
  - Názov a popis
  - Dĺžka cyklu
  - Počet fáz
  - Počet spustení
  - Tlačidlá: "Spustiť simuláciu", "Upraviť", "Zmazať", "Duplikovať"

**Sekcia 3: Nastavenie simulácie**
- Výber konfigurácie zo zoznamu
- Nastavenie parametrov premávky:
  - Intenzita pre každý smer (autá/minútu)
  - Rýchlosť áut (m/s)
  - Trvanie simulácie (s)
- Tlačidlo "Spustiť simuláciu"

**Sekcia 4: Vizualizácia simulácie**
- Canvas zobrazenie križovatky zhora
- 12 semaforov s farebnými šípkami
- Animácia pohybu áut
- Progress bar cyklu (kde sa práve nachádza v cykle)
- Real-time štatistiky:
  - Čas simulácie
  - Počet áut (generovaných, čakajúcich, prejdených)
  - Priemerná čakacia doba
  - Dĺžky radov
- Grafy:
  - Dĺžky radov v čase
  - Čakacie doby
  - Využitie križovatky
- Tlačidlá: Stop, Reset

**Sekcia 5: História simulácií**
- Zoznam dokončených simulácií
- Filter podľa konfigurácie
- Porovnanie výsledkov

- Responzívny dizajn (mobile-friendly)

### 2.2 Workflow používateľa

**Vytvorenie konfigurácie:**
1. Používateľ otvorí editor časovania
2. Nastaví dĺžku cyklu a časovanie každého semaforu
3. Editor zobrazuje konflikty v reálnom čase
4. Používateľ zadá názov a klikne "Uložiť"
5. Frontend: POST `/api/intersection/configurations`
6. Konfigurácia sa objaví v zozname

**Spustenie simulácie:**
1. Používateľ vyberie konfiguráciu zo zoznamu
2. Nastaví intenzitu premávky a trvanie
3. Klikne "Spustiť simuláciu"
4. Frontend: POST `/api/intersection/simulations/start` s `config_id` a parametrami
5. Získanie `simulation_id` a `websocket_url`
6. Otvorenie WebSocket spojenia
7. Real-time vizualizácia

### 2.3 Príklad implementácie

```javascript
// Uloženie konfigurácie časovania
async function saveConfiguration() {
    const configData = {
        name: document.getElementById('configName').value,
        description: document.getElementById('configDesc').value,
        cycle_duration: parseInt(document.getElementById('cycleDuration').value),
        signal_timings: getSignalTimingsFromEditor()
    };
    
    const response = await fetch('/api/intersection/configurations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configData)
    });
    
    const config = await response.json();
    
    if (config.conflicts_detected.length > 0) {
        displayConflicts(config.conflicts_detected);
    } else {
        addConfigToList(config);
        showSuccess("Konfigurácia uložená");
    }
}

// Načítanie všetkých konfigurácií
async function loadConfigurations() {
    const response = await fetch('/api/intersection/configurations');
    const data = await response.json();
    
    displayConfigurationsList(data.configurations);
}

// Spustenie simulácie z vybranej konfigurácie
async function startSimulation(configId) {
    const simulationParams = {
        config_id: configId,
        simulation_duration: parseInt(document.getElementById('duration').value),
        traffic_intensity: {
            north: parseInt(document.getElementById('intensityNorth').value),
            south: parseInt(document.getElementById('intensitySouth').value),
            east: parseInt(document.getElementById('intensityEast').value),
            west: parseInt(document.getElementById('intensityWest').value)
        },
        vehicle_speed: parseFloat(document.getElementById('vehicleSpeed').value)
    };
    
    const response = await fetch('/api/intersection/simulations/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(simulationParams)
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
                initializeIntersectionView(message);
                break;
            case 'state':
                updateSignals(message.signals);
                updateVehicles(message.vehicles);
                updateStatistics(message.statistics);
                updateCycleProgress(message.cycle_time);
                break;
            case 'completed':
                displayFinalStatistics(message.final_statistics);
                ws.close();
                break;
        }
    };
}

// Real-time validácia konfliktov v editore
async function validateTimings() {
    const timings = getSignalTimingsFromEditor();
    
    const response = await fetch('/api/intersection/configurations/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            cycle_duration: getCurrentCycleDuration(),
            signal_timings: timings
        })
    });
    
    const validation = await response.json();
    highlightConflicts(validation.conflicts_detected);
}
```

---

## 3. Minimálne požiadavky na funkcionalitu

### Backend musí:
1. Implementovať model križovatky s 12 semaformi
2. Detekčný systém konfliktov medzi semaformi
3. **Poskytovať API pre správu konfigurácií časovania** (CRUD)
4. **Poskytovať API pre správu simulácií** (spustenie, zastavenie, štatistiky)
5. Ukladať konfigurácie a históriu simulácií do databázy
6. Generovať autá podľa nastavených intenzít
7. Simulovať pohyb áut s rešpektovaním semaforov
8. Bežať simuláciu asynchrónne na pozadí
9. Posielať state updates cez WebSocket
10. Počítať štatistiky v reálnom čase

### Frontend musí:
1. **Poskytovať vizuálny editor pre konfiguráciu časovania**
2. **Real-time detekciu konfliktov v editore**
3. **Zobrazovať zoznam všetkých konfigurácií**
4. Umožňovať výber konfigurácie a nastavenie parametrov premávky
5. **Volať správne API endpointy** (konfigurácie vs simulácie)
6. Pripojiť sa na WebSocket
7. Vizualizovať križovatku s 12 semaformi
8. Animovať pohyb áut
9. Zobrazovať štatistiky v reálnom čase
10. Vykresliť grafy dĺžok radov
11. Umožniť zastavenie simulácie
12. Umožniť úpravu a mazanie konfigurácií
13. Byť responzívny

### Security musí:
1. Zabrániť základným typom útokov
2. Implementovať rate limiting
3. Validovať všetky IDs
4. Používať HTTPS/WSS v produkcii
5. Správne uchovávať citlivé údaje

### DevOps musí:
1. Čistá Git história (min 15 commitov)
2. Kompletná dokumentácia
3. CI/CD pipeline
4. Docker deployment
5. API príklady

---

## 4. Príklady použitia API

### 1. Vytvorenie konfigurácie
```bash
curl -X POST http://localhost:8000/api/intersection/configurations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Ranná špička S-J",
    "description": "Priorita pre Sever-Juh",
    "cycle_duration": 120,
    "signal_timings": {
      "N_S": {"start": 0, "duration": 60},
      "N_L": {"start": 60, "duration": 10},
      ...
    }
  }'
```

### 2. Získanie všetkých konfigurácií
```bash
curl -X GET http://localhost:8000/api/intersection/configurations
```

### 3. Spustenie simulácie
```bash
curl -X POST http://localhost:8000/api/intersection/simulations/start \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "conf_abc123",
    "simulation_duration": 300,
    "traffic_intensity": {
      "north": 25,
      "south": 25,
      "east": 15,
      "west": 15
    },
    "vehicle_speed": 10
  }'
```

### 4. Získanie štatistík
```bash
curl -X GET http://localhost:8000/api/intersection/simulations/sim_xyz789/stats
```

---

## 5. Dátový model

### Tabuľka: configurations
```sql
CREATE TABLE configurations (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    cycle_duration INTEGER NOT NULL,
    signal_timings JSON NOT NULL,
    is_preset BOOLEAN DEFAULT FALSE,
    cycle_utilization FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Tabuľka: simulations
```sql
CREATE TABLE simulations (
    id VARCHAR(50) PRIMARY KEY,
    config_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    simulation_duration INTEGER,
    traffic_intensity JSON,
    vehicle_speed FLOAT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    total_vehicles_generated INTEGER,
    total_vehicles_passed INTEGER,
    average_wait_time FLOAT,
    max_wait_time FLOAT,
    intersection_utilization FLOAT,
    FOREIGN KEY (config_id) REFERENCES configurations(id)
);
```

---

## Tipy a odporúčania

### Pre backend:
- Vytvorte maticu konfliktov (12x12) pre detekciu
- Použite Poissonov proces pre generovanie áut
- Implementujte efektívne dátové štruktúry (deque pre rady)
- Cachujte validáciu konfliktov v konfigurácii

### Pre frontend:
- Použite drag & drop knižnicu pre editor (Interact.js)
- Implementujte debouncing pri validácii
- Použite WebWorkers pre náročné výpočty
- Canvas optimalizácia pre plynulú animáciu

---

**Veľa úspechov pri realizácii projektu! 🚦**

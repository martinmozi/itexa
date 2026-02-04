# Zadanie projektu: Simulátor kmitajúcej pružiny so závažím

## Úvod
Vytvorte webovú aplikáciu pre real-time simuláciu kmitajúcej pružiny so závažím. Aplikácia využíva backend v Pythone pre fyzikálne výpočty, frontend v JavaScripte pre vizualizáciu a WebSocket pre jednosmernú komunikáciu v reálnom čase.

---

## Architektúra komunikácie

### Communication Flow:
```
FÁZA 1: VYTVORENIE KONFIGURÁCIE
1. Používateľ vyplní formulár (hmotnosť, tuhost, tlmenie, výchylka, krok, metóda)
   ↓
2. Frontend → REST API: POST /api/spring/configurations
   ↓
3. Backend: Validácia parametrov → Výpočet charakteristík → Uloženie konfigurácie
   ↓
4. Backend → Frontend: JSON response (config_id, charakteristiky systému)
   ↓
5. Frontend: Zobrazenie uloženej konfigurácie v zozname

FÁZA 2: SPRÁVA KONFIGURÁCIÍ
6. Frontend → REST API: GET /api/spring/configurations
   ↓
7. Backend → Frontend: Zoznam všetkých uložených konfigurácií
   ↓
8. Frontend: Zobrazenie zoznamu s možnosťou výberu

FÁZA 3: SPUSTENIE SIMULÁCIE
9. Používateľ vyberie konfiguráciu a nastaví trvanie simulácie
   ↓
10. Frontend → REST API: POST /api/spring/simulations/start
    Body: {"config_id": "conf_abc123", "duration": 10.0}
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
16. Backend → Frontend: Posledná správa (type: "completed")
   ↓
17. Frontend: Zatvorenie WebSocket spojenia
```

**Kľúčové charakteristiky:**
- **Oddelenie konfigurácie fyzikálneho systému a spustenia simulácie**
- Konfigurácie obsahujú fyzikálne parametre a numerickú metódu
- Trvanie simulácie sa nastavuje pri spustení, nie pri konfigurácii
- Rovnakú konfiguráciu možno spustiť s rôznym trvaním
- REST API slúži na správu konfigurácií a riadenie simulácií
- WebSocket je **jednosmerný** (server → klient) - prenos simulačných dát
- Backend riadi celý beh simulácie nezávisle po spustení
- Frontend iba vizualizuje prijímané dáta

---

## 1. Backend (Python)

### 1.1 Fyzikálny model
Implementujte simulátor tlmeného kmitavého pohybu podľa diferenciálnej rovnice:
```
m·ẍ + c·ẋ + k·x = 0
```
kde:
- `m` = hmotnosť závaží [kg]
- `c` = koeficient tlmenia [kg/s]
- `k` = tuhost pružiny [N/m]
- `x` = výchylka [m]

**Požiadavky:**
- Implementujte minimálne **3 rôzne numerické metódy**:
  1. **Eulerova metóda** - jednoduchá, 1. rád presnosti
  2. **Runge-Kutta 4. rádu** - presnejšia, 4. rád presnosti
  3. **Analytická** - najpresnejšia (pre podkritické tlmenie)
- Časový krok simulácie: konfigurovateľný používateľom (odporúčané: 0.001 - 0.1s)
- Validácia vstupných parametrov:
  - `m > 0` (hmotnosť musí byť kladná)
  - `k > 0` (tuhost musí byť kladná)
  - `c ≥ 0` (tlmenie nemôže byť záporné)
  - `|x₀| < 5m` (realistická počiatočná výchylka)
  - `0.001 ≤ time_step ≤ 0.1` (stabilita numeriky)
  - `duration > 0` a `duration ≤ 60s` (rozumné trvanie)
- Výpočet charakteristík systému:
  - **Perióda** kmitania: T = 2π/ω
  - **Frekvencia**: f = 1/T
  - **Typ tlmenia**: 
    - Podkritické (c < 2√(km)) - kmitavý pohyb
    - Kritické (c = 2√(km)) - aperiodický návrat
    - Nadkritické (c > 2√(km)) - pomalý návrat
  - **Koeficient tlmenia**: ζ = c / (2√(km))
  - **Uhol fázového posunu** (ak je tlmený)
- Asynchrónny beh simulácie (background task po spustení cez REST API)

### 1.2 REST API - Správa konfigurácií

#### `POST /api/spring/configurations`
Vytvorí a uloží novú konfiguráciu fyzikálneho systému. **Nespúšťa simuláciu**, len validuje a ukladá parametre.

**Request:**
```json
{
  "name": "Slabo tlmený systém",
  "description": "Pružina s malým tlmením pre študijnú ukážku",
  "mass": 1.0,
  "stiffness": 10.0,
  "damping": 0.5,
  "initial_displacement": 0.5,
  "time_step": 0.01,
  "numerical_method": "runge_kutta_4"
}
```

**Podporované numerické metódy:**
- `"euler"` - Eulerova metóda (1. rád presnosti)
- `"runge_kutta_2"` - Runge-Kutta 2. rádu
- `"runge_kutta_4"` - Runge-Kutta 4. rádu (predvolená)
- `"analytical"` - analytická metóda (iba pre podkritické tlmenie)

**Odpoveď:**
```json
{
  "config_id": "conf_abc123",
  "name": "Slabo tlmený systém",
  "description": "Pružina s malým tlmením pre študijnú ukážku",
  "parameters": {
    "mass": 1.0,
    "stiffness": 10.0,
    "damping": 0.5,
    "initial_displacement": 0.5,
    "time_step": 0.01,
    "numerical_method": "runge_kutta_4"
  },
  "characteristics": {
    "period": 2.83,
    "frequency": 0.35,
    "damping_type": "underdamped",
    "damping_ratio": 0.079,
    "natural_frequency": 3.16
  },
  "created_at": "2025-11-08T10:30:00Z"
}
```

#### `GET /api/spring/configurations`
Vráti zoznam všetkých uložených konfigurácií.

**Odpoveď:**
```json
{
  "configurations": [
    {
      "config_id": "conf_abc123",
      "name": "Slabo tlmený systém",
      "description": "Pružina s malým tlmením",
      "parameters": {
        "mass": 1.0,
        "stiffness": 10.0,
        "damping": 0.5,
        "initial_displacement": 0.5
      },
      "characteristics": {
        "period": 2.83,
        "frequency": 0.35,
        "damping_type": "underdamped"
      },
      "created_at": "2025-11-08T10:30:00Z",
      "times_simulated": 7
    },
    {
      "config_id": "conf_def456",
      "name": "Netlmený systém",
      "parameters": {...},
      "characteristics": {...},
      "created_at": "2025-11-08T11:00:00Z",
      "times_simulated": 3
    }
  ]
}
```

#### `GET /api/spring/configurations/{config_id}`
Vráti detail konkrétnej konfigurácie.

#### `PUT /api/spring/configurations/{config_id}`
Aktualizuje existujúcu konfiguráciu.

#### `DELETE /api/spring/configurations/{config_id}`
Zmaže konfiguráciu (iba ak nie je práve spustená žiadna simulácia s touto konfiguráciou).

### 1.3 REST API - Správa simulácií

#### `POST /api/spring/simulations/start`
Spustí simuláciu na základe existujúcej konfigurácie.

**Request:**
```json
{
  "config_id": "conf_abc123",
  "duration": 10.0
}
```

**Validácia:**
- `duration > 0` a `duration ≤ 60s`

**Odpoveď:**
```json
{
  "simulation_id": "sim_xyz789",
  "config_id": "conf_abc123",
  "config_name": "Slabo tlmený systém",
  "websocket_url": "ws://localhost:8000/ws/sim_xyz789",
  "status": "running",
  "duration": 10.0,
  "started_at": "2025-11-08T10:35:00Z"
}
```

#### `GET /api/spring/simulations`
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
      "config_id": "conf_abc123",
      "config_name": "Slabo tlmený systém",
      "status": "running",
      "duration": 10.0,
      "started_at": "2025-11-08T10:35:00Z",
      "elapsed_time": 3.5
    },
    {
      "simulation_id": "sim_abc456",
      "config_id": "conf_abc123",
      "config_name": "Slabo tlmený systém",
      "status": "completed",
      "duration": 10.0,
      "started_at": "2025-11-08T10:30:00Z",
      "completed_at": "2025-11-08T10:30:10Z",
      "total_steps": 1000
    }
  ]
}
```

#### `GET /api/spring/simulations/{simulation_id}`
Vráti detail konkrétnej simulácie.

#### `DELETE /api/spring/simulations/{simulation_id}`
Zastaví bežiacu simuláciu.

**Odpoveď:**
```json
{
  "status": "stopped",
  "simulation_id": "sim_xyz789",
  "elapsed_time": 5.25
}
```

#### Ďalšie endpointy:
- `POST /api/spring/configurations/validate` - validácia parametrov pred vytvorením konfigurácie
- `GET /api/spring/configurations/{config_id}/history` - história simulácií pre danú konfiguráciu
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
  "config_id": "conf_abc123",
  "parameters": {
    "mass": 1.0,
    "stiffness": 10.0,
    "damping": 0.5,
    "initial_displacement": 0.5,
    "time_step": 0.01,
    "method": "runge_kutta_4"
  },
  "characteristics": {
    "period": 2.83,
    "frequency": 0.35,
    "damping_type": "underdamped",
    "damping_ratio": 0.079
  },
  "duration": 10.0
}
```

**Následné správy (dáta simulácie)** - každých 50ms:
```json
{
  "type": "data",
  "time": 0.5,
  "displacement": 0.23,
  "velocity": -1.2,
  "acceleration": 3.4,
  "kinetic_energy": 0.72,
  "potential_energy": 0.26,
  "total_energy": 0.98
}
```

**Posledná správa:**
```json
{
  "type": "completed",
  "total_time": 10.0,
  "total_steps": 1000,
  "final_displacement": 0.05,
  "final_velocity": -0.12
}
```

---

## 2. Frontend (JavaScript/HTML/CSS)

### 2.1 Používateľské rozhranie

**Sekcia 1: Správa konfigurácií**
- Formulár na vytvorenie novej konfigurácie:
  - Názov konfigurácie (string)
  - Popis (voliteľné)
  - Hmotnosť závaží (kg) - default 1.0
  - Tuhost pružiny (N/m) - default 10.0
  - Koeficient tlmenia (kg/s) - default 0.5
  - Počiatočná výchylka (m) - default 0.5
  - **Časový krok simulácie** (s) - default 0.01
  - **Numerická metóda** - dropdown (Euler, RK2, RK4, Analytical)
- Tlačidlo "Uložiť konfiguráciu"
- Zobrazenie charakteristík systému po uložení:
  - Perióda a frekvencia
  - Typ tlmenia (podkritické/kritické/nadkritické)
  - Koeficient tlmenia
- Validácia na strane klienta (rozsahy hodnôt, numerické vstupy)

**Sekcia 2: Zoznam konfigurácií**
- Tabuľka/karty zobrazujúce všetky uložené konfigurácie
- Pre každú konfiguráciu:
  - Názov a popis
  - Základné parametre (m, k, c, x₀)
  - Charakteristiky (perióda, typ tlmenia)
  - Počet spustení
  - Tlačidlá: "Spustiť simuláciu", "Upraviť", "Zmazať", "Duplikovať"
- Možnosť filtrovania (podľa typu tlmenia, metódy)
- Možnosť zoradenia (podľa názvu, dátumu, počtu spustení)

**Sekcia 3: Nastavenie simulácie**
- Výber konfigurácie zo zoznamu
- Nastavenie trvania simulácie (s) - default 10.0
- Tlačidlo "Spustiť simuláciu"

**Sekcia 4: Vizualizácia simulácie**
- Canvas/SVG zobrazenie pružiny a závaží
- Real-time animácia pohybu závaží
- Zobrazenie aktuálnych hodnôt:
  - Čas (s)
  - Výchylka (m)
  - Rýchlosť (m/s)
  - Akcelerácia (m/s²)
  - Kinetická energia (J)
  - Potenciálna energia (J)
  - Celková energia (J)
- Grafy:
  - Výchylka v čase
  - Rýchlosť v čase
  - Fázový diagram (výchylka vs. rýchlosť)
  - Energia v čase
- Tlačidlá: Stop, Reset
- Plynulá animácia (60 FPS)

**Sekcia 5: História simulácií**
- Zoznam posledných spustených simulácií
- Filter podľa konfigurácie
- Porovnanie výsledkov

- Responzívny dizajn (mobile-friendly)

### 2.2 Workflow používateľa

**Vytvorenie konfigurácie:**
1. Používateľ vyplní formulár s fyzikálnymi parametrami
2. Klikne "Uložiť konfiguráciu"
3. Frontend: POST `/api/spring/configurations`
4. Zobrazenie charakteristík systému
5. Konfigurácia sa objaví v zozname konfigurácií

**Spustenie simulácie:**
1. Používateľ vyberie konfiguráciu zo zoznamu
2. Nastaví trvanie simulácie
3. Klikne "Spustiť simuláciu"
4. Frontend: POST `/api/spring/simulations/start` s `config_id` a `duration`
5. Získanie `simulation_id` a `websocket_url`
6. Otvorenie WebSocket spojenia
7. Real-time vizualizácia

**Správa konfigurácií:**
1. Načítanie všetkých konfigurácií: GET `/api/spring/configurations`
2. Zobrazenie v tabuľke/kartách
3. Možnosť úpravy, mazania, duplikovania, spúšťania

### 2.3 Príklad implementácie

```javascript
// Vytvorenie novej konfigurácie
async function saveConfiguration() {
    const configData = {
        name: document.getElementById('configName').value,
        description: document.getElementById('configDesc').value,
        mass: parseFloat(document.getElementById('mass').value),
        stiffness: parseFloat(document.getElementById('stiffness').value),
        damping: parseFloat(document.getElementById('damping').value),
        initial_displacement: parseFloat(document.getElementById('displacement').value),
        time_step: parseFloat(document.getElementById('timeStep').value),
        numerical_method: document.getElementById('method').value
    };
    
    const response = await fetch('/api/spring/configurations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configData)
    });
    
    const config = await response.json();
    displayCharacteristics(config.characteristics);
    addConfigToList(config);
}

// Načítanie všetkých konfigurácií
async function loadConfigurations() {
    const response = await fetch('/api/spring/configurations');
    const data = await response.json();
    
    displayConfigurations(data.configurations);
}

// Spustenie simulácie z vybranej konfigurácie
async function startSimulation(configId) {
    const duration = parseFloat(document.getElementById('duration').value);
    
    const response = await fetch('/api/spring/simulations/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            config_id: configId,
            duration: duration
        })
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
                initializeVisualization(message.parameters, message.characteristics);
                break;
            case 'data':
                updateVisualization(message);
                updateGraphs(message);
                displayValues(message);
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

// Aktualizácia vizualizácie
function updateVisualization(data) {
    // Aktualizuj pozíciu závaží
    drawSpring(data.displacement);
    
    // Aktualizuj grafy
    addDataToChart('displacement', data.time, data.displacement);
    addDataToChart('velocity', data.time, data.velocity);
    updatePhaseSpace(data.displacement, data.velocity);
    addDataToChart('energy', data.time, data.total_energy);
    
    // Aktualizuj zobrazované hodnoty
    document.getElementById('time').textContent = data.time.toFixed(2);
    document.getElementById('displacement').textContent = data.displacement.toFixed(3);
    document.getElementById('velocity').textContent = data.velocity.toFixed(3);
    document.getElementById('acceleration').textContent = data.acceleration.toFixed(3);
}
```

### 2.4 Vizualizácia
Vytvorte animáciu pomocou **Canvas API** alebo **SVG**:
- Grafické znázornenie pružiny a závaží
  - Pružina sa správne rozťahuje/sťahuje
  - Realistický efekt pomocou sínusového tvaru závitov
- Real-time pohyb závaží podľa prijatých dát
- Grafy (Chart.js/D3.js):
  - Výchylka v čase
  - Rýchlosť v čase
  - Fázový diagram (elipsa pre netlmený systém, špirála pre tlmený)
  - Energia v čase (konštantná pre netlmený, klesajúca pre tlmený)
- Plynulá animácia (60 FPS)
- Zobrazenie aktuálneho času, výchylky, rýchlosti, akcelerácie, energií

---

## 3. Minimálne požiadavky na funkcionalitu

### Backend musí:
1. Správne implementovať fyzikálny model tlmeného kmitavého pohybu s minimálne 3 numerickými metódami
2. **Poskytnúť API pre správu konfigurácií** (CRUD operácie)
3. **Poskytnúť API pre správu simulácií** (spustenie, zastavenie, história)
4. Ukladať konfigurácie a históriu simulácií do databázy (SQLite/PostgreSQL)
5. Generovať unikátne `config_id` a `simulation_id`
6. Bežať simuláciu asynchrónne (background task) po spustení cez REST API
7. Posielať dáta cez WebSocket v správnom formáte (setup → data → completed)
8. Podporovať viacero súčasných simulácií
9. Validovať všetky vstupy a odmietnuť neplatné hodnoty
10. Počítať charakteristiky systému pri vytvorení konfigurácie

### Frontend musí:
1. Poskytovať formulár pre vytvorenie novej konfigurácie
2. **Zobrazovať zoznam všetkých uložených konfigurácií**
3. Umožňovať výber konfigurácie a nastavenie trvania simulácie
4. **Volať správne API endpointy** (konfigurácie vs simulácie)
5. Pripojiť sa na WebSocket pomocou `simulation_id`
6. Spracovať správy typu "setup", "data" a "completed"
7. Vizualizovať pohyb pružiny v reálnom čase
8. Vykresliť grafy (výchylka, rýchlosť, fázový diagram, energia)
9. Správne spracovať výpadky spojenia
10. Umožniť manuálne zastavenie simulácie
11. Umožniť úpravu a mazanie konfigurácií
12. Byť responzívny (mobile-friendly)

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
curl -X POST http://localhost:8000/api/spring/configurations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Netlmený systém",
    "description": "Ideálna pružina bez tlmenia",
    "mass": 2.0,
    "stiffness": 15.0,
    "damping": 0.0,
    "initial_displacement": 1.0,
    "time_step": 0.01,
    "numerical_method": "runge_kutta_4"
  }'
```

**Odpoveď:**
```json
{
  "config_id": "conf_abc123",
  "name": "Netlmený systém",
  "parameters": {...},
  "characteristics": {
    "period": 2.29,
    "frequency": 0.44,
    "damping_type": "undamped",
    "damping_ratio": 0.0
  },
  "created_at": "2025-11-08T10:30:00Z"
}
```

### 2. Získanie všetkých konfigurácií
```bash
curl -X GET http://localhost:8000/api/spring/configurations
```

### 3. Spustenie simulácie
```bash
curl -X POST http://localhost:8000/api/spring/simulations/start \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "conf_abc123",
    "duration": 15.0
  }'
```

**Odpoveď:**
```json
{
  "simulation_id": "sim_xyz789",
  "config_id": "conf_abc123",
  "config_name": "Netlmený systém",
  "websocket_url": "ws://localhost:8000/ws/sim_xyz789",
  "status": "running",
  "duration": 15.0,
  "started_at": "2025-11-08T10:35:00Z"
}
```

### 4. Získanie všetkých simulácií
```bash
curl -X GET "http://localhost:8000/api/spring/simulations?status=completed&limit=10"
```

### 5. Zastavenie simulácie
```bash
curl -X DELETE http://localhost:8000/api/spring/simulations/sim_xyz789
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
    stiffness FLOAT NOT NULL,
    damping FLOAT NOT NULL,
    initial_displacement FLOAT NOT NULL,
    time_step FLOAT NOT NULL,
    numerical_method VARCHAR(50) NOT NULL,
    period FLOAT,
    frequency FLOAT,
    damping_type VARCHAR(20),
    damping_ratio FLOAT,
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
    duration FLOAT NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    total_steps INTEGER,
    final_displacement FLOAT,
    final_velocity FLOAT,
    FOREIGN KEY (config_id) REFERENCES configurations(id)
);
```

---

## Tipy a odporúčania

### Pre backend:
- Pre numerickú integráciu použite knižnicu `scipy.integrate.odeint` alebo implementujte vlastný solver
- Implementujte analytické riešenie pre podkritické tlmenie:
  ```python
  x(t) = A * exp(-ζ*ω₀*t) * cos(ω_d*t + φ)
  kde ω_d = ω₀ * sqrt(1 - ζ²)
  ```
- Pre systém 1. rádu:
  ```python
  def derivatives(state, t):
      x, v = state
      dxdt = v
      dvdt = -(c/m)*v - (k/m)*x
      return [dxdt, dvdt]
  ```
- Použite ORM (SQLAlchemy) pre prácu s databázou
- Cachujte charakteristiky systému v konfigurácii

### Pre frontend:
- Pre Canvas animáciu použite `requestAnimationFrame()` pre plynulé 60 FPS
- Pri vizualizácii pružiny použite sínusový tvar pre realistický efekt:
  ```javascript
  for (let i = 0; i < coils; i++) {
      const y = startY + (i / coils) * height;
      const x = centerX + amplitude * Math.sin(i * frequency);
      // Vykresli záviť
  }
  ```
- Implementujte zoom/pan pre fázový diagram
- Použite rôzne farby pre rôzne typy tlmenia
- Zobrazujte tooltips s informáciami pri hover nad grafmi

### Pre DevOps:
- Databázové migrácie (Alembic pre SQLAlchemy)
- Backup stratégia pre databázu
- Monitoring aktívnych simulácií
- Health check endpointy

---

## Ďalšie rozšírenia (nepovinné)

- **Porovnanie numerických metód:** Spustite rovnakú konfiguráciu s rôznymi metódami, zobrazujte všetky trajektórie naraz
- **Porovnanie s analytickým riešením:** Vizuálne zobrazenie rozdielu
- **Viacero pružín naraz:** Simulácia systému spojených pružín
- **Export dát:** Export trajektórie do CSV/JSON
- **3D vizualizácia:** Three.js zobrazenie
- **Gamifikácia:** Uhádni parametre systému na základe vizualizácie

---

**Veľa úspechov pri realizácii projektu! **
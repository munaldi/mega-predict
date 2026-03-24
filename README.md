# M-SAP — Mega-Sena Analytics Platform

A Node.js web application that uses **TensorFlow.js** to train a neural network on historical Mega-Sena lottery data and predict the next 6 numbers.

> **Disclaimer:** Lottery draws are genuinely random events. Neural networks cannot predict truly random outcomes. This project is a technical demonstration of TensorFlow.js, not financial advice.

---

## Quick Start

```bash
npm install
npm start
# → http://localhost:3000
```

---

## Project Structure

```
mega-sena-project/
│
├── server.js                        # Express server (API + static files)
├── package.json
│
├── data/
│   └── sorteios.csv                 # 2,062 historical Mega-Sena draws (1996–2018)
│
├── public/
│   └── index.html                   # SPA entry point (SAP Fiori UI)
│
└── src/
    ├── engine/                      # ← INDEPENDENT TRAINING ENGINE
    │   ├── neuralEngine.js          #    Web Worker: TensorFlow.js model
    │   └── engineBridge.js          #    Bridge: Frontend ↔ Worker messaging
    │
    └── frontend/                    # ← FRONTEND (UI only)
        ├── css/
        │   └── style.css            #    SAP Fiori Belize design system
        └── js/
            └── app.js               #    UI controller, API calls, rendering
```

### Architecture: Separation of Concerns

The project enforces a **strict boundary** between the training engine and the frontend:

```
┌─────────────┐    postMessage     ┌──────────────────┐
│  Frontend   │ ◄═══════════════► │  Neural Engine    │
│  (app.js)   │   EngineBridge    │  (Web Worker)     │
│             │                    │                   │
│  • UI render│                    │  • TensorFlow.js  │
│  • API calls│                    │  • Data encoding  │
│  • DOM mgmt │                    │  • Model training │
└──────┬──────┘                    │  • Prediction     │
       │ fetch()                   └───────────────────┘
       │
┌──────▼──────┐
│  Express    │
│  REST API   │
│  (server.js)│
└─────────────┘
```

**Why this matters:**
- The neural engine runs in a **Web Worker** — training never blocks the UI thread
- The engine has **zero DOM dependencies** — it can be reused in Node.js, another framework, or a different UI
- Communication is purely via **message passing** (postMessage), making it trivial to swap implementations

---

## Neural Network Details

### Architecture

```
Input(600) → Dense(256, ReLU) → Dropout(0.3)
           → Dense(128, ReLU) → Dropout(0.2)
           → Dense(64, ReLU)
           → Dense(60, Sigmoid)
```

### Encoding

Each draw is represented as a **60-dimensional binary vector** where position `i` is `1` if number `i+1` was drawn:

```
Draw [5, 12, 33, 41, 50, 60] →
[0,0,0,0,1, 0,0,0,0,0, 0,1,0,..., 1,..., 1,..., 1]
```

The model input is a **sliding window** of 10 consecutive draws, flattened into a 600-dimensional vector.

### Training

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Binary Cross-Entropy (multi-label classification)
- **Epochs:** 30
- **Batch Size:** 64
- **Validation Split:** 10%

---

## API Endpoints

| Method | Route               | Description                    |
|--------|---------------------|--------------------------------|
| GET    | `/api/draws`        | All draws (for training)       |
| GET    | `/api/draws/page/N` | Paginated draws (20 per page)  |
| POST   | `/api/draws`        | Add a new draw `{numbers:[]}` |
| GET    | `/api/frequency`    | Number frequency analysis      |

---

## Screens

### Sorteios
- Paginated table of all historical draws
- Filter by number
- Add new draw results
- Frequency heatmap with most/least common numbers

### Neural Prediction
- One-click training with live progress (loss, accuracy, val loss)
- Predicted 6 numbers with confidence scores
- Full 60-number heatmap showing model confidence for each ball

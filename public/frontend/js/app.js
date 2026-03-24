// ═══════════════════════════════════════════════════════════════
// src/frontend/js/app.js
// ═══════════════════════════════════════════════════════════════
// Controlador do frontend. Gerencia as telas, renderiza tabelas
// e se comunica com o motor neural via EngineBridge.
// ═══════════════════════════════════════════════════════════════

// ── EngineBridge (inlined) ────────────────────────────────────────
class EngineBridge {
    constructor() {
        this.worker = new Worker('/engine/neuralEngine.js');
        this.listeners = {};
        this.worker.onmessage = (e) => {
            const { type, data } = e.data;
            if (this.listeners[type]) {
                this.listeners[type].forEach(fn => fn(data));
            }
        };
    }
    on(type, callback) {
        if (!this.listeners[type]) this.listeners[type] = [];
        this.listeners[type].push(callback);
        return this;
    }
    off(type) {
        if (type) { delete this.listeners[type]; } else { this.listeners = {}; }
        return this;
    }
    train(draws) { this.worker.postMessage({ action: 'train', draws }); }
    predict(draws) { this.worker.postMessage({ action: 'predict', draws }); }
    trainAndPredict(draws) { this.worker.postMessage({ action: 'trainAndPredict', draws }); }
    destroy() { this.worker.terminate(); this.listeners = {}; }
}

// ── Estado ──────────────────────────────────────────────────────
const state = {
    draws: [],
    page: 0,
    totalPages: 1,
    filter: '',
    engine: null,
    training: false,
    trained: false,
    logs: [],
    prediction: null,
    allScores: null,
};

// ── Inicialização ─────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    // Inicializar ponte com o motor neural
    state.engine = new EngineBridge();
    setupEngineListeners();

    // Configurar navegação
    setupNavigation();

    // Carregar dados iniciais
    await loadDraws();
    await loadFrequency();

    // Configurar handlers de eventos
    setupAddDraw();
    setupFilter();
    setupTrainButton();
    setupBulkImport();

    document.getElementById('btn-refresh').addEventListener('click', async () => {
        const btn = document.getElementById('btn-refresh');
        btn.disabled = true;
        await loadDraws();
        await loadFrequency();
        btn.disabled = false;
    });

    // Mostrar tela de sorteios por padrão
    switchScreen('lottery');
});

// ── Navegação ─────────────────────────────────────────────────

function setupNavigation() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            switchScreen(tab.dataset.screen);
        });
    });
}

function switchScreen(screenId) {
    // Atualizar abas
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.nav-tab[data-screen="${screenId}"]`)?.classList.add('active');

    // Atualizar telas
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(`screen-${screenId}`)?.classList.add('active');
}

// ── Tela de Sorteios ─────────────────────────────────────────────

async function loadDraws() {
    try {
        const res = await fetch('/api/draws');
        const data = await res.json();
        state.draws = data.draws;
        state.totalPages = Math.ceil(data.total / 20);
        renderDrawTable();
        updateFooter();
    } catch (err) {
        console.error('Falha ao carregar sorteios:', err);
    }
}

function getFilteredDraws() {
    if (!state.filter) return state.draws;
    const num = parseInt(state.filter);
    if (isNaN(num)) return state.draws;
    return state.draws.filter(d => d.includes(num));
}

function renderDrawTable() {
    const filtered = getFilteredDraws();
    const pageSize = 20;
    const total = filtered.length;
    state.totalPages = Math.ceil(total / pageSize);

    // Mais recentes primeiro
    const start = Math.max(0, total - (state.page + 1) * pageSize);
    const end = total - state.page * pageSize;
    const pageDraws = filtered.slice(start, end).reverse();

    const tbody = document.getElementById('draws-tbody');
    tbody.innerHTML = '';

    pageDraws.forEach((d, i) => {
        const idx = total - state.page * pageSize - i;
        const tr = document.createElement('tr');

        // Número do concurso
        const tdC = document.createElement('td');
        tdC.className = 'concurso-num';
        tdC.textContent = idx;
        tr.appendChild(tdC);

        // 6 dezenas
        d.forEach(n => {
            const td = document.createElement('td');
            const hue = (n * 6) % 360;
            td.innerHTML = `<span class="ball" style="background:hsl(${hue},55%,92%);color:hsl(${hue},65%,35%)">${String(n).padStart(2, '0')}</span>`;
            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    });

    // Atualizar contagem e paginação
    document.getElementById('draw-count').textContent = `${filtered.length} sorteios`;
    document.getElementById('page-info').textContent = `Página ${state.page + 1} de ${state.totalPages}`;
    document.getElementById('btn-older').disabled = state.page >= state.totalPages - 1;
    document.getElementById('btn-newer').disabled = state.page <= 0;
}

function setupFilter() {
    const input = document.getElementById('filter-input');
    input.addEventListener('input', () => {
        state.filter = input.value;
        state.page = 0;
        renderDrawTable();
    });

    document.getElementById('btn-older').addEventListener('click', () => {
        state.page++;
        renderDrawTable();
    });
    document.getElementById('btn-newer').addEventListener('click', () => {
        state.page--;
        renderDrawTable();
    });
}

function setupAddDraw() {
    const inputs = document.querySelectorAll('.add-draw__input');
    const btn = document.getElementById('btn-add-draw');

    btn.addEventListener('click', async () => {
        const nums = Array.from(inputs).map(inp => parseInt(inp.value));

        if (nums.some(n => isNaN(n) || n < 1 || n > 60)) return;
        if (new Set(nums).size !== 6) return;

        try {
            const res = await fetch('/api/draws', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ numbers: nums }),
            });

            if (res.ok) {
                inputs.forEach(inp => inp.value = '');
                await loadDraws();
                await loadFrequency();
            }
        } catch (err) {
            console.error('Falha ao adicionar sorteio:', err);
        }
    });
}

// ── Importação em Lote (CSV) ──────────────────────────────────────

function readFileAsText(file, onProgress) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = () => reject(new Error('Falha ao ler arquivo'));
        reader.onprogress = e => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
        reader.readAsText(file);
    });
}

function setProgress(pct, label) {
    document.getElementById('csv-progress').style.display = 'block';
    document.getElementById('csv-progress-fill').style.width = pct + '%';
    document.getElementById('bulk-import-status').textContent = label;
}

function hideProgress() {
    document.getElementById('csv-progress').style.display = 'none';
    document.getElementById('csv-progress-fill').style.width = '0%';
}

function setupBulkImport() {
    document.getElementById('btn-download-template').addEventListener('click', () => {
        const csv = 'Concurso,D1,D2,D3,D4,D5,D6\n';
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
        a.download = 'template_sorteios.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(a.href);
    });

    document.getElementById('input-csv-upload').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Passo 1 — leitura do arquivo (0–40%)
        setProgress(0, 'Lendo arquivo...');
        let text;
        try {
            text = await readFileAsText(file, pct => setProgress(pct * 40, 'Lendo arquivo...'));
        } catch (err) {
            setProgress(0, 'Erro ao ler arquivo.');
            e.target.value = '';
            return;
        }

        // Passo 2 — parse do CSV (40–70%)
        setProgress(40, 'Processando CSV...');
        const lines = text.trim().split(/\r?\n/).slice(1);
        const incoming = [];
        for (const line of lines) {
            const cols = line.trim().split(',');
            if (cols.length < 7) continue;
            const concurso = parseInt(cols[0], 10);
            const numbers = [cols[1], cols[2], cols[3], cols[4], cols[5], cols[6]].map(s => parseInt(s.trim(), 10));
            if (isNaN(concurso) || numbers.some(isNaN)) continue;
            incoming.push({ concurso, numbers });
        }

        if (incoming.length === 0) {
            setProgress(0, 'Nenhum dado válido encontrado no arquivo.');
            e.target.value = '';
            return;
        }
        setProgress(70, `${incoming.length} linha(s) encontradas, enviando...`);

        // Passo 3 — envio ao servidor (70–100%)
        try {
            const res = await fetch('/api/draws/bulk', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ draws: incoming }),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const { added, skipped } = await res.json();

            setProgress(100, `${added} adicionado(s), ${skipped} ignorado(s) (já existiam ou inválidos)`);

            if (added > 0) {
                state.page = 0; // volta para os mais recentes
                await loadDraws();
                await loadFrequency();
            }
        } catch (err) {
            setProgress(0, 'Erro ao importar: ' + err.message);
            console.error(err);
        }

        e.target.value = '';
    });
}

async function loadFrequency() {
    try {
        const res = await fetch('/api/frequency');
        const data = await res.json();
        renderFrequency(data);
    } catch (err) {
        console.error('Falha ao carregar frequência:', err);
    }
}

function renderFrequency(data) {
    const grid = document.getElementById('freq-grid');
    const maxFreq = Math.max(...data.frequency);
    grid.innerHTML = '';

    data.frequency.forEach((f, i) => {
        const pct = f / maxFreq;
        const hue = pct > 0.8 ? 0 : pct > 0.5 ? 30 : 210;
        const alpha = (0.15 + pct * 0.6).toFixed(2);

        const cell = document.createElement('div');
        cell.className = 'freq-cell';
        cell.innerHTML = `
            <div class="freq-cell__bar" style="background:hsla(${hue},70%,50%,${alpha});color:hsl(${hue},70%,30%)">
                ${String(i + 1).padStart(2, '0')}
            </div>
            <div class="freq-cell__count">${f}</div>
        `;
        grid.appendChild(cell);
    });

    document.getElementById('most-frequent').textContent =
        data.mostFrequent.map(x => String(x.number).padStart(2, '0')).join(', ');
    document.getElementById('least-frequent').textContent =
        data.leastFrequent.map(x => String(x.number).padStart(2, '0')).join(', ');
}

// ── Tela de Previsão Neural ─────────────────────────────────────────

function setupTrainButton() {
    document.getElementById('btn-train').addEventListener('click', () => {
        if (state.training) return;
        startTraining();
    });
}

function startTraining() {
    state.training = true;
    state.logs = [];
    state.prediction = null;
    state.allScores = null;

    // Atualizar interface
    const btn = document.getElementById('btn-train');
    btn.disabled = true;
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="#fff" stroke-width="1.2"><path d="M2 14V6h3v8H2zm4.5 0V2h3v12h-3zM11 14V8h3v6h-3z"/></svg> Treinamento em andamento...`;

    document.getElementById('training-card').classList.remove('hidden');
    document.getElementById('prediction-card').classList.add('hidden');
    document.getElementById('heatmap-card').classList.add('hidden');

    // Limpar tabela de log
    document.getElementById('epoch-tbody').innerHTML = '';

    // Atualizar contagem de sorteios no cabeçalho
    document.getElementById('draw-count-predict').textContent = state.draws.length;

    // Enviar para o motor neural
    state.engine.trainAndPredict(state.draws);
}

function setupEngineListeners() {
    state.engine
        .on('epoch', (data) => {
            state.logs.push(data);
            renderEpochLog(data);
            updateProgressBar(data.epoch, data.totalEpochs);
            updateMetrics(data);
        })
        .on('trained', () => {
            document.querySelector('.training-card__status').textContent = 'Concluído';
            document.querySelector('.training-card__status').className = 'training-card__status training-card__status--done';
            document.querySelector('.progress-bar__fill').className = 'progress-bar__fill progress-bar__fill--done';
        })
        .on('prediction', (data) => {
            state.training = false;
            state.prediction = data.top6;
            state.allScores = data.scores;

            renderPrediction(data.top6);
            renderHeatmap(data.scores, data.top6);

            const btn = document.getElementById('btn-train');
            btn.disabled = false;
            btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="#fff" stroke-width="1.2"><path d="M4 3l9 5-9 5V3z"/></svg> Retreinar Modelo`;
        })
        .on('error', (data) => {
            state.training = false;
            console.error('Erro do motor neural:', data.message);
            const btn = document.getElementById('btn-train');
            btn.disabled = false;
            btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="#fff" stroke-width="1.2"><path d="M4 3l9 5-9 5V3z"/></svg> Treinar e Prever`;
        });
}

function renderEpochLog(data) {
    const tbody = document.getElementById('epoch-tbody');
    const tr = document.createElement('tr');
    tr.innerHTML = `
        <td style="text-align:center">${data.epoch}</td>
        <td style="text-align:center;color:var(--error)">${data.loss.toFixed(5)}</td>
        <td style="text-align:center;color:var(--success)">${(data.acc * 100).toFixed(2)}%</td>
        <td style="text-align:center;color:var(--warning)">${data.valLoss.toFixed(5)}</td>
    `;
    tbody.appendChild(tr);

    // Rolar automaticamente para o final
    const container = document.querySelector('.epoch-log');
    container.scrollTop = container.scrollHeight;

    // Atualizar status
    document.querySelector('.training-card__status').textContent = `Época ${data.epoch}/${data.totalEpochs}`;
}

function updateProgressBar(epoch, total) {
    const pct = (epoch / total * 100).toFixed(1);
    document.querySelector('.progress-bar__fill').style.width = `${pct}%`;
}

function updateMetrics(data) {
    document.getElementById('metric-loss').textContent = data.loss.toFixed(4);
    document.getElementById('metric-acc').textContent = (data.acc * 100).toFixed(1) + '%';
    document.getElementById('metric-val').textContent = data.valLoss.toFixed(4);
}

function renderPrediction(top6) {
    const container = document.getElementById('prediction-balls');
    container.innerHTML = '';

    top6.forEach((n, i) => {
        const div = document.createElement('div');
        div.className = 'prediction-ball';
        div.style.animationDelay = `${i * 0.1}s`;
        div.textContent = String(n).padStart(2, '0');
        container.appendChild(div);
    });

    document.getElementById('prediction-concurso').textContent = `#${state.draws.length + 1}`;
    document.getElementById('prediction-card').classList.remove('hidden');
}

function renderHeatmap(scores, top6) {
    const grid = document.getElementById('heatmap-grid');
    grid.innerHTML = '';

    scores.forEach((s, i) => {
        const num = i + 1;
        const isTop = top6.includes(num);
        const cell = document.createElement('div');
        cell.className = `heatmap__cell${isTop ? ' heatmap__cell--top' : ''}`;

        if (!isTop) {
            cell.style.background = `rgba(10,110,209,${s * 0.8})`;
            cell.style.color = s > 0.3 ? '#fff' : 'var(--text)';
        }

        cell.innerHTML = `
            <div class="heatmap__num">${String(num).padStart(2, '0')}</div>
            <div class="heatmap__score">${(s * 100).toFixed(1)}%</div>
        `;
        grid.appendChild(cell);
    });

    document.getElementById('heatmap-card').classList.remove('hidden');
}

// ── Rodapé ─────────────────────────────────────────────────────

function updateFooter() {
    document.getElementById('footer-count').textContent = state.draws.length;
}

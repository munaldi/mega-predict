// ═══════════════════════════════════════════════════════════════
// server.js — Servidor Express para M-SAP Mega-Sena Analytics
// Serve o frontend e fornece uma API REST para dados de sorteios
// ═══════════════════════════════════════════════════════════════

const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;
const isVercel = !!process.env.VERCEL;

// ── Middleware ──────────────────────────────────────────────────
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ── Armazém de Dados (em memória, carregado do CSV) ────────────
let draws = [];

function loadDraws() {
    const csvPath = path.join(__dirname, 'data', 'sorteios.csv');
    if (!fs.existsSync(csvPath)) {
        console.warn('⚠  data/sorteios.csv não encontrado — iniciando com dados vazios');
        return;
    }
    const raw = fs.readFileSync(csvPath, 'utf-8');
    const lines = raw.trim().split('\n').slice(1); // pula o cabeçalho
    const seen = new Set();

    lines.forEach(line => {
        const cols = line.split(',');
        const concurso = parseInt(cols[1]);
        if (seen.has(concurso)) return; // evita duplicatas
        seen.add(concurso);

        const numbers = [
            parseInt(cols[3]),
            parseInt(cols[4]),
            parseInt(cols[5]),
            parseInt(cols[6]),
            parseInt(cols[7]),
            parseInt(cols[8]),
        ];

        if (numbers.every(n => n >= 1 && n <= 60)) {
            draws.push({ concurso, numbers });
        }
    });

    draws.sort((a, b) => a.concurso - b.concurso);
    console.log(`✓  ${draws.length} sorteios únicos carregados do CSV`);
}

// ── Rotas da API ─────────────────────────────────────────────

// GET /api/draws — retorna todos os sorteios (usado pelo motor de treino)
app.get('/api/draws', (req, res) => {
    res.json({
        total: draws.length,
        draws: draws.map(d => d.numbers),
    });
});

// GET /api/draws/page/:page — sorteios paginados para a tabela
app.get('/api/draws/page/:page', (req, res) => {
    const pageSize = 20;
    const page = parseInt(req.params.page) || 0;
    const total = draws.length;
    const totalPages = Math.ceil(total / pageSize);

    // Retorna os mais recentes primeiro
    const start = Math.max(0, total - (page + 1) * pageSize);
    const end = total - page * pageSize;
    const slice = draws.slice(start, end).reverse();

    res.json({ page, totalPages, total, draws: slice });
});

// POST /api/draws — adiciona um novo sorteio
app.post('/api/draws', (req, res) => {
    const { numbers } = req.body;

    if (!Array.isArray(numbers) || numbers.length !== 6) {
        return res.status(400).json({ error: 'Deve fornecer exatamente 6 números' });
    }

    const nums = numbers.map(Number);
    if (nums.some(n => n < 1 || n > 60 || !Number.isInteger(n))) {
        return res.status(400).json({ error: 'Os números devem ser inteiros entre 1 e 60' });
    }
    if (new Set(nums).size !== 6) {
        return res.status(400).json({ error: 'Os números devem ser únicos' });
    }

    const nextConcurso = draws.length > 0 ? draws[draws.length - 1].concurso + 1 : 1;
    const newDraw = { concurso: nextConcurso, numbers: nums.sort((a, b) => a - b) };
    draws.push(newDraw);

    res.status(201).json(newDraw);
});

// GET /api/frequency — análise de frequência dos números
app.get('/api/frequency', (req, res) => {
    const freq = new Array(60).fill(0);
    draws.forEach(d => d.numbers.forEach(n => freq[n - 1]++));

    const ranked = freq
        .map((count, i) => ({ number: i + 1, count }))
        .sort((a, b) => b.count - a.count);

    res.json({
        frequency: freq,
        mostFrequent: ranked.slice(0, 6),
        leastFrequent: ranked.slice(-6).reverse(),
    });
});

// ── Servir frontend para todas as outras rotas ─────────────────
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Carrega os dados uma única vez na inicialização da função/processo.
if (draws.length === 0) {
    loadDraws();
}

// Em ambiente local, inicia o servidor HTTP normalmente.
if (!isVercel) {
    app.listen(PORT, () => {
        console.log(`\n🎱  M-SAP Mega-Sena Analytics rodando em http://localhost:${PORT}\n`);
    });
}

// Em Vercel, o Express app é usado como handler serverless.
module.exports = app;

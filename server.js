// ═══════════════════════════════════════════════════════════════
// server.js — Servidor Express para M-SAP Mega-Sena Analytics
// Serve o frontend e fornece uma API REST para dados de sorteios
// ═══════════════════════════════════════════════════════════════

const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// ── Middleware ──────────────────────────────────────────────────
app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ── Armazém de Dados (carregado do JSON) ────────────────────────
const draws = require('./data/sorteios.json').slice();

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

// POST /api/draws/bulk — importa múltiplos sorteios; ignora concursos duplicados
app.post('/api/draws/bulk', (req, res) => {
    const { draws: incoming } = req.body;
    if (!Array.isArray(incoming)) {
        return res.status(400).json({ error: 'Formato inválido' });
    }

    const existingConcursos = new Set(draws.map(d => d.concurso));
    let added = 0, skipped = 0;

    incoming.forEach(item => {
        const concurso = parseInt(item.concurso);
        const numbers = (item.numbers || []).map(Number);

        if (isNaN(concurso) || existingConcursos.has(concurso)) { skipped++; return; }
        if (numbers.length !== 6 || numbers.some(n => isNaN(n) || n < 1 || n > 60)) { skipped++; return; }
        if (new Set(numbers).size !== 6) { skipped++; return; }

        draws.push({ concurso, numbers: numbers.sort((a, b) => a - b) });
        existingConcursos.add(concurso);
        added++;
    });

    draws.sort((a, b) => a.concurso - b.concurso);
    res.json({ added, skipped });
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

app.listen(PORT, () => {
    console.log(`\n🎱  M-SAP Mega-Sena Analytics rodando em http://localhost:${PORT}\n`);
});

module.exports = app;

// ═══════════════════════════════════════════════════════════════
// src/engine/neuralEngine.js
// ═══════════════════════════════════════════════════════════════
//
// Motor de treinamento e previsão com TensorFlow.js.
// Roda dentro de um Web Worker — totalmente independente do frontend.
//
// ┌─────────────────────────────────────────────────────────────┐
// │                    ARQUITETURA DA REDE                      │
// │                                                             │
// │  Entrada (600 valores)                                      │
// │    ↓                                                        │
// │  Camada Densa (256 neurônios, ativação ReLU)                │
// │    ↓                                                        │
// │  Dropout (30% — descarta neurônios aleatoriamente)          │
// │    ↓                                                        │
// │  Camada Densa (128 neurônios, ativação ReLU)                │
// │    ↓                                                        │
// │  Dropout (20%)                                              │
// │    ↓                                                        │
// │  Camada Densa (64 neurônios, ativação ReLU)                 │
// │    ↓                                                        │
// │  Camada de Saída (60 neurônios, ativação Sigmoid)           │
// │    → Cada neurônio representa um número da Mega-Sena (1-60) │
// │    → Valor entre 0 e 1 = confiança de que será sorteado     │
// └─────────────────────────────────────────────────────────────┘
//
// CODIFICAÇÃO DOS DADOS:
//   Cada sorteio é transformado num vetor binário de 60 posições.
//   Posição i = 1 se o número (i+1) foi sorteado, 0 caso contrário.
//   Exemplo: sorteio [5, 12, 33] → [0,0,0,0,1, 0,0,0,0,0, 0,1,..., 1,...]
//
//   O modelo recebe WINDOW_SIZE sorteios consecutivos concatenados,
//   formando um vetor de (WINDOW_SIZE × 60) = 600 valores.
//
// COMUNICAÇÃO (via postMessage com o frontend):
//   O Worker recebe:
//     { action: 'train',           draws: [[n,...], ...] }
//     { action: 'predict',         draws: [...] }
//     { action: 'trainAndPredict', draws: [...] }
//
//   O Worker envia:
//     { type: 'status',     data: { message: '...' } }
//     { type: 'epoch',      data: { epoch, totalEpochs, loss, acc, valLoss } }
//     { type: 'trained',    data: { success: true } }
//     { type: 'prediction', data: { top6: [...], scores: [...] } }
//     { type: 'error',      data: { message: '...' } }
//
// ═══════════════════════════════════════════════════════════════

// Importa o TensorFlow.js via CDN (necessário dentro do Web Worker)
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');

// ── Hiperparâmetros ────────────────────────────────────────────
// Esses valores controlam o comportamento do treinamento.
// Alterar eles muda a forma como a rede aprende.

const WINDOW_SIZE = 10;     // Quantos sorteios passados o modelo analisa por vez
                            // → Janela de 10 sorteios = o modelo "olha" os 10 últimos
                            //   resultados para tentar prever o próximo

const NUM_BALLS   = 60;     // A Mega-Sena sorteia números de 1 a 60

const EPOCHS      = 30;     // Quantas vezes o modelo passa por TODOS os dados de treino
                            // → Mais épocas = mais tempo de treino, potencialmente mais preciso
                            // → Épocas demais = overfitting (decora ao invés de aprender)

const BATCH_SIZE  = 64;     // Quantas amostras o modelo processa por vez antes de atualizar pesos
                            // → Batches maiores = mais rápido, mas usa mais memória
                            // → Batches menores = mais lento, mas pode generalizar melhor

const LEARN_RATE  = 0.001;  // Taxa de aprendizado do otimizador Adam
                            // → Muito alta = o modelo "pula" o ponto ótimo
                            // → Muito baixa = treinamento muito lento

const VAL_SPLIT   = 0.1;    // 10% dos dados são separados para validação
                            // → Serve para detectar overfitting: se a perda de treino
                            //   cai mas a de validação sobe, o modelo está decorando

// ── Estado do Worker ───────────────────────────────────────────
// O modelo treinado fica armazenado aqui entre chamadas de predict()
let model = null;

// ═══════════════════════════════════════════════════════════════
// PREPARAÇÃO DOS DADOS
// ═══════════════════════════════════════════════════════════════

/**
 * Codifica um único sorteio como vetor binário de 60 dimensões.
 *
 * Por que binário?
 *   → Redes neurais trabalham com números. Precisamos transformar
 *     "sorteou o número 5" numa representação numérica.
 *   → O vetor binário é a forma mais direta: cada posição
 *     representa um número possível (1-60), com 1 = sorteado.
 *
 * Exemplo:
 *   Sorteio [5, 12, 33, 41, 50, 60] gera:
 *   índice:  0  1  2  3  4  5  6 ... 11 ... 32 ... 40 ... 49 ... 59
 *   valor:   0  0  0  0  1  0  0 ...  1 ...  1 ...  1 ...  1 ...  1
 *                       ↑              ↑      ↑      ↑      ↑      ↑
 *                     num 5          num 12  num 33  num 41  num 50  num 60
 *
 * @param {number[]} draw - Array com 6 números sorteados (ex: [5, 12, 33, 41, 50, 60])
 * @returns {Float32Array} - Vetor binário de 60 posições
 */
function encodeDraw(draw) {
    const vec = new Float32Array(NUM_BALLS);  // Cria vetor zerado com 60 posições
    draw.forEach(n => { vec[n - 1] = 1; });  // Marca 1 nas posições dos números sorteados
    return vec;                               // (n-1 porque arrays começam do índice 0)
}

/**
 * Constrói os pares (X, Y) de treinamento a partir do histórico completo.
 *
 * ESTRATÉGIA DE JANELA DESLIZANTE:
 *   Imagine o histórico como uma fila de sorteios:
 *     [sorteio_1] [sorteio_2] [sorteio_3] ... [sorteio_N]
 *
 *   Para cada posição i (a partir de WINDOW_SIZE):
 *     X = últimos WINDOW_SIZE sorteios antes de i, concatenados
 *     Y = o sorteio na posição i (o que queremos prever)
 *
 *   Exemplo com WINDOW_SIZE = 3:
 *     Amostra 1: X = [sorteio_1 + sorteio_2 + sorteio_3]  →  Y = sorteio_4
 *     Amostra 2: X = [sorteio_2 + sorteio_3 + sorteio_4]  →  Y = sorteio_5
 *     Amostra 3: X = [sorteio_3 + sorteio_4 + sorteio_5]  →  Y = sorteio_6
 *     ...
 *
 *   Cada X tem dimensão (WINDOW_SIZE × 60) = 600 valores
 *   Cada Y tem dimensão 60 (vetor binário do sorteio alvo)
 *
 * @param {number[][]} draws - Array de sorteios [[n1,n2,...n6], ...]
 * @returns {{ xs: tf.Tensor2D, ys: tf.Tensor2D, inputDim: number, sampleCount: number }}
 */
function prepareTrainingData(draws) {
    const xs = [];  // Entradas (features)
    const ys = [];  // Rótulos (labels / o que queremos prever)

    // Percorre todos os sorteios a partir da posição WINDOW_SIZE
    // (precisamos de WINDOW_SIZE sorteios anteriores para montar cada X)
    for (let i = WINDOW_SIZE; i < draws.length; i++) {

        // ── Montar o vetor de entrada (X) ──
        // Concatena os vetores binários dos últimos WINDOW_SIZE sorteios
        const window = [];
        for (let j = i - WINDOW_SIZE; j < i; j++) {
            window.push(...encodeDraw(draws[j]));
            // Cada encodeDraw retorna 60 valores
            // Ao final do loop, 'window' tem (WINDOW_SIZE × 60) valores
        }
        xs.push(window);

        // ── Montar o rótulo (Y) ──
        // O sorteio atual é o "gabarito" que a rede deve aprender
        ys.push(Array.from(encodeDraw(draws[i])));
    }

    // Converte arrays JavaScript em Tensores do TensorFlow
    // Tensor2D: cada linha = uma amostra, cada coluna = uma feature
    return {
        xs: tf.tensor2d(xs),            // Forma: [numAmostras, 600]
        ys: tf.tensor2d(ys),            // Forma: [numAmostras, 60]
        inputDim: WINDOW_SIZE * NUM_BALLS, // = 600 (dimensão da entrada)
        sampleCount: xs.length,          // Total de amostras geradas
    };
}

// ═══════════════════════════════════════════════════════════════
// CONSTRUÇÃO DO MODELO (ARQUITETURA DA REDE NEURAL)
// ═══════════════════════════════════════════════════════════════

/**
 * Cria e compila o modelo sequencial (camadas empilhadas uma sobre a outra).
 *
 * A rede é um "classificador multi-rótulo": para cada um dos 60 números,
 * ela prevê independentemente a probabilidade dele aparecer no próximo sorteio.
 *
 * @param {number} inputDim - Dimensão do vetor de entrada (600)
 * @returns {tf.Sequential} - Modelo compilado pronto para treinar
 */
function buildModel(inputDim) {
    const m = tf.sequential();

    // ┌─────────────────────────────────────────────────────────────┐
    // │ CAMADA 1 — Entrada ampla (256 neurônios)                   │
    // │                                                             │
    // │ Por que 256?                                                │
    // │   → É uma camada larga que consegue capturar muitas         │
    // │     combinações diferentes de features. Como a entrada      │
    // │     tem 600 dimensões, precisamos de capacidade suficiente  │
    // │     para detectar padrões complexos.                        │
    // │                                                             │
    // │ Ativação ReLU (Rectified Linear Unit):                      │
    // │   → f(x) = max(0, x)                                       │
    // │   → Mantém apenas sinais positivos, zerando os negativos    │
    // │   → Permite que a rede aprenda padrões não-lineares         │
    // │   → É a ativação mais usada em camadas ocultas              │
    // └─────────────────────────────────────────────────────────────┘
    m.add(tf.layers.dense({
        inputShape: [inputDim],   // Forma da entrada: vetor de 600 valores
        units: 256,               // 256 neurônios nesta camada
        activation: 'relu',       // Ativação ReLU
    }));

    // ┌─────────────────────────────────────────────────────────────┐
    // │ DROPOUT 1 — Regularização (30%)                             │
    // │                                                             │
    // │ O que faz?                                                  │
    // │   → Durante o treinamento, "desliga" aleatoriamente 30%     │
    // │     dos neurônios a cada passo                              │
    // │                                                             │
    // │ Por quê?                                                    │
    // │   → Evita overfitting (o modelo decorar os dados ao invés   │
    // │     de aprender padrões generalizáveis)                     │
    // │   → Força a rede a não depender de nenhum neurônio          │
    // │     específico — como se vários modelos menores             │
    // │     aprendessem juntos                                      │
    // │                                                             │
    // │ Nota: durante a previsão (predict), o dropout é desligado   │
    // │   automaticamente — todos os neurônios são usados           │
    // └─────────────────────────────────────────────────────────────┘
    m.add(tf.layers.dropout({ rate: 0.3 }));

    // ┌─────────────────────────────────────────────────────────────┐
    // │ CAMADA 2 — Compressão (128 neurônios)                       │
    // │                                                             │
    // │ Por que menos neurônios que a camada anterior?               │
    // │   → O afunilamento progressivo (256 → 128 → 64) força      │
    // │     a rede a "destilar" a informação, mantendo apenas       │
    // │     os padrões mais relevantes                              │
    // │   → É como um resumo: de muitos sinais, extrai os          │
    // │     mais importantes                                        │
    // └─────────────────────────────────────────────────────────────┘
    m.add(tf.layers.dense({ units: 128, activation: 'relu' }));

    // ┌─────────────────────────────────────────────────────────────┐
    // │ DROPOUT 2 — Regularização adicional (20%)                   │
    // │                                                             │
    // │ Taxa menor que o primeiro dropout porque a camada já é      │
    // │ menor — não queremos descartar informação demais neste      │
    // │ ponto da rede                                               │
    // └─────────────────────────────────────────────────────────────┘
    m.add(tf.layers.dropout({ rate: 0.2 }));

    // ┌─────────────────────────────────────────────────────────────┐
    // │ CAMADA 3 — Refinamento final (64 neurônios)                 │
    // │                                                             │
    // │ A camada mais estreita da rede antes da saída.              │
    // │ Aqui a informação já está bastante comprimida — apenas      │
    // │ os padrões mais fortes e representativos sobrevivem         │
    // └─────────────────────────────────────────────────────────────┘
    m.add(tf.layers.dense({ units: 64, activation: 'relu' }));

    // ┌─────────────────────────────────────────────────────────────┐
    // │ CAMADA DE SAÍDA — 60 neurônios com Sigmoid                  │
    // │                                                             │
    // │ Cada neurônio representa um número da Mega-Sena (1 a 60)   │
    // │                                                             │
    // │ Ativação Sigmoid:                                           │
    // │   → f(x) = 1 / (1 + e^(-x))                                │
    // │   → Comprime qualquer valor para o intervalo [0, 1]         │
    // │   → Interpretação: probabilidade daquele número ser         │
    // │     sorteado no próximo concurso                            │
    // │                                                             │
    // │ Exemplo de saída:                                           │
    // │   neurônio 0 = 0.85 → "85% de confiança que o número 1     │
    // │                        será sorteado"                       │
    // │   neurônio 4 = 0.12 → "12% de confiança que o número 5     │
    // │                        será sorteado"                       │
    // │                                                             │
    // │ Por que Sigmoid e não Softmax?                               │
    // │   → Softmax serve para classificação exclusiva              │
    // │     (apenas UMA classe pode ser a resposta)                 │
    // │   → Sigmoid permite classificação multi-rótulo              │
    // │     (VÁRIOS números podem ser sorteados ao mesmo tempo)     │
    // │   → Na Mega-Sena, 6 números são sorteados — então          │
    // │     precisamos de classificação multi-rótulo                │
    // └─────────────────────────────────────────────────────────────┘
    m.add(tf.layers.dense({ units: NUM_BALLS, activation: 'sigmoid' }));

    // ┌─────────────────────────────────────────────────────────────┐
    // │ COMPILAÇÃO — Define como o modelo aprende                   │
    // │                                                             │
    // │ Otimizador Adam:                                            │
    // │   → Algoritmo que ajusta os pesos da rede a cada passo      │
    // │   → Adam = Adaptive Moment Estimation                       │
    // │   → Combina as vantagens do SGD com momentum e RMSProp      │
    // │   → Adapta a taxa de aprendizado para cada peso             │
    // │     individualmente — pesos que mudam pouco recebem          │
    // │     ajustes maiores, e vice-versa                           │
    // │                                                             │
    // │ Função de perda — Binary Cross-Entropy:                     │
    // │   → Mede o quão "errado" o modelo está                      │
    // │   → Ideal para problemas de classificação binária           │
    // │     (cada número: sorteado ou não sorteado)                 │
    // │   → Valor menor = previsão mais próxima da realidade        │
    // │                                                             │
    // │ Métrica — Accuracy (Acurácia):                              │
    // │   → Porcentagem de previsões corretas                       │
    // │   → Usada apenas para monitoramento, NÃO influencia         │
    // │     o treinamento (quem influencia é a função de perda)     │
    // └─────────────────────────────────────────────────────────────┘
    m.compile({
        optimizer: tf.train.adam(LEARN_RATE),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    return m;
}

// ═══════════════════════════════════════════════════════════════
// TREINAMENTO
// ═══════════════════════════════════════════════════════════════

/**
 * Treina a rede neural com os dados históricos de sorteios.
 *
 * O processo de treinamento funciona assim:
 *   1. Prepara os dados (transforma sorteios em tensores)
 *   2. Cria a arquitetura do modelo
 *   3. Executa o loop de treinamento por EPOCHS épocas:
 *      a. Para cada batch de amostras:
 *         - Faz a previsão (forward pass)
 *         - Calcula o erro (loss)
 *         - Propaga o erro de volta (backpropagation)
 *         - Ajusta os pesos da rede (gradient descent)
 *      b. Ao final de cada época, calcula métricas e envia ao frontend
 *   4. Libera memória dos tensores de treino
 *
 * @param {number[][]} draws - Histórico completo de sorteios
 */
async function train(draws) {
    // Se já existe um modelo treinado de uma execução anterior,
    // libera a memória dele antes de criar um novo
    if (model) {
        model.dispose();
        model = null;
    }

    // ── Passo 1: Preparar dados de treinamento ──
    const data = prepareTrainingData(draws);
    send('status', {
        message: `Preparadas ${data.sampleCount} amostras de treino (dimensão entrada=${data.inputDim})`
    });

    // ── Passo 2: Criar a arquitetura do modelo ──
    model = buildModel(data.inputDim);

    // ── Passo 3: Treinar o modelo ──
    // model.fit() executa o loop de treinamento completo
    await model.fit(data.xs, data.ys, {
        epochs: EPOCHS,           // Número total de passadas pelos dados
        batchSize: BATCH_SIZE,    // Amostras processadas por vez

        shuffle: true,            // Embaralha os dados a cada época
                                  // → Evita que o modelo aprenda a "ordem"
                                  //   dos dados ao invés dos padrões

        validationSplit: VAL_SPLIT,  // Separa 10% dos dados para validação
                                      // → Dados que o modelo NUNCA vê durante treino
                                      // → Serve para detectar overfitting

        callbacks: {
            // Executado ao final de cada época — envia métricas para o frontend
            onEpochEnd: (epoch, logs) => {
                send('epoch', {
                    epoch: epoch + 1,        // Época atual (começa de 1)
                    totalEpochs: EPOCHS,     // Total de épocas
                    loss: logs.loss,          // Perda no conjunto de treino
                                             //   → O quão "errado" o modelo está nos dados que ele viu
                    acc: logs.acc,            // Acurácia no treino
                                             //   → Porcentagem de acertos
                    valLoss: logs.val_loss,   // Perda no conjunto de validação
                                             //   → O quão "errado" em dados que ele NÃO viu
                                             //   → Se cresce enquanto loss cai = overfitting
                });
            },
        },
    });

    // ── Passo 4: Limpeza de memória ──
    // Tensores ocupam memória GPU/CPU. Precisamos liberar manualmente
    // pois o garbage collector do JS não gerencia tensores do TensorFlow
    data.xs.dispose();
    data.ys.dispose();

    // Avisa o frontend que o treinamento foi concluído com sucesso
    send('trained', { success: true });
}

// ═══════════════════════════════════════════════════════════════
// PREVISÃO
// ═══════════════════════════════════════════════════════════════

/**
 * Usa o modelo treinado para prever os próximos 6 números.
 *
 * O processo:
 *   1. Pega os últimos WINDOW_SIZE sorteios do histórico
 *   2. Codifica-os como vetor binário concatenado (entrada do modelo)
 *   3. Passa pelo modelo treinado (forward pass)
 *   4. O modelo retorna 60 scores (um por número possível)
 *   5. Seleciona os 6 números com maior score de confiança
 *
 * IMPORTANTE: A previsão NÃO usa dropout — todos os neurônios
 * participam para dar a melhor estimativa possível.
 *
 * @param {number[][]} draws - Histórico completo de sorteios
 */
function predict(draws) {
    // Verifica se o modelo foi treinado
    if (!model) {
        send('error', { message: 'Modelo não treinado. Envie "train" primeiro.' });
        return;
    }

    // ── Passo 1: Montar a entrada a partir dos últimos sorteios ──
    // Pega exatamente os últimos WINDOW_SIZE sorteios
    const lastWindow = [];
    for (let j = draws.length - WINDOW_SIZE; j < draws.length; j++) {
        lastWindow.push(...encodeDraw(draws[j]));
    }
    // lastWindow agora tem (WINDOW_SIZE × 60) = 600 valores

    // ── Passo 2: Converter para tensor e passar pelo modelo ──
    // tensor2d com shape [1, 600] — 1 amostra de 600 features
    const input = tf.tensor2d([lastWindow]);

    // model.predict() executa o forward pass:
    //   entrada (600) → camadas ocultas → saída (60 probabilidades)
    const prediction = model.predict(input);

    // ── Passo 3: Extrair os scores como array JavaScript ──
    // dataSync() bloqueia até o tensor estar pronto e retorna Float32Array
    const scores = Array.from(prediction.dataSync());

    // Limpeza de memória dos tensores temporários
    input.dispose();
    prediction.dispose();

    // ── Passo 4: Selecionar os 6 números mais prováveis ──
    // Cria array de objetos {número, score} e ordena por score decrescente
    const ranked = scores
        .map((score, i) => ({ number: i + 1, score }))  // i+1 porque números vão de 1 a 60
        .sort((a, b) => b.score - a.score);              // Maior score primeiro

    // Pega os 6 primeiros (maiores scores) e ordena por número
    const top6 = ranked
        .slice(0, 6)                                     // Top 6 mais confiantes
        .map(x => x.number)                              // Extrai apenas os números
        .sort((a, b) => a - b);                          // Ordena em ordem crescente (como na Mega-Sena)

    // Envia para o frontend: os 6 números escolhidos + todos os 60 scores
    // (os scores completos são usados para renderizar o mapa de calor)
    send('prediction', { top6, scores });
}

// ═══════════════════════════════════════════════════════════════
// COMUNICAÇÃO COM O FRONTEND
// ═══════════════════════════════════════════════════════════════

/**
 * Envia uma mensagem para o thread principal (frontend).
 * O Web Worker se comunica com o frontend exclusivamente via postMessage.
 *
 * @param {string} type - Tipo da mensagem ('epoch', 'trained', 'prediction', 'error', 'status')
 * @param {object} data - Dados da mensagem
 */
function send(type, data) {
    self.postMessage({ type, data });
}

/**
 * Handler de mensagens recebidas do frontend.
 * O frontend envia ações ('train', 'predict', 'trainAndPredict')
 * e o worker executa a operação correspondente.
 *
 * Todas as operações são envolvidas em try/catch para que
 * erros não derrubem o worker silenciosamente — o erro
 * é enviado de volta ao frontend para exibição ao usuário.
 */
self.onmessage = async (e) => {
    const { action, draws } = e.data;

    try {
        switch (action) {
            case 'train':
                // Apenas treina o modelo, sem fazer previsão
                await train(draws);
                break;

            case 'predict':
                // Apenas prevê (modelo já deve estar treinado)
                predict(draws);
                break;

            case 'trainAndPredict':
                // Treina e em seguida faz a previsão — fluxo principal
                await train(draws);
                predict(draws);
                break;

            default:
                send('error', { message: `Ação desconhecida: ${action}` });
        }
    } catch (err) {
        send('error', { message: err.message || String(err) });
    }
};

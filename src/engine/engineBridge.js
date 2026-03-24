// ═══════════════════════════════════════════════════════════════
// src/engine/engineBridge.js
// ═══════════════════════════════════════════════════════════════
// Ponte entre o frontend e o motor neural (Web Worker).
// O frontend NUNCA toca no TensorFlow diretamente — ele só
// se comunica através desta ponte via promessas e callbacks.
//
// Por que usar um Web Worker?
//   → O treinamento da rede neural é pesado e pode levar segundos
//   → Se rodasse no thread principal, a interface ficaria travada
//   → O Worker roda em thread separada, mantendo a UI responsiva
// ═══════════════════════════════════════════════════════════════

export class EngineBridge {
    constructor() {
        // Cria o Web Worker apontando para o motor neural
        this.worker = new Worker('/src/engine/neuralEngine.js');

        // Mapa de listeners: tipo → [callbacks]
        this.listeners = {};

        // Quando o Worker envia uma mensagem, redireciona
        // para todos os callbacks registrados para aquele tipo
        this.worker.onmessage = (e) => {
            const { type, data } = e.data;
            if (this.listeners[type]) {
                this.listeners[type].forEach(fn => fn(data));
            }
        };
    }

    /**
     * Registra um callback para um tipo específico de mensagem do motor.
     * Tipos disponíveis: 'epoch', 'trained', 'prediction', 'error', 'status'
     *
     * Exemplo:
     *   bridge.on('epoch', (data) => console.log(`Época ${data.epoch}`));
     *
     * @param {string} type - Tipo da mensagem
     * @param {function} callback - Função a ser chamada quando a mensagem chegar
     * @returns {EngineBridge} - Retorna this para encadeamento
     */
    on(type, callback) {
        if (!this.listeners[type]) this.listeners[type] = [];
        this.listeners[type].push(callback);
        return this;
    }

    /**
     * Remove todos os listeners de um tipo, ou todos se nenhum tipo for informado.
     *
     * @param {string} [type] - Tipo a limpar (opcional)
     * @returns {EngineBridge}
     */
    off(type) {
        if (type) {
            delete this.listeners[type];
        } else {
            this.listeners = {};
        }
        return this;
    }

    /**
     * Envia os dados de sorteios para o motor treinar a rede neural.
     * O progresso é reportado via callbacks 'epoch' e 'trained'.
     *
     * @param {number[][]} draws - Array de sorteios [[n1,...,n6], ...]
     */
    train(draws) {
        this.worker.postMessage({ action: 'train', draws });
    }

    /**
     * Solicita uma previsão ao motor (o modelo já deve estar treinado).
     * O resultado chega via callback 'prediction'.
     *
     * @param {number[][]} draws - Array de sorteios para basear a previsão
     */
    predict(draws) {
        this.worker.postMessage({ action: 'predict', draws });
    }

    /**
     * Treina e prevê em uma única operação — fluxo principal da aplicação.
     * Envia callbacks 'epoch' durante treino, 'trained' ao concluir,
     * e 'prediction' com o resultado final.
     *
     * @param {number[][]} draws - Histórico completo de sorteios
     */
    trainAndPredict(draws) {
        this.worker.postMessage({ action: 'trainAndPredict', draws });
    }

    /**
     * Encerra o Worker e limpa todos os listeners.
     * Chamar quando a ponte não for mais necessária.
     */
    destroy() {
        this.worker.terminate();
        this.listeners = {};
    }
}

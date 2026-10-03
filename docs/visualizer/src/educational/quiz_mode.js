/**
 * educational/quiz_mode.js
 * =========================
 * Interactive Pedagogy & Architecture Mastery Quiz Engine.
 * Provides progressive assessment quizzes covering attention mathematics,
 * low-rank SVD compression, variance stabilization, and GQA efficiency.
 */

import { escapeHtml } from '../utils/dom.js';

export const QUIZ_QUESTIONS = [
    {
        id: 'q1',
        title: 'Question 1: Softmax Variance Stabilization',
        question: 'Why must the Query-Key dot products (q_i · k_j^T) be divided by √d_k before softmax in multi-head attention?',
        options: [
            'To convert negative logits into non-negative values',
            'Because Var(q_i · k_j^T) = d_k under unit variance; without √d_k, large magnitudes push softmax into zero-gradient saturation',
            'To ensure the attention weights sum to 1.0 along sequence rows',
            'To enforce lower-triangular causal masking'
        ],
        correctIndex: 1,
        explanation: 'For independent random components with zero mean and unit variance, the dot product has variance d_k. Dividing by √d_k restores variance to 1.0, preventing vanishing gradients in deep backpropagation.'
    },
    {
        id: 'q2',
        title: 'Question 2: SVD Low-Rank Parameter Savings',
        question: 'When decomposing a 768 × 768 dense weight matrix W into rank r=64 bottleneck factors U (768×64) and V^T (64×768), what is the parameter reduction ratio?',
        options: [
            '2.00× (50.0% savings)',
            '4.00× (75.0% savings)',
            '6.00× (83.33% savings)',
            '12.00× (91.67% savings)'
        ],
        correctIndex: 2,
        explanation: 'Dense parameters = 768 × 768 = 589,824. Low-rank parameters = 64 × (768 + 768) = 98,304. Ratio = 589,824 / 98,304 = 6.00×, achieving 83.33% parameter savings.'
    },
    {
        id: 'q3',
        title: 'Question 3: GQA KV-Cache Memory Efficiency',
        question: 'In Grouped-Query Attention (GQA) with 14 Query heads and 2 Key-Value heads (7:1 ratio), how much KV-cache memory is saved compared to standard MHA?',
        options: [
            'No savings; KV cache size is independent of heads',
            '50% savings',
            '85.7% savings (7× reduction in KV cache storage)',
            '98% savings'
        ],
        correctIndex: 2,
        explanation: 'Standard MHA requires caching 14 KV head pairs per token. GQA caches only 2 shared KV pairs. KV-cache memory shrinks by 14 / 2 = 7× (85.71% memory reduction).'
    },
    {
        id: 'q4',
        title: 'Question 4: Recurrent vs Dense Attention Complexity',
        question: 'Why does a recurrent model like HGRN2 or Mamba scale linearly O(S) with sequence length S, whereas standard Transformer attention scales quadratically O(S^2)?',
        options: [
            'Recurrent models truncate the sequence after 512 tokens',
            'Recurrent models update a fixed-size latent state h_t at each step instead of materializing an S × S pairwise interaction matrix',
            'Recurrent models use 8-bit integer quantization',
            'Recurrent models use sparse random projection masks'
        ],
        correctIndex: 1,
        explanation: 'Recurrence computes h_t = f_t ⊙ h_{t-1} + i_t ⊙ v_t with a constant-sized hidden state tensor, avoiding all-to-all pairwise quadratic token comparisons.'
    },
    {
        id: 'q5',
        title: 'Question 5: Causal Attention Masking',
        question: 'In autoregressive generation (GPT-2, Qwen), what value is added to upper-triangular logits before softmax, and why?',
        options: [
            'Zero (0.0) to preserve current probabilities',
            '-Infinity (-∞), so exp(-∞) becomes exactly 0.0, strictly forbidding tokens from peeking into future tokens',
            '+Infinity (+∞) to maximize forward likelihood',
            '-1.0 to shift distribution down'
        ],
        correctIndex: 1,
        explanation: 'Adding -∞ ensures that when exponentiated in softmax (e^x), future positions evaluate to exactly 0.0, preserving strict temporal causality.'
    }
];

/**
 * Renders the Interactive Mastery Quiz into a DOM container.
 */
export function renderQuizWidget(container, onComplete = null) {
    if (!container) return;

    let currentQ = 0;
    let score = 0;
    let answered = false;

    const renderQuestion = () => {
        if (currentQ >= QUIZ_QUESTIONS.length) {
            // Quiz Complete View
            const pct = Math.round((score / QUIZ_QUESTIONS.length) * 100);
            container.innerHTML = `
                <div class="quiz-complete" style="text-align:center; padding:24px; font-family:var(--font-mono);">
                    <div style="font-size:36px; margin-bottom:12px;">🏆</div>
                    <h3 style="color:#fff; font-size:18px; margin:0 0 8px;">Mastery Quiz Completed!</h3>
                    <p style="color:var(--text-muted); font-size:12px; margin-bottom:16px;">
                        Score: <b style="color:var(--cyan); font-size:16px;">${score} / ${QUIZ_QUESTIONS.length}</b> (${pct}%)
                    </p>
                    <button id="btn-restart-quiz" class="primary" style="margin:0 auto; padding:8px 18px;">↻ Retake Quiz</button>
                </div>
            `;
            const restartBtn = container.querySelector('#btn-restart-quiz');
            if (restartBtn) restartBtn.onclick = () => { currentQ = 0; score = 0; answered = false; renderQuestion(); };
            if (onComplete) onComplete({ score, total: QUIZ_QUESTIONS.length, percentage: pct });
            return;
        }

        const q = QUIZ_QUESTIONS[currentQ];
        answered = false;

        container.innerHTML = `
            <div class="quiz-widget" style="display:flex; flex-direction:column; gap:12px; font-family:var(--font-mono); font-size:11px; background:#040911; padding:14px; border:1px solid var(--line); border-radius:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="hud-badge" style="background:#8b5cf6; color:#fff; font-weight:800;">MASTERY QUIZ</span>
                    <span style="color:var(--text-muted); font-size:10px;">Question ${currentQ + 1} of ${QUIZ_QUESTIONS.length}</span>
                </div>

                <h4 style="color:#fff; font-size:13px; margin:0; line-height:1.4;">${escapeHtml(q.question)}</h4>

                <!-- Options List -->
                <div class="quiz-options" style="display:flex; flex-direction:column; gap:6px;">
                    ${q.options.map((opt, i) => `
                        <button class="quiz-opt-btn" data-idx="${i}"
                            style="text-align:left; padding:8px 12px; background:rgba(255,255,255,0.03); border:1px solid var(--line); border-radius:6px; color:#e2e8f0; font-family:var(--font-mono); font-size:11px; cursor:pointer; transition:all 0.15s ease;">
                            <b style="color:var(--cyan);">${String.fromCharCode(65 + i)}.</b> ${escapeHtml(opt)}
                        </button>
                    `).join('')}
                </div>

                <!-- Explanation Box -->
                <div id="quiz-explanation" style="display:none; padding:10px; border-radius:6px; font-size:11px; line-height:1.5;"></div>

                <div style="display:flex; justify-content:flex-end;">
                    <button id="btn-next-q" class="secondary" style="display:none; padding:6px 14px; font-size:11px;">Next Question →</button>
                </div>
            </div>
        `;

        const optButtons = container.querySelectorAll('.quiz-opt-btn');
        const expBox = container.querySelector('#quiz-explanation');
        const nextBtn = container.querySelector('#btn-next-q');

        optButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                if (answered) return;
                answered = true;
                const chosen = parseInt(btn.dataset.idx, 10);
                const isCorrect = chosen === q.correctIndex;
                if (isCorrect) score++;

                optButtons.forEach((b, idx) => {
                    b.style.pointerEvents = 'none';
                    if (idx === q.correctIndex) {
                        b.style.background = 'rgba(16,185,129,0.2)';
                        b.style.borderColor = '#10b981';
                        b.style.color = '#34d399';
                    } else if (idx === chosen && !isCorrect) {
                        b.style.background = 'rgba(239,68,68,0.2)';
                        b.style.borderColor = '#ef4444';
                        b.style.color = '#f87171';
                    }
                });

                if (expBox) {
                    expBox.style.display = 'block';
                    expBox.style.background = isCorrect ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)';
                    expBox.style.border = `1px solid ${isCorrect ? '#10b981' : '#ef4444'}`;
                    expBox.style.color = isCorrect ? '#6ee7b7' : '#fca5a5';
                    expBox.innerHTML = `<b>${isCorrect ? '✓ Correct!' : '✗ Incorrect.'}</b> ${escapeHtml(q.explanation)}`;
                }

                if (nextBtn) {
                    nextBtn.style.display = 'inline-block';
                    nextBtn.onclick = () => { currentQ++; renderQuestion(); };
                }
            });
        });
    };

    renderQuestion();
}

/**
 * react/components/modals/LearnModal.jsx
 * =====================================
 * Pedagogical masterclass modal explaining the 9 attention paradigms, SVD compression, and linear recurrence.
 */

import React from 'react';

export default function LearnModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop react-modal" data-testid="learn-modal" role="dialog" aria-modal="true">
      <div className="modal-container learn-container">
        <div className="modal-header">
          <h3>Attention Lab · Masterclass Primer</h3>
          <button type="button" className="btn-close-modal" onClick={onClose} aria-label="Close modal">✕</button>
        </div>

        <div className="learn-content-scroll">
          <section className="learn-section">
            <h4>1. The Quadratic Attention Dilemma</h4>
            <p>
              Standard Multi-Head Attention (Vaswani et al., 2017) computes pairwise inner products between all pairs of tokens in a sequence:
              <code> Attention(Q, K, V) = softmax(QK^T / √d_k) V </code>.
              Because the resulting score matrix has dimensions <strong>S × S</strong>, the computational cost and memory footprint scale quadratically with sequence length O(S²).
            </p>
          </section>

          <section className="learn-section">
            <h4>2. Low-Rank Factorization & SVD (Category E)</h4>
            <p>
              Weight projection matrices exhibit steep singular decay (Eckart-Young theorem).
              By factorizing W = U Σ V^T and truncating to rank r ≪ min(d_in, d_out), we retain over 95% of spectral energy while reducing parameter count and FLOPs from O(d²) to O(2rd).
            </p>
          </section>

          <section className="learn-section">
            <h4>3. Linear Sequence Models & Recurrence (Category D)</h4>
            <p>
              Recurrent models (HGRN-2, Mamba, Delta-Net) eliminate the S × S attention matrix entirely.
              State updates <code>h_t = A h_{'{t-1}'} + B x_t</code> maintain constant O(1) memory complexity during autoregressive token generation.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}

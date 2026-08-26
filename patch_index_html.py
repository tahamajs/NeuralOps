from pathlib import Path

p = Path("Codes/framework/web/index.html")
content = p.read_text(encoding="utf-8")

# 1. Update loss-meta-badges
old_badges = '''            <div class="loss-meta-badges">
              <span class="meta-badge badge-module" id="live-loss-module">svd_linear_attention</span>
              <span class="meta-badge" id="live-loss-model">BERT</span>
              <span class="meta-badge" id="live-loss-dataset">SST-2</span>
              <span class="meta-badge badge-time" id="live-loss-time">Live</span>
            </div>'''

new_badges = '''            <div class="loss-meta-badges" style="display:flex; flex-wrap:wrap; gap:6px;">
              <span class="meta-badge badge-module" id="live-loss-module">full</span>
              <span class="meta-badge" id="live-loss-model">DistilBERT</span>
              <span class="meta-badge" id="live-loss-dataset">SST-2</span>
              <span class="meta-badge" id="live-loss-seed" style="background:rgba(14,165,233,0.2); color:#38bdf8; border:1px solid rgba(14,165,233,0.3);">Seed 42</span>
              <span class="meta-badge" id="live-loss-epoch" style="background:rgba(139,92,246,0.2); color:#c084fc; border:1px solid rgba(139,92,246,0.3);">Epoch 1/3</span>
              <span class="meta-badge" id="live-loss-val" style="background:rgba(16,185,129,0.2); color:#34d399; border:1px solid rgba(16,185,129,0.3);">Loss: -</span>
              <span class="meta-badge badge-time" id="live-loss-time">Live</span>
            </div>'''

content = content.replace(old_badges, new_badges, 1)

# Add progress mini-bar right above canvas
old_canvas = '''          <div class="chart-container">
            <canvas id="chart-live-loss"></canvas>
          </div>'''

new_canvas = '''          <div class="chart-container">
            <div style="margin-bottom: 8px; height: 5px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden;">
              <div id="live-loss-progress" style="height: 100%; width: 0%; background: linear-gradient(90deg, #6366f1, #06b6d4); transition: width 0.4s ease;"></div>
            </div>
            <canvas id="chart-live-loss"></canvas>
          </div>'''

content = content.replace(old_canvas, new_canvas, 1)

# 2. Add section-module-progress right before Recent Execution Stream
module_table_section = '''      <!-- Per-Module Live Status Table -->
      <section class="table-section" id="section-module-progress" style="margin-bottom: 24px;">
        <div class="table-header-flex">
          <div>
            <h2>🔬 Fast Quality Sweep Live Matrix (6 Modules)</h2>
            <span class="table-sub">Live multi-seed progress, per-epoch loss convergence, and real-time accuracy</span>
          </div>
          <div class="live-run-tag" style="font-size: 0.75rem; background: rgba(99,102,241,0.15); color: #818cf8; padding: 5px 12px; border-radius: 9999px; border: 1px solid rgba(99,102,241,0.3); font-weight:600;">
            ⚡ DistilBERT Pretrained (768d, 12h, 6L) • SST-2
          </div>
        </div>
        <div class="table-wrapper">
          <table class="live-table">
            <thead>
              <tr>
                <th>Module Architecture</th>
                <th>Status</th>
                <th>Current Seed</th>
                <th>Seeds Done</th>
                <th>Epoch</th>
                <th>Step</th>
                <th>Live Loss</th>
                <th>Mean Acc</th>
                <th>F1 Score</th>
                <th>Fidelity</th>
                <th>Progress</th>
                <th>ETA</th>
              </tr>
            </thead>
            <tbody id="module-progress-tbody">
              <!-- Dynamically populated from /api/stats module_progress -->
            </tbody>
          </table>
        </div>
      </section>

      <!-- Live Execution Table -->'''

content = content.replace('      <!-- Live Execution Table -->', module_table_section, 1)

p.write_text(content, encoding="utf-8")
print("index.html updated successfully.")

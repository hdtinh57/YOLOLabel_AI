/**
 * MLOps Dashboard — Training history, model registry, charts
 */
const Dashboard = {
  charts: {},
  selectedRuns: new Set(),
  currentRuns: [],
  currentModels: [],

  // --- Init ---
  init() {
    document.getElementById('dash-tab-overview')?.addEventListener('click', () => this.switchTab('overview'));
    document.getElementById('dash-tab-registry')?.addEventListener('click', () => this.switchTab('registry'));
    document.getElementById('btn-compare-runs')?.addEventListener('click', () => this.compareSelected());
    document.getElementById('run-detail-close')?.addEventListener('click', () => this.closeRunDetail());
    document.getElementById('compare-close')?.addEventListener('click', () => this.closeCompare());
  },

  // --- Tab switching ---
  switchTab(tab) {
    document.querySelectorAll('.dash-tab').forEach(t => t.classList.remove('active'));
    document.getElementById(`dash-tab-${tab}`)?.classList.add('active');

    document.getElementById('dash-overview-section').style.display = tab === 'overview' ? 'block' : 'none';
    document.getElementById('dash-registry-section').style.display = tab === 'registry' ? 'block' : 'none';
  },

  // --- Show/Hide Dashboard ---
  show() {
    document.querySelector('.app-main').style.display = 'none';
    document.getElementById('dashboard-view').classList.add('active');
    this.load();
  },

  hide() {
    document.getElementById('dashboard-view').classList.remove('active');
    document.querySelector('.app-main').style.display = 'flex';
    this.destroyCharts();
  },

  // --- Load all data ---
  async load() {
    try {
      const [stats, runs, models] = await Promise.all([
        API.mlops.getDashboardStats(),
        API.mlops.getRuns(),
        API.mlops.getModels(),
      ]);
      this.renderKPIs(stats);
      this.currentRuns = runs.runs || runs;
      this.renderRunsTable(this.currentRuns);
      this.currentModels = models.versions || models;
      this.renderRegistry(this.currentModels);
    } catch (err) {
      console.error('Dashboard load error:', err);
    }
  },

  // --- KPI Cards ---
  renderKPIs(stats) {
    const el = document.getElementById('kpi-cards');
    if (!el) return;

    const kpis = [
      { label: 'Total Runs', value: stats.total_runs ?? 0, cls: 'accent-blue' },
      { label: 'Best mAP@50', value: stats.best_map50 != null ? (stats.best_map50 * 100).toFixed(1) + '%' : '—', cls: 'accent-green' },
      { label: 'Production Model', value: stats.production_model ?? '—', cls: 'accent-cyan' },
      { label: 'AL Cycles', value: stats.total_al_cycles ?? 0, cls: 'accent-amber' },
    ];

    el.innerHTML = kpis.map(k => `
      <div class="kpi-card">
        <span class="kpi-label">${k.label}</span>
        <span class="kpi-value ${k.cls}">${k.value}</span>
      </div>
    `).join('');
  },

  // --- Runs Table ---
  renderRunsTable(runs) {
    const tbody = document.getElementById('runs-tbody');
    if (!tbody) return;

    if (!runs || runs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="dash-empty-cell" style="text-align: center; padding: 40px; color: var(--text-tertiary);">No training runs yet. Train a model to get started.</td></tr>`;
      return;
    }

    tbody.innerHTML = runs.map(r => {
      const date = new Date(r.started_at * 1000).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
      const duration = r.duration_sec ? this.formatDuration(r.duration_sec) : '—';
      const mAP50 = r.best_mAP50 != null ? (r.best_mAP50 * 100).toFixed(1) + '%' : '—';
      const mAP5095 = r.best_mAP50_95 != null ? (r.best_mAP50_95 * 100).toFixed(1) + '%' : '—';
      const checked = this.selectedRuns.has(r.id) ? 'checked' : '';

      return `
        <tr data-run-id="${r.id}">
          <td><input type="checkbox" class="run-checkbox" data-run-id="${r.id}" ${checked}></td>
          <td><code style="font-size:11px; color: var(--accent-cyan);" title="${r.id}">${r.id.substring(0, 20)}</code></td>
          <td><span class="status-badge ${r.status}">${r.status}</span></td>
          <td>${r.base_model || '—'}</td>
          <td>${r.epochs_done ?? 0}/${r.epochs_total ?? 0}</td>
          <td class="metric-cell">${mAP50}</td>
          <td class="metric-cell">${mAP5095}</td>
          <td>${date}</td>
          <td>${duration}</td>
          <td>
            <button class="btn btn-xs btn-ghost" onclick="Dashboard.showRunDetail('${r.id}')" title="View Details">📊</button>
            <button class="btn btn-xs btn-ghost" onclick="Dashboard.deleteRun('${r.id}')" title="Delete Run">🗑️</button>
          </td>
        </tr>
      `;
    }).join('');

    // Checkbox listeners
    tbody.querySelectorAll('.run-checkbox').forEach(cb => {
      cb.addEventListener('change', (e) => {
        const runId = e.target.dataset.runId;
        if (e.target.checked) {
          if (this.selectedRuns.size >= 5) {
            e.target.checked = false;
            return;
          }
          this.selectedRuns.add(runId);
        } else {
          this.selectedRuns.delete(runId);
        }
        this.updateCompareButton();
      });
    });

    // Row click → detail
    tbody.querySelectorAll('tr').forEach(tr => {
      tr.addEventListener('click', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
        const runId = tr.dataset.runId;
        if (runId) this.showRunDetail(runId);
      });
    });
  },

  updateCompareButton() {
    const btn = document.getElementById('btn-compare-runs');
    if (!btn) return;
    btn.disabled = this.selectedRuns.size < 2;
    btn.textContent = `Compare (${this.selectedRuns.size})`;
  },

  // --- Run Detail ---
  async showRunDetail(runId) {
    try {
      const detail = await API.mlops.getRunDetail(runId);
      const modal = document.getElementById('run-detail-modal');
      if (!modal) return;

      const run = detail.run || detail;
      const epochs = detail.epoch_metrics || run.epoch_metrics || [];

      // Header
      document.getElementById('detail-run-id').textContent = run.id;
      document.getElementById('detail-status').innerHTML = `<span class="status-badge ${run.status}">${run.status}</span>`;
      document.getElementById('detail-base-model').textContent = run.base_model || '—';

      // Stats grid
      const grid = document.getElementById('detail-stats-grid');
      const mAP50 = run.best_mAP50 != null ? (run.best_mAP50 * 100).toFixed(1) + '%' : '—';
      const mAP5095 = run.best_mAP50_95 != null ? (run.best_mAP50_95 * 100).toFixed(1) + '%' : '—';
      const precision = run.best_precision != null ? (run.best_precision * 100).toFixed(1) + '%' : '—';
      const recall = run.best_recall != null ? (run.best_recall * 100).toFixed(1) + '%' : '—';

      grid.innerHTML = [
        { label: 'mAP@50', value: mAP50 },
        { label: 'mAP@50:95', value: mAP5095 },
        { label: 'Precision', value: precision },
        { label: 'Recall', value: recall },
      ].map(s => `
        <div class="detail-stat">
          <div class="detail-stat-label">${s.label}</div>
          <div class="detail-stat-value">${s.value}</div>
        </div>
      `).join('');

      // Charts
      if (epochs.length > 0) {
        this.renderDetailCharts(epochs);
      }

      // Actions
      const promoteBtn = document.getElementById('detail-promote-btn');
      if (promoteBtn) {
        promoteBtn.onclick = () => this.promoteRun(run.id);
      }

      modal.style.display = 'flex';
    } catch (err) {
      console.error('Run detail error:', err);
    }
  },

  closeRunDetail() {
    const modal = document.getElementById('run-detail-modal');
    if (modal) modal.style.display = 'none';
    this.destroyCharts();
  },

  // --- Charts ---
  renderDetailCharts(epochs) {
    this.destroyCharts();

    const labels = epochs.map(e => e.epoch);

    // Loss chart
    const lossCtx = document.getElementById('chart-loss')?.getContext('2d');
    if (lossCtx) {
      this.charts.loss = new Chart(lossCtx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            { label: 'Train Box', data: epochs.map(e => e.train_box_loss), borderColor: '#3b82f6', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
            { label: 'Train Cls', data: epochs.map(e => e.train_cls_loss), borderColor: '#06b6d4', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
            { label: 'Val Box', data: epochs.map(e => e.val_box_loss), borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [4, 2] },
            { label: 'Val Cls', data: epochs.map(e => e.val_cls_loss), borderColor: '#ef4444', borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [4, 2] },
          ],
        },
        options: this.chartOptions('Loss'),
      });
    }

    // mAP chart
    const mapCtx = document.getElementById('chart-map')?.getContext('2d');
    if (mapCtx) {
      this.charts.map = new Chart(mapCtx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            { label: 'mAP@50', data: epochs.map(e => e.mAP50), borderColor: '#10b981', borderWidth: 2, pointRadius: 0, tension: 0.3, fill: { target: 'origin', above: 'rgba(16, 185, 129, 0.05)' } },
            { label: 'mAP@50:95', data: epochs.map(e => e.mAP50_95), borderColor: '#06b6d4', borderWidth: 2, pointRadius: 0, tension: 0.3 },
            { label: 'Precision', data: epochs.map(e => e.precision_b), borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [3, 2] },
            { label: 'Recall', data: epochs.map(e => e.recall_b), borderColor: '#ec4899', borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [3, 2] },
          ],
        },
        options: this.chartOptions('Metrics'),
      });
    }
  },

  chartOptions(title) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#9898ab', font: { size: 10, family: 'Inter' }, boxWidth: 12, padding: 12 },
        },
        tooltip: {
          backgroundColor: '#1a1a27',
          borderColor: '#2a2a3a',
          borderWidth: 1,
          titleColor: '#e8e8ef',
          bodyColor: '#9898ab',
          titleFont: { size: 11 },
          bodyFont: { size: 10 },
          padding: 8,
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(42, 42, 58, 0.5)' },
          ticks: { color: '#6b6b80', font: { size: 10 } },
          title: { display: true, text: 'Epoch', color: '#6b6b80', font: { size: 10 } },
        },
        y: {
          grid: { color: 'rgba(42, 42, 58, 0.5)' },
          ticks: { color: '#6b6b80', font: { size: 10 } },
          title: { display: true, text: title, color: '#6b6b80', font: { size: 10 } },
        },
      },
    };
  },

  destroyCharts() {
    Object.values(this.charts).forEach(c => c?.destroy());
    this.charts = {};
  },

  // --- Compare Runs ---
  async compareSelected() {
    if (this.selectedRuns.size < 2) return;

    try {
      const result = await API.mlops.compareRuns([...this.selectedRuns]);
      const modal = document.getElementById('compare-modal');
      if (!modal) return;

      const runs = result.runs || result;
      const tbody = document.getElementById('compare-tbody');

      const metrics = ['best_mAP50', 'best_mAP50_95', 'best_precision', 'best_recall', 'epochs_done', 'base_model', 'duration_sec'];
      const metricLabels = { best_mAP50: 'mAP@50', best_mAP50_95: 'mAP@50:95', best_precision: 'Precision', best_recall: 'Recall', epochs_done: 'Epochs', base_model: 'Base Model', duration_sec: 'Duration' };

      // Header
      document.getElementById('compare-thead').innerHTML = `
        <tr>
          <th>Metric</th>
          ${runs.map(r => `<th style="color: var(--accent-cyan);">${r.id.substring(0, 16)}</th>`).join('')}
        </tr>
      `;

      // Find best values for highlighting
      const bestValues = {};
      ['best_mAP50', 'best_mAP50_95', 'best_precision', 'best_recall'].forEach(m => {
        bestValues[m] = Math.max(...runs.map(r => r[m] ?? 0));
      });

      tbody.innerHTML = metrics.map(m => {
        const label = metricLabels[m];
        return `<tr>
          <td style="color: var(--text-secondary); font-weight: 500;">${label}</td>
          ${runs.map(r => {
            let val = r[m];
            let cls = '';
            if (['best_mAP50', 'best_mAP50_95', 'best_precision', 'best_recall'].includes(m)) {
              if (val != null) {
                cls = val === bestValues[m] ? 'better' : '';
                val = (val * 100).toFixed(1) + '%';
              } else {
                val = '—';
              }
            } else if (m === 'duration_sec') {
              val = val ? this.formatDuration(val) : '—';
            } else {
              val = val ?? '—';
            }
            return `<td class="${cls}">${val}</td>`;
          }).join('')}
        </tr>`;
      }).join('');

      modal.style.display = 'flex';
    } catch (err) {
      console.error('Compare error:', err);
    }
  },

  closeCompare() {
    const modal = document.getElementById('compare-modal');
    if (modal) modal.style.display = 'none';
  },

  // --- Model Registry ---
  renderRegistry(models) {
    const list = document.getElementById('registry-list');
    if (!list) return;

    if (!models || models.length === 0) {
      list.innerHTML = `<div class="dash-empty"><p>No model versions</p><p class="hint">Complete a training run and promote it</p></div>`;
      return;
    }

    list.innerHTML = models.map(m => {
      const isProd = m.stage === 'production';
      const mAP = m.mAP50_95 != null ? (m.mAP50_95 * 100).toFixed(1) + '%' : '—';

      return `
        <div class="registry-item ${isProd ? 'production' : ''}">
          <span class="registry-version">v${m.version}</span>
          <div class="registry-meta">
            <span class="registry-run-id">${m.run_id}</span>
            <span class="registry-metric">mAP@50:95 → ${mAP}</span>
          </div>
          <span class="stage-badge ${m.stage}">${m.stage}</span>
          <div class="registry-actions">
            ${m.stage !== 'production' ? `<button class="btn btn-xs btn-accent" onclick="Dashboard.setStage(${m.id}, 'production')">→ Prod</button>` : ''}
            ${m.stage !== 'archived' ? `<button class="btn btn-xs btn-ghost" onclick="Dashboard.setStage(${m.id}, 'archived')">Archive</button>` : ''}
          </div>
        </div>
      `;
    }).join('');
  },

  // --- Actions ---
  async promoteRun(runId) {
    try {
      await API.mlops.promoteRun(runId);
      this.closeRunDetail();
      this.load();
    } catch (err) {
      console.error('Promote error:', err);
    }
  },

  async setStage(versionId, stage) {
    try {
      await API.mlops.setStage(versionId, stage);
      this.load();
    } catch (err) {
      console.error('Stage transition error:', err);
    }
  },

  async deleteRun(runId) {
    if (!confirm(`Delete run ${runId}? This cannot be undone.`)) return;
    try {
      await API.mlops.deleteRun(runId);
      this.selectedRuns.delete(runId);
      this.load();
    } catch (err) {
      console.error('Delete error:', err);
    }
  },

  // --- Helpers ---
  formatDuration(sec) {
    if (sec < 60) return `${Math.round(sec)}s`;
    if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return `${h}h ${m}m`;
  },
};

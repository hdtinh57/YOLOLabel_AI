/**
 * Playground Controller - AI Engineer Pro Layout
 */
class PlaygroundManager {
    constructor() {
        this.view = document.getElementById('playground-view');
        
        // Settings form elements
        this.modelSelect = document.getElementById('pg-model-select');
        this.confInput = document.getElementById('pg-confidence');
        this.confVal = document.getElementById('pg-confidence-val');
        this.iouInput = document.getElementById('pg-iou');
        this.iouVal = document.getElementById('pg-iou-val');
        this.imgszSelect = document.getElementById('pg-imgsz');
        this.maxDetInput = document.getElementById('pg-max-det');
        this.agnosticNmsInput = document.getElementById('pg-agnostic-nms');
        
        // Actions
        this.btnReset = document.getElementById('pg-btn-reset');
        this.btnClear = document.getElementById('pg-btn-clear');
        this.btnPredict = document.getElementById('pg-btn-predict');
        
        // Workspace elements
        this.statusText = document.getElementById('pg-status-text');
        this.detCount = document.getElementById('pg-det-count');
        this.infTime = document.getElementById('pg-inf-time');
        
        this.uploadState = document.getElementById('pg-upload-state');
        this.fileInput = document.getElementById('pg-file-input');
        this.resultState = document.getElementById('pg-result-state');
        this.mediaContainer = document.getElementById('pg-media-container');
        this.loadingOverlay = document.getElementById('pg-loading-overlay');

        this.currentFile = null;

        this._bindEvents();
    }

    async init() {
        await this.loadModels();
    }

    _bindEvents() {
        // Range Sliders
        this.confInput.addEventListener('input', (e) => { this.confVal.textContent = e.target.value; });
        this.iouInput.addEventListener('input', (e) => { this.iouVal.textContent = e.target.value; });

        // Sidebar Actions
        this.btnReset.addEventListener('click', () => this.resetSettings());
        this.btnClear.addEventListener('click', () => this.clearWorkspace());
        this.btnPredict.addEventListener('click', () => this.runInference());

        // Upload Area Clicks & DND
        this.uploadState.addEventListener('click', () => this.fileInput.click());
        
        this.uploadState.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadState.classList.add('dragover');
        });
        
        this.uploadState.addEventListener('dragleave', () => {
            this.uploadState.classList.remove('dragover');
        });
        
        this.uploadState.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadState.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this._handleFile(e.dataTransfer.files[0]);
            }
        });

        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this._handleFile(e.target.files[0]);
            }
        });
    }

    resetSettings() {
        this.confInput.value = 0.25;
        this.confVal.textContent = "0.25";
        this.iouInput.value = 0.45;
        this.iouVal.textContent = "0.45";
        this.imgszSelect.value = "640";
        this.maxDetInput.value = 300;
        this.agnosticNmsInput.checked = false;
        app.toast('info', 'Settings reset to defaults');
    }

    async loadModels() {
        try {
            const models = await API.mlops.getModels();
            const stageOrder = { 'production': 0, 'staging': 1, 'none': 2, 'archived': 3 };
            models.sort((a, b) => stageOrder[a.stage] - stageOrder[b.stage]);

            if (models.length === 0) {
                this.modelSelect.innerHTML = `<option value="" disabled selected>No models available in Registry</option>`;
                return;
            }

            this.modelSelect.innerHTML = models.map(m => {
                const mapTxt = m.mAP50_95 ? `(mAP: ${(m.mAP50_95 * 100).toFixed(1)}%)` : '';
                const stageTxt = m.stage.toUpperCase();
                return `<option value="${m.id}">[${stageTxt}] v${m.version} ${mapTxt}</option>`;
            }).join('');
        } catch (e) {
            console.error("Failed to load models for playground", e);
            this.modelSelect.innerHTML = `<option value="" disabled selected>Failed to load models</option>`;
        }
    }

    async _handleFile(file) {
        const isImage = file.type.startsWith('image/');
        const isVideo = file.type.startsWith('video/');
        
        if (!isImage && !isVideo) {
            app.toast('error', 'Unsupported file type. Please upload Image or Video.');
            return;
        }

        this.currentFile = file;

        // Display preview
        this.uploadState.style.display = 'none';
        this.resultState.style.display = 'flex';
        this.btnClear.disabled = false;
        this.btnPredict.disabled = false;
        
        this.statusText.textContent = "Ready to predict";
        this.detCount.textContent = "-";
        this.infTime.textContent = "-";

        const objectUrl = URL.createObjectURL(file);
        if (isVideo) {
            this.mediaContainer.innerHTML = `<video src="${objectUrl}" controls loop></video>`;
        } else {
            this.mediaContainer.innerHTML = `<img src="${objectUrl}" alt="Preview" />`;
        }
    }

    async runInference() {
        if (!this.currentFile) return;

        if (!this.modelSelect.value) {
            app.toast('warning', 'Please select a model version first');
            return;
        }

        this.showLoading();
        this.btnPredict.disabled = true;
        this.btnClear.disabled = true;

        const formData = new FormData();
        formData.append('file', this.currentFile);
        formData.append('model_version_id', this.modelSelect.value);
        formData.append('confidence', this.confInput.value);
        formData.append('iou', this.iouInput.value);
        formData.append('imgsz', this.imgszSelect.value);
        formData.append('max_det', this.maxDetInput.value);
        formData.append('agnostic_nms', this.agnosticNmsInput.checked);

        try {
            const res = await fetch('/api/playground/predict', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Inference failed');
            }

            const data = await res.json();
            const isVideo = this.currentFile.type.startsWith('video/');
            this.renderResult(data.url, isVideo, data.total_detections, data.duration_ms);
            
        } catch (e) {
            app.toast('error', e.message);
        } finally {
            this.hideLoading();
            this.btnClear.disabled = false;
            this.btnPredict.disabled = false;
        }
    }

    renderResult(url, isVideo, detections, duration_ms) {
        this.uploadState.style.display = 'none';
        this.resultState.style.display = 'flex';
        this.btnClear.disabled = false;
        this.btnPredict.disabled = false;
        
        this.statusText.textContent = "Success";
        this.detCount.textContent = detections;
        this.infTime.textContent = `${(duration_ms/1000).toFixed(3)}s`;

        // Add timestamp to prevent browser caching the old file
        const noCacheUrl = `${url}?t=${Date.now()}`;
        
        if (isVideo) {
            this.mediaContainer.innerHTML = `<video src="${noCacheUrl}" controls autoplay loop></video>`;
        } else {
            this.mediaContainer.innerHTML = `<img src="${noCacheUrl}" alt="Inference Result" />`;
        }
    }

    clearWorkspace() {
        this.resultState.style.display = 'none';
        this.uploadState.style.display = 'flex';
        this.mediaContainer.innerHTML = '';
        this.fileInput.value = '';
        this.btnClear.disabled = true;
        this.btnPredict.disabled = true;
        this.currentFile = null;
        
        // Reset metrics
        this.statusText.textContent = "Ready";
        this.detCount.textContent = "-";
        this.infTime.textContent = "-";
    }

    showLoading() {
        this.uploadState.style.display = 'none';
        this.resultState.style.display = 'flex';
        this.loadingOverlay.style.display = 'flex';
        this.mediaContainer.innerHTML = '';
        
        this.statusText.textContent = "Processing...";
        this.detCount.textContent = "-";
        this.infTime.textContent = "-";
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    show() {
        this.view.style.display = 'grid'; // Note: grid for the 2-column layout
        this.init();
    }

    hide() {
        this.view.style.display = 'none';
    }
}

// Global instance
window.Playground = new PlaygroundManager();

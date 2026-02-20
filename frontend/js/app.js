/**
 * App Controller — orchestrates all modules
 */
class App {
    constructor() {
        this.canvas = new CanvasEngine('annotation-canvas', 'canvas-container');
        this.classManager = new ClassManager();
        this.gallery = new GalleryManager();
        this.shortcuts = new ShortcutManager(this);

        this.currentImage = null;
        this.currentSplit = 'train';
        this.isDirty = false;
        this.trainingPollInterval = null;

        this._init();
    }

    async _init() {
        // Wire up callbacks
        this.canvas.onBoxesChanged = (boxes) => this._onBoxesChanged(boxes);
        this.canvas.onSelectionChanged = (idx) => this._onSelectionChanged(idx);
        this.canvas.onMouseMove = (x, y) => {
            document.getElementById('status-pos').textContent = `x: ${x}, y: ${y}`;
        };

        this.classManager.onClassesChanged = (classes) => {
            this.canvas.classes = classes;
            this.canvas.render();
        };
        this.classManager.onActiveClassChanged = (classId) => {
            this.canvas.activeClassId = classId;
            this._updateStatusTool();
        };

        this.gallery.onImageSelected = (name, split) => this._loadImage(name, split);

        // Bind UI buttons
        this._bindUI();
        this._bindProjectUI();

        // Load initial data
        await this.classManager.load();
        this.canvas.classes = this.classManager.classes;
        this.canvas.activeClassId = this.classManager.activeClassId;

        await this._loadProjects();
        
        // Check for landing requirement
        if (!this.currentProject || this.currentProject === 'default') {
            document.getElementById('project-landing').style.display = 'flex';
            this._renderLanding();
            // Don't load other things if we are in landing mode?
            // Actually, we can load them but the overlay blocks.
            // But maybe better to stop?
            // User can't interact anyway.
        } else {
            await this.gallery.load(this.currentSplit);
            await this._loadALStats();
            await this._loadSettings();
            await this._loadALModels();
        }

        this.toast('info', 'YOLOLabel AI ready');
    }

    _renderLanding() {
        const list = document.getElementById('landing-project-list');
        list.innerHTML = this.projects
            .filter(p => p.name !== 'default')
            .map(p => `
                <button class="btn btn-ghost" style="width: 100%; text-align: left; padding: 12px; border: 1px solid #334155; margin-bottom: 8px; justify-content: space-between;" onclick="app.switchProject('${p.name}')">
                    <span style="font-weight: 500; font-size: 1rem;">${p.name}</span>
                    <span style="color: #64748b; font-size: 0.85rem;">Select &rarr;</span>
                </button>
            `).join('');
            
        // Bind create
        document.getElementById('landing-create-project').onclick = async () => {
            const name = document.getElementById('landing-new-project-name').value;
            if (name) {
                try {
                    await API.createProject(name);
                    await API.switchProject(name);
                    window.location.reload();
                } catch (e) {
                    alert(e.message);
                }
            }
        };
    }
    
    // Make switchProject public for onclick
    async switchProject(name) {
        try {
            await API.switchProject(name);
            window.location.reload();
        } catch (e) {
            console.error(e);
        }
    }

    _bindUI() {
        // Dashboard and Playground toggles
        const mlopsBtn = document.getElementById('btn-mlops-tab');
        const playgroundBtn = document.getElementById('btn-playground-tab');

        const hideAllViews = () => {
            const dashView = document.getElementById('dashboard-view');
            const pgView = document.getElementById('playground-view');
            const canvasArea = document.getElementById('canvas-area');
            const rightPanel = document.getElementById('right-panel');
            const galleryPanel = document.getElementById('gallery-panel');
            
            if (dashView) dashView.classList.remove('active');
            if (pgView) Playground.hide();

            // Hide primary labeling views if switching to full screen modules
            if (canvasArea) canvasArea.style.display = 'none';
            if (rightPanel) rightPanel.style.display = 'none';
            if (galleryPanel) galleryPanel.style.display = 'none';
            
            document.querySelectorAll('.split-btn').forEach(b => b.classList.remove('active'));
        };

        const showLabelingViews = () => {
             const dashView = document.getElementById('dashboard-view');
             if (dashView) dashView.classList.remove('active');
             Playground.hide();
             
             // Restore the main application container
             const appMain = document.querySelector('.app-main');
             if (appMain) appMain.style.display = 'flex';
             
             document.getElementById('canvas-area').style.display = 'flex';
             document.getElementById('right-panel').style.display = 'flex';
             document.getElementById('gallery-panel').style.display = 'flex';
             
             // Trigger resize on the canvas to prevent UI visual glitch
             if (this.canvas && typeof this.canvas._resizeCanvas === 'function') {
                 setTimeout(() => this.canvas._resizeCanvas(), 50);
             }
        };

        if (mlopsBtn) {
            mlopsBtn.addEventListener('click', () => {
                const dashView = document.getElementById('dashboard-view');
                const isActive = dashView.classList.contains('active');

                if (isActive) {
                    showLabelingViews();
                    Dashboard.hide();
                    const splitBtn = document.querySelector(`.split-btn[data-split="${this.currentSplit}"]`);
                    if (splitBtn) splitBtn.classList.add('active');
                } else {
                    hideAllViews();
                    mlopsBtn.classList.add('active');
                    Dashboard.show();
                }
            });

            Dashboard.init();
        }

        if (playgroundBtn) {
            playgroundBtn.addEventListener('click', () => {
                const pgView = document.getElementById('playground-view');
                const isActive = pgView.style.display === 'flex';

                if (isActive) {
                    showLabelingViews();
                    const splitBtn = document.querySelector(`.split-btn[data-split="${this.currentSplit}"]`);
                    if (splitBtn) splitBtn.classList.add('active');
                } else {
                    hideAllViews();
                    playgroundBtn.classList.add('active');
                    Playground.show();
                }
            });
        }

        document.querySelectorAll('.split-btn[data-split]').forEach(btn => {
            btn.addEventListener('click', () => {
                showLabelingViews();
                document.querySelectorAll('.split-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentSplit = btn.dataset.split;
                this.gallery.load(this.currentSplit);
            });
        });

        // Tool buttons
        document.querySelectorAll('[data-tool]').forEach(btn => {
            btn.addEventListener('click', () => this.setTool(btn.dataset.tool));
        });

        // Zoom
        document.getElementById('btn-zoom-in').addEventListener('click', () => this.canvas.zoomIn());
        document.getElementById('btn-zoom-out').addEventListener('click', () => this.canvas.zoomOut());
        document.getElementById('btn-zoom-fit').addEventListener('click', () => this.canvas.fitToView());

        // Navigation
        document.getElementById('btn-prev').addEventListener('click', () => this.prevImage());
        document.getElementById('btn-next').addEventListener('click', () => this.nextImage());

        // Delete
        document.getElementById('btn-delete-box').addEventListener('click', () => this.deleteSelectedBox());
        document.getElementById('btn-undo').addEventListener('click', () => this.canvas.undo());

        // Upload
        document.getElementById('btn-upload').addEventListener('click', () => {
            document.getElementById('file-input').click();
        });
        document.getElementById('file-input').addEventListener('change', (e) => this._handleUpload(e));

        // Export
        document.getElementById('btn-export').addEventListener('click', () => this._exportDataset());

        // Accept/Reject All
        document.getElementById('btn-accept-all').addEventListener('click', () => this.acceptAllPredictions());
        document.getElementById('btn-reject-all').addEventListener('click', () => this.rejectAllPredictions());

        // Active Learning
        document.getElementById('btn-auto-predict').addEventListener('click', () => this._autoPredict());
        document.getElementById('btn-train').addEventListener('click', () => this._startTraining());

        // Settings modal
        document.getElementById('btn-settings').addEventListener('click', () => this._openSettings());
        document.getElementById('settings-close').addEventListener('click', () => this._closeSettings());
        document.getElementById('settings-cancel').addEventListener('click', () => this._closeSettings());
        document.getElementById('settings-save').addEventListener('click', () => this._saveSettings());

        // Confidence slider display
        document.getElementById('setting-confidence').addEventListener('input', (e) => {
            document.getElementById('setting-confidence-val').textContent = e.target.value;
        });

        // Training modal dismiss
        document.getElementById('training-dismiss').addEventListener('click', () => {
            document.getElementById('training-modal').style.display = 'none';
        });

        // Stop Training
        document.getElementById('training-stop').addEventListener('click', async () => {
            try {
                const res = await API.stopTraining();
                if (res.status === 'stopping') {
                    this.toast('info', 'Stopping training... please wait.');
                    document.getElementById('training-stop').disabled = true;
                    document.getElementById('training-stop').textContent = 'Stopping...';
                }
            } catch (e) {
                this.toast('error', `Stop failed: ${e.message}`);
            }
        });
    }

    // --- Image Loading ---
    async _loadImage(name, split) {
        // Auto-save previous
        if (this.isDirty && this.currentImage) {
            await this.saveCurrentLabels();
        }

        this.currentImage = name;
        this.currentSplit = split;

        const url = API.getImageUrl(split, name);
        await this.canvas.loadImage(url, name);

        // Load labels
        try {
            const data = await API.getLabels(split, name);
            let boxes = data.boxes || [];

            // Also check cached predictions
            if (boxes.length === 0) {
                try {
                    const preds = await API.getCachedPredictions(name);
                    if (preds.boxes && preds.boxes.length > 0) {
                        boxes = preds.boxes;
                    }
                } catch (e) { /* no predictions */ }
            }

            this.canvas.setBoxes(boxes);
        } catch (e) {
            this.canvas.setBoxes([]);
        }

        this.isDirty = false;
        this._updateNavInfo();
        this._updateAnnotationList();
        document.getElementById('status-image').textContent = name;

        // Show accept/reject if has predictions
        const hasPredicted = this.canvas.boxes.some(b => b.is_predicted);
        document.getElementById('annotation-actions').style.display = hasPredicted ? 'flex' : 'none';
    }

    // --- Save ---
    async saveCurrentLabels() {
        if (!this.currentImage) return;
        const boxes = this.canvas.getBoxes().map(b => ({
            class_id: b.class_id,
            x_center: b.x_center,
            y_center: b.y_center,
            width: b.width,
            height: b.height,
        }));

        try {
            await API.saveLabels(this.currentImage, boxes, this.currentSplit);
            this.isDirty = false;
            this.gallery.updateImageStatus(
                this.currentImage,
                boxes.length > 0 ? 'labeled' : 'unlabeled',
                boxes.length
            );
            this.toast('success', `Saved ${boxes.length} annotations`);
            await this._loadALStats();
        } catch (e) {
            this.toast('error', `Save failed: ${e.message}`);
        }
    }

    // --- Navigation ---
    async nextImage() {
        if (this.isDirty) await this.saveCurrentLabels();
        this.gallery.selectNext();
    }

    async prevImage() {
        if (this.isDirty) await this.saveCurrentLabels();
        this.gallery.selectPrev();
    }

    // --- Tool ---
    setTool(tool) {
        this.canvas.setTool(tool);
        document.querySelectorAll('[data-tool]').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tool === tool);
        });
        this._updateStatusTool();
    }

    // --- Box Operations ---
    deleteSelectedBox() {
        this.canvas.deleteSelected();
    }

    acceptAllPredictions() {
        this.canvas.acceptAllPredictions();
        document.getElementById('annotation-actions').style.display = 'none';
        this.isDirty = true;
    }

    rejectAllPredictions() {
        this.canvas.rejectAllPredictions();
        document.getElementById('annotation-actions').style.display = 'none';
        this.isDirty = true;
    }

    // --- UI Updates ---
    _onBoxesChanged(boxes) {
        this.isDirty = true;
        this._updateAnnotationList();
        this.classManager.updateCounts(boxes);

        const hasPredicted = boxes.some(b => b.is_predicted);
        document.getElementById('annotation-actions').style.display = hasPredicted ? 'flex' : 'none';
    }

    _onSelectionChanged(index) {
        this._updateAnnotationList();
    }

    _updateAnnotationList() {
        const listEl = document.getElementById('annotation-list');
        const countEl = document.getElementById('annotation-count');
        const boxes = this.canvas.boxes;

        countEl.textContent = boxes.length;

        if (boxes.length === 0) {
            listEl.innerHTML = '<div class="annotation-empty">No annotations</div>';
            return;
        }

        listEl.innerHTML = boxes.map((box, idx) => {
            const cls = this.classManager.getClassInfo(box.class_id);
            const color = cls ? cls.color : '#999';
            const name = cls ? cls.name : `class_${box.class_id}`;
            const conf = box.confidence !== null && box.confidence !== undefined ?
                `${(box.confidence * 100).toFixed(0)}%` : '';
            const isActive = idx === this.canvas.selectedBoxIndex;
            const isPredicted = box.is_predicted;

            return `
                <div class="annotation-item ${isActive ? 'active' : ''} ${isPredicted ? 'predicted' : ''}"
                     data-box-index="${idx}">
                    <span class="annotation-class-color" style="background:${color}"></span>
                    <span class="annotation-label">${name}</span>
                    ${conf ? `<span class="annotation-conf">${conf}</span>` : ''}
                    <button class="class-delete" data-delete-box="${idx}" title="Delete">&times;</button>
                </div>`;
        }).join('');

        listEl.querySelectorAll('.annotation-item').forEach(el => {
            el.addEventListener('click', (e) => {
                if (e.target.closest('[data-delete-box]')) return;
                this.canvas.selectBox(parseInt(el.dataset.boxIndex));
            });
        });

        listEl.querySelectorAll('[data-delete-box]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.canvas.deleteBox(parseInt(btn.dataset.deleteBox));
            });
        });
    }

    _updateNavInfo() {
        document.getElementById('nav-info').textContent = this.gallery.getNavInfo();
    }

    _updateStatusTool() {
        const toolNames = { draw: 'Draw', select: 'Select', pan: 'Pan' };
        const cls = this.classManager.getClassInfo(this.canvas.activeClassId);
        const clsName = cls ? cls.name : '';
        document.getElementById('status-tool').textContent =
            `Tool: ${toolNames[this.canvas.tool] || 'Draw'}${clsName ? ` · ${clsName}` : ''}`;
    }

    // --- Upload ---
    async _handleUpload(e) {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        try {
            const result = await API.uploadImages(files, this.currentSplit);
            this.toast('success', `Uploaded ${result.count} images`);
            await this.gallery.load(this.currentSplit);
        } catch (err) {
            this.toast('error', `Upload failed: ${err.message}`);
        }
        e.target.value = '';
    }

    // --- Export ---
    async _exportDataset() {
        try {
            // Save current first
            if (this.isDirty) await this.saveCurrentLabels();

            const result = await API.exportDataset();
            this.toast('success',
                `Exported! ${result.total_images} images, ${result.total_labels} labels, ${result.classes.length} classes`);
        } catch (e) {
            this.toast('error', `Export failed: ${e.message}`);
        }
    }

    // --- Active Learning ---
    async _autoPredict() {
        try {
            if (!this.bestModel) {
                this.toast('warning', 'No model available for auto-prediction.');
                return;
            }
            const modelPath = this.bestModel.path || this.bestModel.id; // handle varying API specs 
            this.toast('info', `Running auto-prediction with ${this.bestModel.name || this.bestModel.version}...`);
            
            const result = await API.predictUnlabeled(this.currentSplit, null, modelPath);
            this.toast('success', `Predicted ${result.count} images`);
            await this.gallery.load(this.currentSplit);

            // If current image has no labels, reload to show predictions
            if (this.currentImage) {
                const preds = await API.getCachedPredictions(this.currentImage);
                if (preds.boxes && preds.boxes.length > 0 && this.canvas.boxes.length === 0) {
                    this.canvas.setBoxes(preds.boxes);
                    document.getElementById('annotation-actions').style.display = 'flex';
                }
            }
        } catch (e) {
            this.toast('error', `Prediction failed: ${e.message}`);
        }

        await this._loadALStats();
    }

    async _loadALModels() {
        try {
            const models = await API.getAvailableModels();
            if (!models || models.length === 0) {
                document.getElementById('stat-model').textContent = 'No models found';
                const btnAutoPredict = document.getElementById('btn-auto-predict');
                if (btnAutoPredict) btnAutoPredict.disabled = true;
                return;
            }

            // Find best model: 1. production stage -> 2. highest mAP
            let bestModel = models.find(m => (m.stage || '').toLowerCase() === 'production');
            
            if (!bestModel) {
                bestModel = models.reduce((best, current) => {
                    const currentMap = current.metrics?.['mAP50-95'] || current.metrics?.['mAP_50_95'] || current.metrics?.mAP50 || 0;
                    const bestMap = best.metrics?.['mAP50-95'] || best.metrics?.['mAP_50_95'] || best.metrics?.mAP50 || 0;
                    return currentMap > bestMap ? current : best;
                }, models[0]);
            }

            this.bestModel = bestModel;

            const modelLabel = document.getElementById('stat-model');
            if (modelLabel) {
                const name = bestModel.name || bestModel.version || 'Loaded';
                const stage = bestModel.stage ? `[${bestModel.stage.toUpperCase()}] ` : '';
                modelLabel.textContent = `${stage}${name}`;
                modelLabel.title = `Path: ${bestModel.path}`;
            }

            const btnAutoPredict = document.getElementById('btn-auto-predict');
            if (btnAutoPredict) {
                btnAutoPredict.disabled = false;
            }
            
            const indicator = document.getElementById('al-indicator');
            if (indicator) {
                indicator.className = 'al-indicator loaded';
            }
        } catch (e) {
            console.error('Failed to load AL models:', e);
            document.getElementById('stat-model').textContent = 'Error loading';
        }
    }

    async _startTraining() {
        const config = {
            model_name: document.getElementById('setting-model')?.value || 'yolo26n.pt',
            epochs: parseInt(document.getElementById('setting-epochs')?.value || '50'),
            imgsz: parseInt(document.getElementById('setting-imgsz')?.value || '640'),
            batch: parseInt(document.getElementById('setting-batch')?.value || '16'),
            patience: 10,
            
            // Advanced
            optimizer: document.getElementById('setting-optimizer').value,
            device: document.getElementById('setting-device').value,
            lr0: parseFloat(document.getElementById('setting-lr0').value),
            momentum: parseFloat(document.getElementById('setting-momentum').value),
            mosaic: parseFloat(document.getElementById('setting-mosaic').value),
            mixup: parseFloat(document.getElementById('setting-mixup').value),
        };

        try {
            // Ensure labels are saved & exported
            if (this.isDirty) await this.saveCurrentLabels();
            await API.exportDataset();

            const result = await API.startTraining(config);
            if (result.error) {
                this.toast('error', result.error);
                return;
            }

            this.toast('info', 'Training started!');
            this._showTrainingModal();
            this._pollTrainingStatus();

            // Update AI indicator
            const indicator = document.getElementById('al-indicator');
            indicator.className = 'al-indicator training';
        } catch (e) {
            this.toast('error', `Training failed: ${e.message}`);
        }
    }

    async _loadALStats() {
        try {
            const stats = await API.getALStats(this.currentSplit); // { labeled: N, unlabeled: N, predicted: N, total: N }
            
            const container = document.getElementById('gallery-stats');
            if (container) {
                container.querySelector('.stat-badge.labeled').textContent = stats.labeled || 0;
                container.querySelector('.stat-badge.unlabeled').textContent = stats.unlabeled || 0;
                container.querySelector('.stat-badge.predicted').textContent = stats.predicted || 0;
            }
            
            // Update Active Learning Progress Container
            const trainThreshold = parseInt(document.getElementById('setting-threshold')?.value || '10');
            const progressFill = document.querySelector('#label-progress .progress-fill');
            const progressText = document.getElementById('stat-progress-text');
            
            if (progressFill && progressText) {
                const labeledCount = stats.labeled || 0;
                let percentage = (labeledCount / trainThreshold) * 100;
                if (percentage > 100) percentage = 100;
                
                progressFill.style.width = `${percentage}%`;
                progressText.textContent = `${labeledCount} / ${trainThreshold}`;
                
                if (percentage >= 100) {
                    progressFill.style.background = 'var(--success)';
                } else {
                    progressFill.style.background = 'var(--accent-gradient)';
                }
            }
        } catch (e) {
            console.warn('Failed to load AL stats:', e);
        }
    }

    _showTrainingModal() {
        document.getElementById('training-modal').style.display = 'flex';
        
        // Reset UI state
        const spinner = document.querySelector('.training-spinner');
        if (spinner) spinner.style.display = 'block';
        const icon = document.getElementById('training-complete-icon');
        if (icon) icon.remove();
        document.getElementById('training-status-text').classList.remove('text-success');

        document.getElementById('training-progress-fill').style.width = '0%';
        document.getElementById('training-progress-fill').style.width = '0%';
        document.getElementById('training-progress-label').textContent = '0 / 50 epochs';
        document.getElementById('training-status-text').textContent = 'Starting...';
        document.getElementById('training-metrics').innerHTML = '';
        
        const stopBtn = document.getElementById('training-stop');
        stopBtn.disabled = false;
        stopBtn.textContent = 'Stop Training';
    }

    async _pollTrainingStatus() {
        if (this.trainingPollInterval) clearInterval(this.trainingPollInterval);
        
        this.trainingPollInterval = setInterval(async () => {
            try {
                const status = await API.getTrainingStatus();
                
                const current = status.current_epoch || 0;
                const total = status.total_epochs || 1;
                const percent = Math.round((current / total) * 100);
                
                document.getElementById('training-progress-fill').style.width = `${percent}%`;
                document.getElementById('training-progress-label').textContent = `${current} / ${total} epochs (${percent}%)`;
                
                // Check for terminal states
                if (['completed', 'failed', 'stopped'].includes(status.status)) {
                     const isEarlyStop = status.status === 'completed' && current < total;
                     let text = '';
                     let iconColor = '';
                     let iconSVG = '';

                     if (status.status === 'completed') {
                         text = isEarlyStop ? `Early Stopping at Epoch ${current}!` : 'Training Completed!';
                         iconColor = '#10b981'; // green
                         iconSVG = `
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="${iconColor}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                                <polyline points="22 4 12 14.01 9 11.01"></polyline>
                            </svg>`;
                     } else if (status.status === 'failed') {
                         text = `Error: ${status.error || 'Unknown error'}`;
                         iconColor = '#ef4444'; // red
                         iconSVG = `
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="${iconColor}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>`;
                     } else {
                         text = 'Training Stopped.';
                         iconColor = '#f59e0b'; // amber
                         iconSVG = `
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="${iconColor}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            </svg>`;
                     }
                     
                     const statusText = document.getElementById('training-status-text');
                     statusText.textContent = text;
                     statusText.style.color = iconColor;
                     
                     // Replace spinner with icon
                     const spinner = document.querySelector('.training-spinner');
                     if (spinner) {
                        spinner.style.display = 'none';
                        if (!document.getElementById('training-complete-icon')) {
                            const icon = document.createElement('div');
                            icon.id = 'training-complete-icon';
                            icon.innerHTML = iconSVG;
                            icon.style.margin = '0 auto 16px auto';
                            spinner.parentNode.insertBefore(icon, spinner);
                        }
                     }

                     clearInterval(this.trainingPollInterval);
                     
                     if (status.status === 'completed') {
                        this.toast('success', text);
                        await this._loadSettings(); // Reload settings/models
                     } else if (status.status === 'failed') {
                        this.toast('error', 'Training failed');
                     } else {
                        this.toast('warning', 'Training stopped');
                     }
                     
                     // Enable dismiss button
                     document.getElementById('training-stop').style.display = 'none';
                     document.getElementById('training-dismiss').style.display = 'inline-flex';
                     
                     // Update indicator
                     document.getElementById('al-indicator').className = 'al-indicator';
                     
                     // Reload stats
                     this._loadALStats();
                     return; // exit loop
                }
                
                // Running state update
                document.getElementById('training-status-text').textContent = status.is_training ? 'Training...' : 'Idle';
            } catch (e) {
                console.error('Poll failed', e);
            }
        }, 1000);
    }


    // --- Settings ---
    async _loadSettings() {
        try {
            const settings = await API.getSettings();
            document.getElementById('setting-threshold').value = settings.auto_train_threshold;
            document.getElementById('setting-confidence').value = settings.prediction_confidence;
            document.getElementById('setting-confidence-val').textContent = settings.prediction_confidence;
            
            // Basic Training Defaults
            if (settings.epochs) document.getElementById('setting-epochs').value = settings.epochs;
            if (settings.batch) document.getElementById('setting-batch').value = settings.batch;
            if (settings.imgsz) document.getElementById('setting-imgsz').value = settings.imgsz;

            // Advanced defaults
            if (settings.optimizer) document.getElementById('setting-optimizer').value = settings.optimizer;
            if (settings.device) document.getElementById('setting-device').value = settings.device;
            if (settings.lr0 !== undefined) document.getElementById('setting-lr0').value = settings.lr0;
            if (settings.momentum !== undefined) document.getElementById('setting-momentum').value = settings.momentum;
            if (settings.mosaic !== undefined) document.getElementById('setting-mosaic').value = settings.mosaic;
            if (settings.mixup !== undefined) document.getElementById('setting-mixup').value = settings.mixup;

            // Populate model dropdown
            const modelSelect = document.getElementById('setting-model');
            modelSelect.innerHTML = settings.available_models.map(m =>
                `<option value="${m}" ${m === settings.base_model ? 'selected' : ''}>${m}</option>`
            ).join('');
        } catch (e) {
            // Settings not critical
            console.error(e);
        }
    }

    _openSettings() {
        document.getElementById('settings-modal').style.display = 'flex';
    }

    _closeSettings() {
        document.getElementById('settings-modal').style.display = 'none';
    }

    async _saveSettings() {
        try {
            await API.updateSettings({
                auto_train_threshold: parseInt(document.getElementById('setting-threshold').value),
                prediction_confidence: parseFloat(document.getElementById('setting-confidence').value),
                prediction_confidence: parseFloat(document.getElementById('setting-confidence').value),
                base_model: document.getElementById('setting-model').value,
                
                // Basic Training
                epochs: parseInt(document.getElementById('setting-epochs').value),
                batch: parseInt(document.getElementById('setting-batch').value),
                imgsz: parseInt(document.getElementById('setting-imgsz').value),
                
                // Advanced
                optimizer: document.getElementById('setting-optimizer').value,
                device: document.getElementById('setting-device').value,
                lr0: parseFloat(document.getElementById('setting-lr0').value),
                momentum: parseFloat(document.getElementById('setting-momentum').value),
                mosaic: parseFloat(document.getElementById('setting-mosaic').value),
                mixup: parseFloat(document.getElementById('setting-mixup').value),
            });
            this._closeSettings();
            this.toast('success', 'Settings saved');
            await this._loadALStats();
        } catch (e) {
            this.toast('error', `Settings save failed: ${e.message}`);
        }
    }

    // --- Projects ---
    async _loadProjects() {
        try {
            this.projects = await API.getProjects(); // Store for landing
            const activeData = await API.getActiveProject();
            const active = activeData.active_project;

            const select = document.getElementById('project-select');
            select.innerHTML = this.projects.map(p => 
                `<option value="${p.name}" ${p.name === active ? 'selected' : ''}>
                    ${p.name} (${p.image_count} imgs)
                </option>`
            ).join('');
            
            this.currentProject = active;
        } catch (e) {
            console.error('Failed to load projects', e);
        }
    }

    _bindProjectUI() {
        // Switch Project
        document.getElementById('project-select').addEventListener('change', async (e) => {
            try {
                await API.switchProject(e.target.value);
                window.location.reload();
            } catch (err) {
                this.toast('error', `Switch failed: ${err.message}`);
            }
        });

        // New Project
        const modal = document.getElementById('new-project-modal');
        const input = document.getElementById('new-project-name');

        document.getElementById('btn-new-project').addEventListener('click', () => {
             modal.style.display = 'flex';
             input.value = '';
             input.focus();
        });

        document.getElementById('new-project-close').addEventListener('click', () => modal.style.display = 'none');
        document.getElementById('new-project-cancel').addEventListener('click', () => modal.style.display = 'none');

        document.getElementById('new-project-create').addEventListener('click', async () => {
            const name = input.value.trim();
            if (!name) return;
            
            try {
                await API.createProject(name);
                await API.switchProject(name);
                window.location.reload();
            } catch (err) {
                this.toast('error', `Create failed: ${err.message}`);
            }
        });

        // Delete Project
        document.getElementById('btn-delete-project').addEventListener('click', async () => {
            const select = document.getElementById('project-select');
            const name = select.value;
            if (name === 'default') return this.toast('warning', 'Cannot delete default project');
            
            if (!confirm(`Are you sure you want to delete project "${name}"? This cannot be undone.`)) return;

            try {
                await API.deleteProject(name);
                // Switch to default if we deleted the active one (though backend prevents deleting active)
                // Backend raises error if deleting active. 
                // So user must switch first.
                // But UI shows active project in selector. 
                // So this button deletes the *currently selected* active project?
                // Logic: You usually delete *other* projects.
                // But here the selector shows the *active* project.
                // So if I click delete, I am trying to delete the active project.
                // Backend forbids this.
                // I should probably allow deleting *other* projects from a list, but for now simple UI:
                // "Cannot delete active project. Switch to another project first."
                this.toast('error', 'Cannot delete active project. Switch to another project first.');
            } catch (err) {
                this.toast('error', `Delete failed: ${err.message}`);
            }
        });

        // Close Project (Return to Landing)
        document.getElementById('btn-close-project').addEventListener('click', async () => {
            if (confirm('Close current project and return to project selection?')) {
                try {
                    await API.switchProject('default');
                    window.location.reload();
                } catch (e) {
                    this.toast('error', `Close failed: ${e.message}`);
                }
            }
        });

    }

    // --- Toast ---
    toast(type, message, duration = 3000) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = message;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
}

// --- Bootstrap ---
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});

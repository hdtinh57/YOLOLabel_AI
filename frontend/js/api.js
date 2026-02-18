/**
 * API Client — communicates with FastAPI backend
 */
const API = {
    BASE: '/api',

    async request(path, options = {}) {
        const url = `${this.BASE}${path}`;
        try {
            const res = await fetch(url, {
                headers: { 'Content-Type': 'application/json', ...options.headers },
                ...options,
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            return await res.json();
        } catch (err) {
            console.error(`API Error [${path}]:`, err);
            throw err;
        }
    },

    // --- Projects ---
    async getProjects() {
        return this.request('/projects');
    },

    async createProject(name) {
        return this.request('/projects', {
            method: 'POST',
            body: JSON.stringify({ name }),
        });
    },

    async switchProject(name) {
        return this.request(`/projects/${name}/activate`, { method: 'POST' });
    },

    async deleteProject(name) {
        return this.request(`/projects/${name}`, { method: 'DELETE' });
    },

    async getActiveProject() {
        return this.request('/projects/active');
    },

    // --- Images ---
    async getImages(split = 'train') {
        return this.request(`/images?split=${split}`);
    },

    getImageUrl(split, name) {
        return `${this.BASE}/images/${split}/${encodeURIComponent(name)}`;
    },

    getThumbnailUrl(split, name) {
        return `${this.BASE}/images/thumbnail/${split}/${encodeURIComponent(name)}`;
    },

    async uploadImages(files, split = 'train') {
        const formData = new FormData();
        for (const f of files) formData.append('files', f);
        const res = await fetch(`${this.BASE}/images/upload?split=${split}`, {
            method: 'POST',
            body: formData,
        });
        return res.json();
    },

    async getDimensions(split, name) {
        return this.request(`/images/dimensions/${split}/${encodeURIComponent(name)}`);
    },

    // --- Labels ---
    async getLabels(split, imageName) {
        return this.request(`/labels/${split}/${encodeURIComponent(imageName)}`);
    },

    async saveLabels(imageName, boxes, split = 'train') {
        return this.request('/labels/save', {
            method: 'POST',
            body: JSON.stringify({ image_name: imageName, boxes, split }),
        });
    },

    async deleteLabels(split, imageName) {
        return this.request(`/labels/${split}/${encodeURIComponent(imageName)}`, { method: 'DELETE' });
    },

    // --- Classes ---
    async getClasses() {
        return this.request('/labels/classes/list');
    },

    async addClass(name) {
        return this.request(`/labels/classes/add?name=${encodeURIComponent(name)}`, { method: 'POST' });
    },

    async removeClass(classId) {
        return this.request(`/labels/classes/${classId}`, { method: 'DELETE' });
    },

    async renameClass(classId, newName) {
        return this.request(`/labels/classes/${classId}?new_name=${encodeURIComponent(newName)}`, { method: 'PUT' });
    },

    async setClasses(classes) {
        return this.request('/labels/classes/set', {
            method: 'POST',
            body: JSON.stringify(classes),
        });
    },

    // --- Export ---
    async exportDataset() {
        return this.request('/labels/export');
    },

    // --- Model ---
    async getModelInfo() {
        return this.request('/model/info');
    },

    async getAvailableModels() {
        return this.request('/model/available');
    },

    async loadModel(modelName, customPath) {
        const params = new URLSearchParams();
        if (modelName) params.set('model_name', modelName);
        if (customPath) params.set('custom_path', customPath);
        return this.request(`/model/load?${params}`, { method: 'POST' });
    },

    async startTraining(config) {
        return this.request('/model/train', {
            method: 'POST',
            body: JSON.stringify(config),
        });
    },

    async stopTraining() {
        return this.request('/model/stop-training', { method: 'POST' });
    },

    async getTrainingStatus() {
        return this.request('/model/training-status');
    },

    async getSettings() {
        return this.request('/settings');
    },

    async updateSettings(settings) {
        return this.request('/settings', {
            method: 'POST',
            body: JSON.stringify(settings),
        });
    },

    // --- Predict ---
    async predictUnlabeled(split = 'train', confidence = null, modelPath = null) {
        const params = new URLSearchParams({ split });
        if (confidence !== null) params.set('confidence', confidence);
        if (modelPath) params.set('model_path', modelPath);
        return this.request(`/predict/unlabeled?${params}`, { method: 'POST' });
    },

    async getSmartQueue(split = 'train') {
        return this.request(`/predict/queue?split=${split}`);
    },

    async getCachedPredictions(imageName) {
        return this.request(`/predict/cached/${encodeURIComponent(imageName)}`);
    },

    async getALStats(split = 'train') {
        return this.request(`/predict/stats?split=${split}`);
    },

    async shouldTrain(split = 'train') {
        return this.request(`/predict/should-train?split=${split}`);
    },

    // --- MLOps ---
    mlops: {
        async getDashboardStats() {
            return API.request('/mlops/dashboard');
        },

        async getRuns(page = 1, limit = 50) {
            return API.request(`/mlops/runs?page=${page}&limit=${limit}`);
        },

        async getRunDetail(runId) {
            return API.request(`/mlops/runs/${encodeURIComponent(runId)}`);
        },

        async compareRuns(runIds) {
            return API.request('/mlops/runs/compare', {
                method: 'POST',
                body: JSON.stringify({ run_ids: runIds }),
            });
        },

        async deleteRun(runId) {
            return API.request(`/mlops/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
        },

        async getModels() {
            return API.request('/mlops/models');
        },

        async promoteRun(runId) {
            return API.request(`/mlops/models/promote/${encodeURIComponent(runId)}`, { method: 'POST' });
        },

        async setStage(versionId, stage) {
            return API.request(`/mlops/models/${versionId}/stage?stage=${encodeURIComponent(stage)}`, {
                method: 'PUT',
            });
        },

        async getALCycles() {
            return API.request('/mlops/al/cycles');
        },

        async getConfig() {
            return API.request('/mlops/config');
        },

        async updateConfig(config) {
            return API.request('/mlops/config', {
                method: 'PUT',
                body: JSON.stringify(config),
            });
        },
    },
};

/**
 * Gallery Manager — image list, filtering, smart queue
 */
class GalleryManager {
    constructor() {
        this.images = [];
        this.filteredImages = [];
        this.currentIndex = -1;
        this.currentSplit = 'train';
        this.filter = 'all';

        this.listEl = document.getElementById('gallery-list');
        this.searchEl = document.getElementById('gallery-search');
        this.filterEl = document.getElementById('gallery-filter');

        this.onImageSelected = null;

        this._bindEvents();
    }

    _bindEvents() {
        this.searchEl.addEventListener('input', () => this._applyFilter());
        this.filterEl.addEventListener('change', () => {
            this.filter = this.filterEl.value;
            this._applyFilter();
        });
    }

    async load(split = 'train') {
        this.currentSplit = split;
        try {
            if (this.filter === 'smart') {
                this.images = await API.getSmartQueue(split);
            } else {
                this.images = await API.getImages(split);
            }
            this._applyFilter();
            this._updateStats();
        } catch (e) {
            console.warn('Failed to load images:', e);
            this.images = [];
            this._applyFilter();
        }
    }

    _applyFilter() {
        const search = this.searchEl.value.toLowerCase();
        let imgs = this.images;

        // Status filter
        if (this.filter !== 'all' && this.filter !== 'smart') {
            imgs = imgs.filter(i => i.status === this.filter);
        }

        // Search
        if (search) {
            imgs = imgs.filter(i => i.name.toLowerCase().includes(search));
        }

        this.filteredImages = imgs;
        this.renderList();
    }

    renderList() {
        if (this.filteredImages.length === 0) {
            this.listEl.innerHTML = `
                <div class="gallery-empty">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3">
                        <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
                    </svg>
                    <p>No images found</p>
                    <p class="hint">Upload images or place them in<br><code>data/images/${this.currentSplit}/</code></p>
                </div>`;
            return;
        }

        this.listEl.innerHTML = this.filteredImages.map((img, idx) => {
            const statusIcon = img.status === 'labeled' ? '✅' :
                               img.status === 'predicted' ? '🤖' : '❓';
            const isActive = idx === this.currentIndex;
            const thumbUrl = API.getThumbnailUrl(this.currentSplit, img.name);

            let meta = img.status;
            if (img.annotation_count > 0) meta += ` · ${img.annotation_count} boxes`;
            if (img.uncertainty_score !== undefined && img.uncertainty_score < 1.0 && img.status === 'predicted') {
                meta += ` · ${(img.uncertainty_score * 100).toFixed(0)}% unc.`;
            }

            return `
                <div class="gallery-item ${isActive ? 'active' : ''}" data-index="${idx}" data-name="${img.name}">
                    <img class="gallery-thumb" src="${thumbUrl}" alt="" loading="lazy"
                         onerror="this.style.display='none'">
                    <div class="gallery-info">
                        <div class="gallery-name">${img.name}</div>
                        <div class="gallery-meta">${meta}</div>
                    </div>
                    <span class="gallery-status">${statusIcon}</span>
                </div>`;
        }).join('');

        // Click handlers
        this.listEl.querySelectorAll('.gallery-item').forEach(el => {
            el.addEventListener('click', () => {
                const idx = parseInt(el.dataset.index);
                this.selectImage(idx);
            });
        });
    }

    selectImage(index) {
        if (index < 0 || index >= this.filteredImages.length) return;
        this.currentIndex = index;
        this.renderList();

        // Scroll active into view
        const activeEl = this.listEl.querySelector('.gallery-item.active');
        if (activeEl) activeEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' });

        const img = this.filteredImages[index];
        if (this.onImageSelected) {
            this.onImageSelected(img.name, this.currentSplit);
        }
    }

    selectNext() {
        if (this.filteredImages.length === 0) return;
        const next = (this.currentIndex + 1) % this.filteredImages.length;
        this.selectImage(next);
    }

    selectPrev() {
        if (this.filteredImages.length === 0) return;
        const prev = this.currentIndex <= 0 ? this.filteredImages.length - 1 : this.currentIndex - 1;
        this.selectImage(prev);
    }

    getCurrentImage() {
        if (this.currentIndex < 0 || this.currentIndex >= this.filteredImages.length) return null;
        return this.filteredImages[this.currentIndex];
    }

    getNavInfo() {
        return `${this.currentIndex + 1} / ${this.filteredImages.length}`;
    }

    updateImageStatus(imageName, status, annotationCount) {
        const img = this.images.find(i => i.name === imageName);
        if (img) {
            img.status = status;
            img.annotation_count = annotationCount;
        }
        this._updateStats();
        this.renderList();
    }

    _updateStats() {
        const labeled = this.images.filter(i => i.status === 'labeled').length;
        const predicted = this.images.filter(i => i.status === 'predicted').length;
        const unlabeled = this.images.filter(i => i.status === 'unlabeled').length;

        const statsEl = document.getElementById('gallery-stats');
        if (statsEl) {
            statsEl.innerHTML = `
                <span class="stat-badge labeled">${labeled}</span>
                <span class="stat-badge predicted">${predicted}</span>
                <span class="stat-badge unlabeled">${unlabeled}</span>
            `;
        }
    }
}

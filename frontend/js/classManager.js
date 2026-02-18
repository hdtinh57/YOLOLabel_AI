/**
 * Class Manager — handles class CRUD in the UI
 */
class ClassManager {
    constructor() {
        this.classes = [];
        this.activeClassId = 0;
        this.listEl = document.getElementById('class-list');
        this.addBtn = document.getElementById('btn-add-class');

        this.onClassesChanged = null;
        this.onActiveClassChanged = null;

        this.addBtn.addEventListener('click', () => this.showAddForm());
    }

    async load() {
        try {
            const data = await API.getClasses();
            this.classes = Array.isArray(data) ? data : [];
            if (this.classes.length > 0) this.activeClassId = this.classes[0].id;
            this.renderList();
        } catch (e) {
            console.warn('Failed to load classes:', e);
        }
    }

    renderList() {
        if (this.classes.length === 0) {
            this.listEl.innerHTML = '<div class="class-empty">No classes defined</div>';
            return;
        }

        this.listEl.innerHTML = this.classes.map((cls, idx) => `
            <div class="class-item ${cls.id === this.activeClassId ? 'active' : ''}"
                 data-class-id="${cls.id}" title="Press ${idx + 1} to select">
                <span class="class-color" style="background:${cls.color}"></span>
                <span class="class-name">${cls.name}</span>
                <span class="class-hotkey">${idx < 9 ? idx + 1 : ''}</span>
                <span class="class-count" data-class-count="${cls.id}">0</span>
                <button class="class-delete" data-delete-id="${cls.id}" title="Remove">&times;</button>
            </div>
        `).join('');

        // Click handlers
        this.listEl.querySelectorAll('.class-item').forEach(el => {
            el.addEventListener('click', (e) => {
                if (e.target.closest('.class-delete')) return;
                this.setActive(parseInt(el.dataset.classId));
            });
        });

        this.listEl.querySelectorAll('.class-delete').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.removeClass(parseInt(btn.dataset.deleteId));
            });
        });
    }

    setActive(classId) {
        this.activeClassId = classId;
        this.renderList();
        if (this.onActiveClassChanged) this.onActiveClassChanged(classId);
    }

    selectByIndex(index) {
        if (index >= 0 && index < this.classes.length) {
            this.setActive(this.classes[index].id);
        }
    }

    async addClass(name) {
        try {
            const cls = await API.addClass(name);
            this.classes.push(cls);
            this.setActive(cls.id);
            if (this.onClassesChanged) this.onClassesChanged(this.classes);
            return cls;
        } catch (e) {
            console.error('Failed to add class:', e);
        }
    }

    async removeClass(classId) {
        try {
            await API.removeClass(classId);
            this.classes = this.classes.filter(c => c.id !== classId);
            // Re-index
            this.classes.forEach((c, i) => c.id = i);
            if (this.activeClassId === classId) {
                this.activeClassId = this.classes.length > 0 ? this.classes[0].id : 0;
            }
            this.renderList();
            if (this.onClassesChanged) this.onClassesChanged(this.classes);
        } catch (e) {
            console.error('Failed to remove class:', e);
        }
    }

    showAddForm() {
        // Remove existing form if any
        const existing = this.listEl.querySelector('.class-add-form');
        if (existing) { existing.remove(); return; }

        const form = document.createElement('div');
        form.className = 'class-add-form';
        form.innerHTML = `
            <input type="text" placeholder="Class name..." autofocus>
            <button class="btn btn-xs btn-accent">Add</button>
        `;

        const input = form.querySelector('input');
        const addBtn = form.querySelector('button');

        const doAdd = () => {
            const name = input.value.trim();
            if (name) {
                this.addClass(name);
                form.remove();
            }
        };

        addBtn.addEventListener('click', doAdd);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') doAdd();
            if (e.key === 'Escape') form.remove();
            e.stopPropagation(); // Prevent shortcuts
        });

        this.listEl.appendChild(form);
        input.focus();
    }

    updateCounts(boxes) {
        const counts = {};
        this.classes.forEach(c => counts[c.id] = 0);
        boxes.forEach(b => {
            if (counts[b.class_id] !== undefined) counts[b.class_id]++;
        });

        this.listEl.querySelectorAll('[data-class-count]').forEach(el => {
            const id = parseInt(el.dataset.classCount);
            el.textContent = counts[id] || 0;
        });
    }

    getClassInfo(classId) {
        return this.classes.find(c => c.id === classId);
    }
}

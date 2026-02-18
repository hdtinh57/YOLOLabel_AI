/**
 * Keyboard Shortcuts Manager
 */
class ShortcutManager {
    constructor(app) {
        this.app = app;
        this._bind();
    }

    _bind() {
        document.addEventListener('keydown', (e) => {
            // Don't capture when typing in input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
                return;
            }

            this._handleKey(e);
        });
    }

    _handleKey(e) {
        const key = e.key.toLowerCase();
        const ctrl = e.ctrlKey || e.metaKey;

        // --- Navigation ---
        if (key === 'arrowright' || key === 'd' && !ctrl) {
            if (key === 'arrowright') { e.preventDefault(); this.app.nextImage(); }
        }
        if (key === 'arrowleft' || key === 'a' && !ctrl) {
            if (key === 'arrowleft') { e.preventDefault(); this.app.prevImage(); }
        }

        // --- Tools ---
        if (key === 'b') { this.app.setTool('draw'); }
        if (key === 'v') { this.app.setTool('select'); }
        if (key === 'h') { this.app.setTool('pan'); }

        // --- Box operations ---
        if (key === 'delete' || key === 'backspace') {
            e.preventDefault();
            this.app.deleteSelectedBox();
        }
        if (key === 'd' && !ctrl) {
            this.app.deleteSelectedBox();
        }

        // --- Save ---
        if (key === 's' && ctrl) {
            e.preventDefault();
            this.app.saveCurrentLabels();
        }
        if (key === 's' && !ctrl) {
            this.app.saveCurrentLabels();
        }

        // --- Undo ---
        if (key === 'z' && ctrl) {
            e.preventDefault();
            this.app.canvas.undo();
        }

        // --- Zoom ---
        if (key === '=' || key === '+') { this.app.canvas.zoomIn(); }
        if (key === '-') { this.app.canvas.zoomOut(); }
        if (key === '0') { this.app.canvas.fitToView(); }

        // --- Class selection (1-9) ---
        if (key >= '1' && key <= '9') {
            const idx = parseInt(key) - 1;
            this.app.classManager.selectByIndex(idx);
        }

        // --- Accept all predictions ---
        if (key === ' ') {
            e.preventDefault();
            this.app.acceptAllPredictions();
        }

        // --- Escape ---
        if (key === 'escape') {
            this.app.canvas.selectBox(-1);
            // Close modals
            document.getElementById('settings-modal').style.display = 'none';
            document.getElementById('training-modal').style.display = 'none';
        }
    }
}

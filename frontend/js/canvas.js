/**
 * Canvas Annotation Engine — bbox drawing, selection, editing
 */
class CanvasEngine {
    constructor(canvasId, containerId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.container = document.getElementById(containerId);

        // State
        this.image = null;
        this.imageName = '';
        this.boxes = [];           // Array of { class_id, x_center, y_center, width, height, confidence, is_predicted }
        this.selectedBoxIndex = -1;
        this.activeClassId = 0;

        // View transform
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;

        // Tool state
        this.tool = 'draw';  // draw | select | pan
        this.isDrawing = false;
        this.isPanning = false;
        this.isResizing = false;
        this.isDragging = false;

        // Drawing temp
        this.drawStart = null;
        this.drawCurrent = null;

        // Drag/resize
        this.dragOffset = { x: 0, y: 0 };
        this.resizeHandle = null;
        this.HANDLE_SIZE = 6;

        // Undo stack
        this.undoStack = [];
        this.maxUndo = 50;

        // Classes reference (set externally)
        this.classes = [];

        // Callbacks
        this.onBoxesChanged = null;
        this.onSelectionChanged = null;
        this.onMouseMove = null;

        this._bindEvents();
    }

    _bindEvents() {
        this.canvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this._onMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });
        this.canvas.addEventListener('dblclick', (e) => this._onDblClick(e));

        // Resize observer
        this._resizeObserver = new ResizeObserver(() => this._resizeCanvas());
        this._resizeObserver.observe(this.container);
    }

    _resizeCanvas() {
        const rect = this.container.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.render();
    }

    // --- Image Management ---
    loadImage(url, name) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                this.image = img;
                this.imageName = name;
                this.canvas.style.display = 'block';
                document.getElementById('canvas-placeholder').style.display = 'none';
                this._resizeCanvas();
                this.fitToView();
                resolve();
            };
            img.onerror = reject;
            img.src = url;
        });
    }

    clearImage() {
        this.image = null;
        this.imageName = '';
        this.boxes = [];
        this.selectedBoxIndex = -1;
        this.canvas.style.display = 'none';
        document.getElementById('canvas-placeholder').style.display = 'flex';
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    // --- Box Management ---
    setBoxes(boxes) {
        this.boxes = boxes.map(b => ({ ...b }));
        this.selectedBoxIndex = -1;
        this.undoStack = [];
        this.render();
        this._notifyBoxesChanged();
    }

    getBoxes() {
        return this.boxes.map(b => ({
            class_id: b.class_id,
            x_center: b.x_center,
            y_center: b.y_center,
            width: b.width,
            height: b.height,
            confidence: b.confidence || null,
            is_predicted: b.is_predicted || false,
        }));
    }

    addBox(box) {
        this._pushUndo();
        this.boxes.push({ ...box });
        this.selectedBoxIndex = this.boxes.length - 1;
        this.render();
        this._notifyBoxesChanged();
    }

    deleteSelected() {
        if (this.selectedBoxIndex < 0) return;
        this._pushUndo();
        this.boxes.splice(this.selectedBoxIndex, 1);
        this.selectedBoxIndex = -1;
        this.render();
        this._notifyBoxesChanged();
    }

    deleteBox(index) {
        if (index < 0 || index >= this.boxes.length) return;
        this._pushUndo();
        this.boxes.splice(index, 1);
        if (this.selectedBoxIndex === index) this.selectedBoxIndex = -1;
        else if (this.selectedBoxIndex > index) this.selectedBoxIndex--;
        this.render();
        this._notifyBoxesChanged();
    }

    selectBox(index) {
        this.selectedBoxIndex = index;
        this.render();
        if (this.onSelectionChanged) this.onSelectionChanged(index);
    }

    acceptAllPredictions() {
        this._pushUndo();
        this.boxes.forEach(b => {
            b.is_predicted = false;
            b.confidence = null;
        });
        this.render();
        this._notifyBoxesChanged();
    }

    rejectAllPredictions() {
        this._pushUndo();
        this.boxes = this.boxes.filter(b => !b.is_predicted);
        this.selectedBoxIndex = -1;
        this.render();
        this._notifyBoxesChanged();
    }

    undo() {
        if (this.undoStack.length === 0) return;
        this.boxes = this.undoStack.pop();
        this.selectedBoxIndex = -1;
        this.render();
        this._notifyBoxesChanged();
    }

    _pushUndo() {
        this.undoStack.push(this.boxes.map(b => ({ ...b })));
        if (this.undoStack.length > this.maxUndo) this.undoStack.shift();
    }

    // --- Tool ---
    setTool(tool) {
        this.tool = tool;
        this.canvas.style.cursor = tool === 'draw' ? 'crosshair' : tool === 'pan' ? 'grab' : 'default';
    }

    // --- Zoom / Pan ---
    fitToView() {
        if (!this.image) return;
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        const iw = this.image.width;
        const ih = this.image.height;
        const padding = 40;

        this.scale = Math.min((cw - padding) / iw, (ch - padding) / ih);
        this.offsetX = (cw - iw * this.scale) / 2;
        this.offsetY = (ch - ih * this.scale) / 2;
        this.render();
    }

    zoomIn() { this._zoom(1.2); }
    zoomOut() { this._zoom(1 / 1.2); }

    _zoom(factor) {
        const cx = this.canvas.width / 2;
        const cy = this.canvas.height / 2;
        this.offsetX = cx - factor * (cx - this.offsetX);
        this.offsetY = cy - factor * (cy - this.offsetY);
        this.scale *= factor;
        this.scale = Math.max(0.1, Math.min(10, this.scale));
        this.render();
    }

    getZoomLevel() { return Math.round(this.scale * 100); }

    // --- Coordinate Conversion ---
    _screenToImage(sx, sy) {
        const x = (sx - this.offsetX) / this.scale;
        const y = (sy - this.offsetY) / this.scale;
        return { x, y };
    }

    _imageToScreen(ix, iy) {
        return {
            x: ix * this.scale + this.offsetX,
            y: iy * this.scale + this.offsetY,
        };
    }

    _screenToNormalized(sx, sy) {
        if (!this.image) return { x: 0, y: 0 };
        const img = this._screenToImage(sx, sy);
        return {
            x: Math.max(0, Math.min(1, img.x / this.image.width)),
            y: Math.max(0, Math.min(1, img.y / this.image.height)),
        };
    }

    // --- Mouse Events ---
    _getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    _onMouseDown(e) {
        const pos = this._getMousePos(e);

        if (this.tool === 'pan' || e.button === 1) {
            this.isPanning = true;
            this.panStart = pos;
            this.panOffsetStart = { x: this.offsetX, y: this.offsetY };
            this.canvas.style.cursor = 'grabbing';
            return;
        }

        if (this.tool === 'select') {
            // Check resize handles first
            const handle = this._getResizeHandle(pos);
            if (handle) {
                this.isResizing = true;
                this.resizeHandle = handle;
                this._pushUndo();
                return;
            }

            // Then check box hit
            const hitIndex = this._hitTest(pos);
            if (hitIndex >= 0) {
                this.selectBox(hitIndex);
                this.isDragging = true;
                const box = this.boxes[hitIndex];
                const boxScreen = this._imageToScreen(
                    (box.x_center - box.width / 2) * this.image.width,
                    (box.y_center - box.height / 2) * this.image.height
                );
                this.dragOffset = { x: pos.x - boxScreen.x, y: pos.y - boxScreen.y };
                this._pushUndo();
                return;
            }

            this.selectBox(-1);
            return;
        }

        if (this.tool === 'draw') {
            this.isDrawing = true;
            this.drawStart = pos;
            this.drawCurrent = pos;
        }
    }

    _onMouseMove(e) {
        const pos = this._getMousePos(e);

        // Report screen position
        if (this.onMouseMove && this.image) {
            const imgPos = this._screenToImage(pos.x, pos.y);
            this.onMouseMove(Math.round(imgPos.x), Math.round(imgPos.y));
        }

        if (this.isPanning) {
            this.offsetX = this.panOffsetStart.x + (pos.x - this.panStart.x);
            this.offsetY = this.panOffsetStart.y + (pos.y - this.panStart.y);
            this.render();
            return;
        }

        if (this.isDrawing) {
            this.drawCurrent = pos;
            this.render();
            this._drawTempBox();
            return;
        }

        if (this.isDragging && this.selectedBoxIndex >= 0) {
            const box = this.boxes[this.selectedBoxIndex];
            const newTopLeft = this._screenToImage(pos.x - this.dragOffset.x, pos.y - this.dragOffset.y);
            const bw = box.width * this.image.width;
            const bh = box.height * this.image.height;
            let newCx = (newTopLeft.x + bw / 2) / this.image.width;
            let newCy = (newTopLeft.y + bh / 2) / this.image.height;
            // Clamp
            newCx = Math.max(box.width / 2, Math.min(1 - box.width / 2, newCx));
            newCy = Math.max(box.height / 2, Math.min(1 - box.height / 2, newCy));
            box.x_center = newCx;
            box.y_center = newCy;
            this.render();
            this._notifyBoxesChanged();
            return;
        }

        if (this.isResizing && this.selectedBoxIndex >= 0) {
            this._doResize(pos);
            return;
        }

        // Update cursor for resize handles
        if (this.tool === 'select') {
            const handle = this._getResizeHandle(pos);
            if (handle) {
                const cursors = { nw: 'nw-resize', ne: 'ne-resize', sw: 'sw-resize', se: 'se-resize',
                    n: 'n-resize', s: 's-resize', e: 'e-resize', w: 'w-resize' };
                this.canvas.style.cursor = cursors[handle.dir] || 'default';
            } else {
                const hit = this._hitTest(pos);
                this.canvas.style.cursor = hit >= 0 ? 'move' : 'default';
            }
        }
    }

    _onMouseUp(e) {
        if (this.isPanning) {
            this.isPanning = false;
            this.canvas.style.cursor = this.tool === 'pan' ? 'grab' : 'default';
            return;
        }

        if (this.isDrawing) {
            this.isDrawing = false;
            const pos = this._getMousePos(e);

            // Minimum size check
            const dx = Math.abs(pos.x - this.drawStart.x);
            const dy = Math.abs(pos.y - this.drawStart.y);
            if (dx > 5 && dy > 5) {
                const p1 = this._screenToNormalized(this.drawStart.x, this.drawStart.y);
                const p2 = this._screenToNormalized(pos.x, pos.y);

                const xMin = Math.max(0, Math.min(p1.x, p2.x));
                const xMax = Math.min(1, Math.max(p1.x, p2.x));
                const yMin = Math.max(0, Math.min(p1.y, p2.y));
                const yMax = Math.min(1, Math.max(p1.y, p2.y));

                const w = xMax - xMin;
                const h = yMax - yMin;

                if (w > 0.005 && h > 0.005) {
                    this.addBox({
                        class_id: this.activeClassId,
                        x_center: xMin + w / 2,
                        y_center: yMin + h / 2,
                        width: w,
                        height: h,
                        confidence: null,
                        is_predicted: false,
                    });
                }
            }

            this.drawStart = null;
            this.drawCurrent = null;
            this.render();
            return;
        }

        if (this.isDragging) {
            this.isDragging = false;
            return;
        }

        if (this.isResizing) {
            this.isResizing = false;
            this.resizeHandle = null;
            return;
        }
    }

    _onWheel(e) {
        e.preventDefault();
        const factor = e.deltaY > 0 ? 1 / 1.1 : 1.1;
        const pos = this._getMousePos(e);
        this.offsetX = pos.x - factor * (pos.x - this.offsetX);
        this.offsetY = pos.y - factor * (pos.y - this.offsetY);
        this.scale *= factor;
        this.scale = Math.max(0.1, Math.min(10, this.scale));
        this.render();
    }

    _onDblClick(e) {
        if (this.tool === 'select') {
            const pos = this._getMousePos(e);
            const hitIndex = this._hitTest(pos);
            if (hitIndex >= 0) {
                // Cycle class on double-click
                const box = this.boxes[hitIndex];
                this._pushUndo();
                box.class_id = (box.class_id + 1) % Math.max(1, this.classes.length);
                box.is_predicted = false;
                this.selectBox(hitIndex);
                this._notifyBoxesChanged();
            }
        }
    }

    // --- Hit Testing ---
    _hitTest(screenPos) {
        if (!this.image) return -1;
        for (let i = this.boxes.length - 1; i >= 0; i--) {
            const b = this.boxes[i];
            const tlx = (b.x_center - b.width / 2) * this.image.width;
            const tly = (b.y_center - b.height / 2) * this.image.height;
            const brx = (b.x_center + b.width / 2) * this.image.width;
            const bry = (b.y_center + b.height / 2) * this.image.height;

            const tl = this._imageToScreen(tlx, tly);
            const br = this._imageToScreen(brx, bry);

            if (screenPos.x >= tl.x && screenPos.x <= br.x &&
                screenPos.y >= tl.y && screenPos.y <= br.y) {
                return i;
            }
        }
        return -1;
    }

    _getResizeHandle(screenPos) {
        if (this.selectedBoxIndex < 0 || !this.image) return null;
        const b = this.boxes[this.selectedBoxIndex];
        const hs = this.HANDLE_SIZE;

        const x1 = (b.x_center - b.width / 2) * this.image.width;
        const y1 = (b.y_center - b.height / 2) * this.image.height;
        const x2 = (b.x_center + b.width / 2) * this.image.width;
        const y2 = (b.y_center + b.height / 2) * this.image.height;

        const handles = [
            { dir: 'nw', ix: x1, iy: y1 },
            { dir: 'ne', ix: x2, iy: y1 },
            { dir: 'sw', ix: x1, iy: y2 },
            { dir: 'se', ix: x2, iy: y2 },
            { dir: 'n', ix: (x1 + x2) / 2, iy: y1 },
            { dir: 's', ix: (x1 + x2) / 2, iy: y2 },
            { dir: 'w', ix: x1, iy: (y1 + y2) / 2 },
            { dir: 'e', ix: x2, iy: (y1 + y2) / 2 },
        ];

        for (const h of handles) {
            const sp = this._imageToScreen(h.ix, h.iy);
            if (Math.abs(screenPos.x - sp.x) <= hs && Math.abs(screenPos.y - sp.y) <= hs) {
                return { dir: h.dir, boxIndex: this.selectedBoxIndex };
            }
        }
        return null;
    }

    _doResize(screenPos) {
        const b = this.boxes[this.selectedBoxIndex];
        const np = this._screenToNormalized(screenPos.x, screenPos.y);
        const dir = this.resizeHandle.dir;

        let x1 = b.x_center - b.width / 2;
        let y1 = b.y_center - b.height / 2;
        let x2 = b.x_center + b.width / 2;
        let y2 = b.y_center + b.height / 2;

        if (dir.includes('w')) x1 = Math.max(0, Math.min(x2 - 0.005, np.x));
        if (dir.includes('e')) x2 = Math.min(1, Math.max(x1 + 0.005, np.x));
        if (dir.includes('n')) y1 = Math.max(0, Math.min(y2 - 0.005, np.y));
        if (dir.includes('s')) y2 = Math.min(1, Math.max(y1 + 0.005, np.y));

        b.x_center = (x1 + x2) / 2;
        b.y_center = (y1 + y2) / 2;
        b.width = x2 - x1;
        b.height = y2 - y1;

        this.render();
        this._notifyBoxesChanged();
    }

    // --- Rendering ---
    render() {
        const ctx = this.ctx;
        const cw = this.canvas.width;
        const ch = this.canvas.height;

        ctx.clearRect(0, 0, cw, ch);

        if (!this.image) return;

        // Draw image
        ctx.save();
        ctx.translate(this.offsetX, this.offsetY);
        ctx.scale(this.scale, this.scale);
        ctx.drawImage(this.image, 0, 0);

        // Draw boxes
        for (let i = 0; i < this.boxes.length; i++) {
            this._drawBox(ctx, this.boxes[i], i === this.selectedBoxIndex, i);
        }

        ctx.restore();

        // Draw resize handles in screen space
        if (this.selectedBoxIndex >= 0) {
            this._drawHandles(this.selectedBoxIndex);
        }

        // Update zoom display
        const zoomEl = document.getElementById('zoom-level');
        if (zoomEl) zoomEl.textContent = `${this.getZoomLevel()}%`;
    }

    _drawBox(ctx, box, isSelected, index) {
        if (!this.image) return;
        const iw = this.image.width;
        const ih = this.image.height;

        const x = (box.x_center - box.width / 2) * iw;
        const y = (box.y_center - box.height / 2) * ih;
        const w = box.width * iw;
        const h = box.height * ih;

        const classInfo = this.classes[box.class_id];
        const color = classInfo ? classInfo.color : '#00ff00';
        const className = classInfo ? classInfo.name : `class_${box.class_id}`;

        // Fill
        ctx.fillStyle = isSelected ? this._hexToRgba(color, 0.25) : this._hexToRgba(color, 0.12);
        ctx.fillRect(x, y, w, h);

        // Border
        ctx.strokeStyle = color;
        ctx.lineWidth = (isSelected ? 2.5 : 1.5) / this.scale;
        if (box.is_predicted) {
            ctx.setLineDash([6 / this.scale, 4 / this.scale]);
        } else {
            ctx.setLineDash([]);
        }
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);

        // Label tag
        const fontSize = Math.max(10, 12 / this.scale);
        ctx.font = `600 ${fontSize}px Inter, sans-serif`;
        let label = className;
        if (box.confidence !== null && box.confidence !== undefined) {
            label += ` ${(box.confidence * 100).toFixed(0)}%`;
        }

        const metrics = ctx.measureText(label);
        const labelW = metrics.width + 8 / this.scale;
        const labelH = fontSize + 6 / this.scale;

        ctx.fillStyle = color;
        ctx.fillRect(x, y - labelH, labelW, labelH);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x + 4 / this.scale, y - 4 / this.scale);
    }

    _drawHandles(boxIndex) {
        const b = this.boxes[boxIndex];
        if (!this.image) return;

        const x1 = (b.x_center - b.width / 2) * this.image.width;
        const y1 = (b.y_center - b.height / 2) * this.image.height;
        const x2 = (b.x_center + b.width / 2) * this.image.width;
        const y2 = (b.y_center + b.height / 2) * this.image.height;

        const points = [
            { ix: x1, iy: y1 }, { ix: x2, iy: y1 },
            { ix: x1, iy: y2 }, { ix: x2, iy: y2 },
            { ix: (x1 + x2) / 2, iy: y1 }, { ix: (x1 + x2) / 2, iy: y2 },
            { ix: x1, iy: (y1 + y2) / 2 }, { ix: x2, iy: (y1 + y2) / 2 },
        ];

        const hs = this.HANDLE_SIZE;
        const ctx = this.ctx;
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 1.5;

        for (const p of points) {
            const sp = this._imageToScreen(p.ix, p.iy);
            ctx.fillRect(sp.x - hs / 2, sp.y - hs / 2, hs, hs);
            ctx.strokeRect(sp.x - hs / 2, sp.y - hs / 2, hs, hs);
        }
    }

    _drawTempBox() {
        if (!this.drawStart || !this.drawCurrent || !this.image) return;

        const ctx = this.ctx;
        const x = Math.min(this.drawStart.x, this.drawCurrent.x);
        const y = Math.min(this.drawStart.y, this.drawCurrent.y);
        const w = Math.abs(this.drawCurrent.x - this.drawStart.x);
        const h = Math.abs(this.drawCurrent.y - this.drawStart.y);

        const classInfo = this.classes[this.activeClassId];
        const color = classInfo ? classInfo.color : '#00ff00';

        ctx.fillStyle = this._hexToRgba(color, 0.15);
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);

        // Size label
        const p1 = this._screenToImage(this.drawStart.x, this.drawStart.y);
        const p2 = this._screenToImage(this.drawCurrent.x, this.drawCurrent.y);
        const pw = Math.abs(p2.x - p1.x);
        const ph = Math.abs(p2.y - p1.y);
        ctx.font = '11px JetBrains Mono';
        ctx.fillStyle = color;
        ctx.fillText(`${Math.round(pw)} × ${Math.round(ph)}`, x + 4, y + h + 14);
    }

    _hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r},${g},${b},${alpha})`;
    }

    _notifyBoxesChanged() {
        if (this.onBoxesChanged) this.onBoxesChanged(this.getBoxes());
    }
}

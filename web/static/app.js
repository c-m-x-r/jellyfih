// Gene definitions
const GENES = [
    { name: 'CP1 X', key: 'cp1_x', idx: 0 },
    { name: 'CP1 Y', key: 'cp1_y', idx: 1 },
    { name: 'CP2 X', key: 'cp2_x', idx: 2 },
    { name: 'CP2 Y', key: 'cp2_y', idx: 3 },
    { name: 'Bell Tip X', key: 'tip_x', idx: 4 },
    { name: 'Bell Tip Y', key: 'tip_y', idx: 5 },
    { name: 'Thickness Base', key: 't_base', idx: 6 },
    { name: 'Thickness Mid', key: 't_mid', idx: 7 },
    { name: 'Thickness Tip', key: 't_tip', idx: 8 },
];

// State
let currentGenome = [];
let bounds = { lower: [], upper: [], default: [] };
let renderTimeout = null;
let evoData = { generations: [], currentGen: null, individuals: [], currentLog: 'evolution_log.csv' };

// Initialize
async function init() {
    // Fetch bounds
    const response = await fetch('/api/bounds');
    bounds = await response.json();

    // Set initial genome
    currentGenome = [...bounds.default];

    // Build UI
    buildGeneControls();
    updateGenomeDisplay();

    // Initial render
    renderMorphology();

    // Event listeners
    document.getElementById('btn-default').addEventListener('click', loadDefault);
    document.getElementById('btn-aurelia').addEventListener('click', loadAurelia);
    document.getElementById('btn-random').addEventListener('click', loadRandom);
    document.getElementById('btn-load').addEventListener('click', loadFromJson);
    document.getElementById('btn-copy').addEventListener('click', copyToClipboard);

    // Evolution browser
    document.getElementById('log-select').addEventListener('change', onLogSelect);
    document.getElementById('gen-select').addEventListener('change', onGenSelect);
    document.getElementById('btn-render-gen').addEventListener('click', renderGeneration);
    document.getElementById('btn-sim-gen').addEventListener('click', simulateGeneration);
    loadEvolutionLogs();
}

// Build gene control sliders
function buildGeneControls() {
    const container = document.getElementById('gene-controls');

    GENES.forEach(gene => {
        const control = document.createElement('div');
        control.className = 'gene-control';

        const label = document.createElement('label');
        label.innerHTML = `
            <span>${gene.name}</span>
            <span class="value" id="value-${gene.idx}">${currentGenome[gene.idx].toFixed(3)}</span>
        `;

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.id = `gene-${gene.idx}`;
        slider.min = bounds.lower[gene.idx];
        slider.max = bounds.upper[gene.idx];
        slider.step = 0.001;
        slider.value = currentGenome[gene.idx];

        slider.addEventListener('input', (e) => {
            currentGenome[gene.idx] = parseFloat(e.target.value);
            document.getElementById(`value-${gene.idx}`).textContent = e.target.value;
            updateGenomeDisplay();
            debouncedRender();
        });

        control.appendChild(label);
        control.appendChild(slider);
        container.appendChild(control);
    });
}

// Update all slider values
function updateSliders() {
    GENES.forEach(gene => {
        const slider = document.getElementById(`gene-${gene.idx}`);
        const valueDisplay = document.getElementById(`value-${gene.idx}`);
        slider.value = currentGenome[gene.idx];
        valueDisplay.textContent = currentGenome[gene.idx].toFixed(3);
    });
}

// Update genome JSON display
function updateGenomeDisplay() {
    const textarea = document.getElementById('genome-json');
    textarea.value = JSON.stringify(currentGenome.map(v => parseFloat(v.toFixed(3))));
}

// Debounced render (300ms delay)
function debouncedRender() {
    clearTimeout(renderTimeout);
    renderTimeout = setTimeout(renderMorphology, 300);
}

// Render morphology via API
async function renderMorphology() {
    const img = document.getElementById('morphology-img');
    const loading = document.getElementById('loading');

    // Show loading
    loading.classList.remove('hidden');
    img.classList.remove('loaded');

    try {
        const response = await fetch('/api/render', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ genome: currentGenome })
        });

        const data = await response.json();

        if (data.error) {
            console.error('Render error:', data.error);
            return;
        }

        // Update image
        img.src = data.image;
        img.onload = () => {
            img.classList.add('loaded');
            loading.classList.add('hidden');
        };

        // Update stats
        updateStats(data.stats);

    } catch (error) {
        console.error('API error:', error);
        loading.textContent = 'Render failed';
    }
}

// Update statistics panel
function updateStats(stats) {
    document.getElementById('stat-total').textContent = stats.n_total.toLocaleString();
    document.getElementById('stat-robot').textContent = `${stats.n_robot.toLocaleString()} (${(stats.n_robot / stats.n_total * 100).toFixed(1)}%)`;
    document.getElementById('stat-jelly').textContent = stats.n_jelly.toLocaleString();
    document.getElementById('stat-muscle').textContent = stats.muscle_count.toLocaleString();
    document.getElementById('stat-water').textContent = stats.n_water.toLocaleString();

    const geomEl = document.getElementById('stat-geometry');
    if (stats.self_intersecting) {
        geomEl.textContent = 'INVALID';
        geomEl.style.color = 'var(--highlight)';
    } else {
        geomEl.textContent = 'Valid';
        geomEl.style.color = 'var(--accent)';
    }
}

// Load default genome
async function loadDefault() {
    const response = await fetch('/api/default');
    const data = await response.json();
    currentGenome = data.genome;
    updateSliders();
    updateGenomeDisplay();
    renderMorphology();
}

// Load Aurelia genome
async function loadAurelia() {
    const response = await fetch('/api/aurelia');
    const data = await response.json();
    currentGenome = data.genome;
    updateSliders();
    updateGenomeDisplay();
    renderMorphology();
}

// Load random genome
async function loadRandom() {
    const response = await fetch('/api/random');
    const data = await response.json();
    currentGenome = data.genome;
    updateSliders();
    updateGenomeDisplay();
    renderMorphology();
}

// Load genome from JSON textarea
function loadFromJson() {
    try {
        const textarea = document.getElementById('genome-json');
        const genome = JSON.parse(textarea.value);

        if (!Array.isArray(genome) || genome.length !== 9) {
            alert('Invalid genome: must be array of 9 numbers');
            return;
        }

        // Validate bounds
        for (let i = 0; i < 9; i++) {
            if (genome[i] < bounds.lower[i] || genome[i] > bounds.upper[i]) {
                alert(`Gene ${i} out of bounds: ${genome[i]} not in [${bounds.lower[i]}, ${bounds.upper[i]}]`);
                return;
            }
        }

        currentGenome = genome;
        updateSliders();
        updateGenomeDisplay();
        renderMorphology();

    } catch (error) {
        alert('Invalid JSON: ' + error.message);
    }
}

// Copy genome to clipboard
async function copyToClipboard() {
    const textarea = document.getElementById('genome-json');
    try {
        await navigator.clipboard.writeText(textarea.value);
        const btn = document.getElementById('btn-copy');
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 1500);
    } catch (error) {
        // Fallback for older browsers
        textarea.select();
        document.execCommand('copy');
    }
}

// === Evolution Browser ===

async function loadEvolutionLogs() {
    try {
        const response = await fetch('/api/evolution/logs');
        const data = await response.json();
        const select = document.getElementById('log-select');

        if (data.logs.length === 0) {
            select.innerHTML = '<option value="">No logs found</option>';
            document.getElementById('evo-status').textContent = 'No evolution logs found';
            return;
        }

        select.disabled = false;
        select.innerHTML = '';
        data.logs.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            if (name === 'evolution_log.csv') opt.selected = true;
            select.appendChild(opt);
        });

        evoData.currentLog = select.value;
        loadEvolutionSummary();
    } catch (error) {
        document.getElementById('evo-status').textContent = 'Failed to load log list';
    }
}

function onLogSelect() {
    const logName = document.getElementById('log-select').value;
    if (!logName) return;
    evoData.currentLog = logName;

    // Reset generation state
    evoData.currentGen = null;
    evoData.individuals = [];
    document.getElementById('evo-tbody').innerHTML =
        '<tr><td colspan="7" class="evo-empty">Select a generation above</td></tr>';
    document.getElementById('evo-grid-container').classList.add('hidden');

    loadEvolutionSummary();
}

async function loadEvolutionSummary() {
    try {
        const response = await fetch(`/api/evolution/summary?log=${encodeURIComponent(evoData.currentLog)}`);
        const data = await response.json();
        evoData.generations = data.generations;

        const select = document.getElementById('gen-select');
        const btnRender = document.getElementById('btn-render-gen');
        const btnSim = document.getElementById('btn-sim-gen');
        const status = document.getElementById('evo-status');

        if (data.generations.length === 0) {
            select.disabled = true;
            btnRender.disabled = true;
            btnSim.disabled = true;
            select.innerHTML = '<option value="">No data</option>';
            status.textContent = `${evoData.currentLog}: no data`;
            return;
        }

        select.disabled = false;
        btnRender.disabled = false;
        btnSim.disabled = false;
        select.innerHTML = '<option value="">Select generation...</option>';

        data.generations.forEach(g => {
            const opt = document.createElement('option');
            opt.value = g.generation;
            opt.textContent = `Gen ${g.generation} — best: ${g.best_fitness.toFixed(3)}, avg: ${g.avg_fitness.toFixed(3)} (${g.count} ind.)`;
            select.appendChild(opt);
        });

        status.textContent = `${evoData.currentLog}: ${data.generations.length} generations, ${data.total_individuals} individuals`;
    } catch (error) {
        document.getElementById('evo-status').textContent = 'Failed to load evolution data';
    }
}

async function onGenSelect() {
    const gen = parseInt(document.getElementById('gen-select').value);
    if (isNaN(gen)) {
        document.getElementById('evo-tbody').innerHTML =
            '<tr><td colspan="7" class="evo-empty">Select a generation above</td></tr>';
        evoData.currentGen = null;
        evoData.individuals = [];
        return;
    }

    try {
        const response = await fetch(`/api/evolution/generation/${gen}?log=${encodeURIComponent(evoData.currentLog)}`);
        const data = await response.json();
        evoData.currentGen = gen;
        evoData.individuals = data.individuals;
        populateIndividualTable(data.individuals);
    } catch (error) {
        document.getElementById('evo-tbody').innerHTML =
            '<tr><td colspan="7" class="evo-empty">Failed to load generation</td></tr>';
    }
}

function populateIndividualTable(individuals) {
    const tbody = document.getElementById('evo-tbody');
    tbody.innerHTML = '';

    individuals.forEach((ind, idx) => {
        const tr = document.createElement('tr');
        if (idx === 0) tr.classList.add('best');

        tr.innerHTML = `
            <td class="mono">${ind.individual}</td>
            <td class="mono">${ind.fitness.toFixed(4)}</td>
            <td class="mono">${ind.final_y.toFixed(4)}</td>
            <td class="mono">${ind.drift.toFixed(4)}</td>
            <td class="mono">${ind.muscle_count}</td>
            <td>${ind.valid ? 'Y' : 'N'}</td>
            <td><button class="btn-load-ind" data-idx="${idx}">Load</button></td>
        `;
        tbody.appendChild(tr);
    });

    // Attach click handlers
    tbody.querySelectorAll('.btn-load-ind').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const idx = parseInt(e.target.dataset.idx);
            loadIndividual(individuals[idx]);
        });
    });
}

function loadIndividual(ind) {
    currentGenome = [...ind.genome];
    updateSliders();
    updateGenomeDisplay();
    renderMorphology();

    // Scroll to viewer
    document.querySelector('.viewer').scrollIntoView({ behavior: 'smooth' });
}

async function renderGeneration() {
    if (!evoData.individuals.length) return;

    const container = document.getElementById('evo-grid-container');
    const loading = document.getElementById('evo-grid-loading');
    const img = document.getElementById('evo-grid-img');
    const btn = document.getElementById('btn-render-gen');

    container.classList.remove('hidden');
    loading.classList.remove('hidden');
    img.style.opacity = '0';
    btn.disabled = true;
    btn.textContent = 'Rendering...';

    try {
        const response = await fetch('/api/render/grid', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ individuals: evoData.individuals })
        });

        const data = await response.json();

        if (data.error) {
            loading.textContent = 'Render failed: ' + data.error;
            return;
        }

        img.src = data.image;
        img.onload = () => {
            img.style.opacity = '1';
            loading.classList.add('hidden');
        };
    } catch (error) {
        loading.textContent = 'Render failed';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Render Generation';
    }
}

// === Simulate Generation ===

let simPollInterval = null;

async function simulateGeneration() {
    if (evoData.currentGen === null) return;

    const btn = document.getElementById('btn-sim-gen');
    const progress = document.getElementById('sim-progress');
    const progressText = document.getElementById('sim-progress-text');
    const progressFill = document.getElementById('sim-progress-fill');

    btn.disabled = true;
    btn.textContent = 'Starting...';
    progress.classList.remove('hidden');
    progressText.textContent = 'Starting simulation (GPU init may take a moment)...';
    progressFill.style.width = '0%';

    // Hide previous video
    document.getElementById('evo-video-container').classList.add('hidden');

    try {
        const response = await fetch('/api/simulate/generation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                generation: evoData.currentGen,
                log: evoData.currentLog,
                frames: parseInt(document.getElementById('sim-frames').value),
                web_palette: true,
            })
        });

        const data = await response.json();

        if (data.error) {
            progressText.textContent = 'Error: ' + data.error;
            btn.disabled = false;
            btn.textContent = 'Simulate Generation';
            return;
        }

        // Start polling for progress
        pollSimStatus(data.video, data.total_frames);

    } catch (error) {
        progressText.textContent = 'Failed to start simulation';
        btn.disabled = false;
        btn.textContent = 'Simulate Generation';
    }
}

function pollSimStatus(expectedVideo, totalFrames) {
    if (simPollInterval) clearInterval(simPollInterval);

    const progressText = document.getElementById('sim-progress-text');
    const progressFill = document.getElementById('sim-progress-fill');
    const btn = document.getElementById('btn-sim-gen');

    simPollInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/simulate/status');
            const status = await response.json();

            if (status.state === 'running' || status.state === 'starting') {
                const frame = status.frame || 0;
                const total = status.total_frames || totalFrames;
                const pct = total > 0 ? (frame / total * 100) : 0;
                progressText.textContent = `Rendering: ${frame}/${total} frames`;
                progressFill.style.width = pct + '%';

            } else if (status.state === 'done') {
                clearInterval(simPollInterval);
                simPollInterval = null;

                progressText.textContent = 'Simulation complete';
                progressFill.style.width = '100%';
                btn.disabled = false;
                btn.textContent = 'Simulate Generation';

                // Show video
                const videoUrl = `/api/simulate/video/${encodeURIComponent(expectedVideo)}`;
                showSimVideo(videoUrl, expectedVideo);

                // Hide progress after a moment
                setTimeout(() => {
                    document.getElementById('sim-progress').classList.add('hidden');
                }, 2000);

            } else if (status.state === 'error') {
                clearInterval(simPollInterval);
                simPollInterval = null;
                progressText.textContent = 'Simulation failed: ' + (status.error || 'unknown error');
                btn.disabled = false;
                btn.textContent = 'Simulate Generation';
            }
        } catch (error) {
            // Network error, keep polling
        }
    }, 1500);
}

function showSimVideo(videoUrl, filename) {
    const container = document.getElementById('evo-video-container');
    const video = document.getElementById('evo-video');
    const source = document.getElementById('evo-video-src');
    const label = document.getElementById('evo-video-label');
    const link = document.getElementById('evo-video-link');

    // Force reload by updating src with cache-bust
    const bustUrl = videoUrl + '?t=' + Date.now();
    source.src = bustUrl;
    video.load();

    label.textContent = filename;
    link.href = bustUrl;
    container.classList.remove('hidden');

    container.scrollIntoView({ behavior: 'smooth' });
}

// Start app
init();

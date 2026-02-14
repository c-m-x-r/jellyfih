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

// Start app
init();

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const dropZoneContent = document.querySelector('.drop-zone-content');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    
    let selectedFile = null;

    // Drag and Drop Events
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        
        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            dropZoneContent.style.opacity = '0';
        };
        reader.readAsDataURL(file);
        
        analyzeBtn.disabled = false;
        resultsSection.style.display = 'none'; // Hide previous results
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI State: Loading
        analyzeBtn.disabled = true;
        loader.style.display = 'block';
        resultsSection.style.display = 'none';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to analyze image');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            loader.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });

    function displayResults(data) {
        // Summary
        document.getElementById('summary-text').textContent = data.summary_report;
        
        // Severity
        const severityStr = data.damage_prediction.overall_severity;
        const severityEl = document.getElementById('severity-val');
        severityEl.textContent = severityStr.toUpperCase();
        
        // Color code severity
        severityEl.className = 'dash-value';
        if(severityStr.toLowerCase() === 'minor') severityEl.classList.add('severity-minor');
        if(severityStr.toLowerCase() === 'moderate') severityEl.classList.add('severity-moderate');
        if(severityStr.toLowerCase() === 'severe') severityEl.classList.add('severity-severe');

        // Parts
        const parts = data.damage_detection.damaged_parts.map(p => p.part);
        // deduplicate just in case
        const uniqueParts = [...new Set(parts)];
        
        document.getElementById('parts-count').textContent = data.damage_detection.total_damaged_parts;
        document.getElementById('parts-val').textContent = uniqueParts.length > 0 ? uniqueParts.join(', ') : 'None';

        // Claim
        const claimAmount = data.claim_prediction.claim_amount;
        document.getElementById('claim-val').textContent = `$${claimAmount.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;

        resultsSection.style.display = 'block';
    }
});

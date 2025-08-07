// FlowFormer++ Web Interface JavaScript

class FlowApp {
    constructor() {
        this.currentSessionId = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkServerStatus();
    }

    setupEventListeners() {
        // File input change events for preview
        document.getElementById('image1').addEventListener('change', (e) => {
            this.previewImage(e.target, 'preview1');
        });

        document.getElementById('image2').addEventListener('change', (e) => {
            this.previewImage(e.target, 'preview2');
        });

        // Form submission
        document.getElementById('uploadForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.computeFlow();
        });

        // Download button
        document.getElementById('downloadBtn').addEventListener('click', () => {
            this.downloadResult();
        });
    }

    previewImage(input, previewId) {
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            const preview = document.getElementById(previewId);
            
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            
            reader.readAsDataURL(input.files[0]);
        }
    }

    async checkServerStatus() {
        try {
            const response = await fetch('/status');
            const status = await response.json();
            
            if (status.model_loaded) {
                console.log('Server is ready, model loaded on:', status.device);
            } else {
                this.showError('Model not loaded on server');
            }
        } catch (error) {
            console.error('Failed to check server status:', error);
        }
    }

    async computeFlow() {
        const form = document.getElementById('uploadForm');
        const formData = new FormData(form);
        
        // Validate files
        const image1 = document.getElementById('image1').files[0];
        const image2 = document.getElementById('image2').files[0];
        
        if (!image1 || !image2) {
            this.showError('Please select both images');
            return;
        }

        // Show progress
        this.showProgress('Uploading images...');
        this.disableForm(true);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.currentSessionId = result.session_id;
                this.showProgress('Flow computed successfully!');
                this.displayResults(result);
                this.hideProgress();
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Error computing flow:', error);
            this.showError('Error computing flow: ' + error.message);
            this.hideProgress();
        } finally {
            this.disableForm(false);
        }
    }

    displayResults(result) {
        // Hide placeholder and show results
        document.getElementById('placeholderResults').style.display = 'none';
        document.getElementById('resultsContent').style.display = 'block';

        // Set flow visualization image
        const flowImg = document.getElementById('flowVisualization');
        flowImg.src = `/result/${result.result_filename}`;

        // Update statistics
        document.getElementById('computationTime').textContent = result.computation_time;
        document.getElementById('flowRange').textContent = 
            `${result.flow_stats.min.toFixed(2)} / ${result.flow_stats.max.toFixed(2)}`;
        document.getElementById('flowShape').textContent = 
            `${result.flow_stats.shape[0]} × ${result.flow_stats.shape[1]}`;
        document.getElementById('flowMean').textContent = result.flow_stats.mean.toFixed(3);
        document.getElementById('flowStd').textContent = result.flow_stats.std.toFixed(3);

        // Update dimension information if available
        if (result.image_info) {
            const img1Dims = result.image_info.image1_dimensions;
            const img2Dims = result.image_info.image2_dimensions;
            const flowDims = result.image_info.flow_dimensions;
            const dimsMatch = result.image_info.dimensions_match;
            
            // Update dimension display (assuming these elements exist in HTML)
            const dimInfoElement = document.getElementById('dimensionInfo');
            if (dimInfoElement) {
                dimInfoElement.innerHTML = `
                    <strong>Dimensions:</strong><br>
                    Image 1: ${img1Dims[0]} × ${img1Dims[1]}<br>
                    Image 2: ${img2Dims[0]} × ${img2Dims[1]}<br>
                    Flow: ${flowDims[0]} × ${flowDims[1]}<br>
                    <span class="${dimsMatch ? 'text-success' : 'text-warning'}">
                        ${dimsMatch ? '✓ Flow matches input dimensions' : '⚠ Flow resized to match input'}
                    </span>
                `;
            }
        }

        // Store result filename for download
        document.getElementById('downloadBtn').dataset.filename = result.result_filename;
    }

    downloadResult() {
        const filename = document.getElementById('downloadBtn').dataset.filename;
        if (filename) {
            const link = document.createElement('a');
            link.href = `/result/${filename}`;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    showProgress(message) {
        document.getElementById('progressText').textContent = message;
        document.getElementById('progressSection').style.display = 'block';
    }

    hideProgress() {
        setTimeout(() => {
            document.getElementById('progressSection').style.display = 'none';
        }, 1000);
    }

    disableForm(disabled) {
        const btn = document.getElementById('computeBtn');
        const inputs = document.querySelectorAll('#uploadForm input');
        
        inputs.forEach(input => input.disabled = disabled);
        btn.disabled = disabled;
        
        if (disabled) {
            btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Computing...';
        } else {
            btn.innerHTML = '<i class="fas fa-calculator me-2"></i>Compute Optical Flow';
        }
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        const modal = new bootstrap.Modal(document.getElementById('errorModal'));
        modal.show();
    }

    // Clean up session files when page is closed
    cleanup() {
        if (this.currentSessionId) {
            navigator.sendBeacon(`/cleanup/${this.currentSessionId}`, '{}');
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.flowApp = new FlowApp();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.flowApp) {
        window.flowApp.cleanup();
    }
});

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatTime(seconds) {
    if (seconds < 1) {
        return (seconds * 1000).toFixed(0) + ' ms';
    }
    return seconds.toFixed(2) + ' s';
}

// Add drag and drop functionality
['image1', 'image2'].forEach(id => {
    const input = document.getElementById(id);
    const container = input.parentElement;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        container.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        container.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        container.addEventListener(eventName, unhighlight, false);
    });
    
    container.addEventListener('drop', handleDrop, false);
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        container.classList.add('highlight');
    }
    
    function unhighlight(e) {
        container.classList.remove('highlight');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            input.files = files;
            const event = new Event('change', { bubbles: true });
            input.dispatchEvent(event);
        }
    }
});

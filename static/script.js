// Traffic Control System JavaScript

let cameraStream = null;
let isProcessing = false;
let trafficStats = {
    totalVehicles: 0,
    emergencyVehicles: 0
};
// When true, automatic signal updates from the server are paused
let manualMode = false;

// Enable/disable manual control UI (buttons + selector)
function setManualControlsEnabled(enabled) {
    const btns = document.querySelectorAll('.manual-control .btn-red, .manual-control .btn-yellow, .manual-control .btn-green');
    btns.forEach(b => {
        if (b) b.disabled = !enabled;
    });
    const sel = document.getElementById('signal-select');
    if (sel) sel.disabled = !enabled;
}

// DOM Elements
const startCameraBtn = document.getElementById('start-camera-btn');
const stopCameraBtn = document.getElementById('stop-camera-btn');
const cameraVideo = document.getElementById('camera-video');
const cameraCanvas = document.getElementById('camera-canvas');
const cameraCtx = cameraCanvas.getContext('2d');
const noCamera = document.getElementById('no-camera');
const videoInput = document.getElementById('video-input');
const videoUploadArea = document.getElementById('video-upload-area');
const processVideoBtn = document.getElementById('process-video-btn');

// Traffic Signal Elements
const redLight = document.getElementById('red-light');
const yellowLight = document.getElementById('yellow-light');
const greenLight = document.getElementById('green-light');
const signalStatus = document.getElementById('signal-status');
const emergencyAlert = document.getElementById('emergency-alert');

// Statistics Elements
const totalVehiclesEl = document.getElementById('total-vehicles');
const currentVehiclesEl = document.getElementById('current-vehicles');
const emergencyCountEl = document.getElementById('emergency-count');
const vehiclesListEl = document.getElementById('vehicles-list');

// Start Camera
startCameraBtn.addEventListener('click', async () => {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        cameraVideo.srcObject = cameraStream;
        startCameraBtn.disabled = true;
        stopCameraBtn.disabled = false;
        noCamera.style.display = 'none';
        showStatus('Camera started. Processing traffic...', 'success');
        
        cameraVideo.addEventListener('loadedmetadata', () => {
            cameraCanvas.width = cameraVideo.videoWidth;
            cameraCanvas.height = cameraVideo.videoHeight;
            processTrafficFrames();
        });
    } catch (error) {
        showStatus('Error accessing camera: ' + error.message, 'error');
    }
});

// Stop Camera
stopCameraBtn.addEventListener('click', stopCamera);

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        cameraVideo.srcObject = null;
        startCameraBtn.disabled = false;
        stopCameraBtn.disabled = true;
        isProcessing = false;
        cameraCtx.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);
        noCamera.style.display = 'block';
        vehiclesListEl.innerHTML = '<p class="no-vehicles">No vehicles detected yet</p>';
        updateSignal('red');
        showStatus('Camera stopped.', 'info');
    }
}

// Process Traffic Frames
async function processTrafficFrames() {
    if (!cameraStream || isProcessing) return;
    
    isProcessing = true;
    
    const processFrame = async () => {
        if (!cameraStream) {
            isProcessing = false;
            return;
        }
        
        cameraCtx.drawImage(cameraVideo, 0, 0);
        const frameData = cameraCanvas.toDataURL('image/jpeg', 0.8);
        
        try {
            const response = await fetch('/process_traffic', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: frameData })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Display processed frame
                const img = new Image();
                img.onload = () => {
                    cameraCtx.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);
                    cameraCtx.drawImage(img, 0, 0, cameraCanvas.width, cameraCanvas.height);
                };
                img.src = data.image;
                
                // Update traffic signal(s) unless manual mode is active
                if (!manualMode) {
                    if (data.signal_states) {
                        updateSignals(data.signal_states);
                    } else {
                        updateSignal(data.signal_state);
                    }
                }
                
                // Update statistics
                updateStatistics(data);
                
                // Display vehicles
                displayVehicles(data.vehicles);
                
                // Show emergency alert
                if (data.emergency_detected) {
                    emergencyAlert.style.display = 'block';
                } else {
                    emergencyAlert.style.display = 'none';
                }
            }
        } catch (error) {
            console.error('Error processing frame:', error);
        }
        
        if (cameraStream) {
            setTimeout(processFrame, 100); // ~10 FPS
        } else {
            isProcessing = false;
        }
    };
    
    processFrame();
}

// Update Traffic Signal Display
function updateSignal(state) {
    // Remove active class from all lights
    redLight.classList.remove('active');
    yellowLight.classList.remove('active');
    greenLight.classList.remove('active');
    
    // Add active class to current state
    if (state === 'red') {
        redLight.classList.add('active');
        signalStatus.textContent = 'RED';
        signalStatus.style.color = '#dc3545';
    } else if (state === 'yellow') {
        yellowLight.classList.add('active');
        signalStatus.textContent = 'YELLOW';
        signalStatus.style.color = '#ffc107';
    } else if (state === 'green') {
        greenLight.classList.add('active');
        signalStatus.textContent = 'GREEN';
        signalStatus.style.color = '#28a745';
    }
}

// Update multiple signals (array of states)
function updateSignals(states) {
    if (!Array.isArray(states)) return;
    states.forEach((state, idx) => {
        // Valid idx 0..3 map to signal-1..signal-4
        const i = idx + 1;
        const redEl = document.getElementById(`signal-${i}-red`);
        const yellowEl = document.getElementById(`signal-${i}-yellow`);
        const greenEl = document.getElementById(`signal-${i}-green`);
        const statusEl = document.getElementById(`signal-${i}-status`);

        if (!redEl || !yellowEl || !greenEl || !statusEl) return;

        // Clear active classes
        redEl.classList.remove('active');
        yellowEl.classList.remove('active');
        greenEl.classList.remove('active');

        if (state === 'red') {
            redEl.classList.add('active');
            statusEl.textContent = 'RED';
            statusEl.style.color = '#dc3545';
        } else if (state === 'yellow') {
            yellowEl.classList.add('active');
            statusEl.textContent = 'YELLOW';
            statusEl.style.color = '#ffc107';
        } else if (state === 'green') {
            greenEl.classList.add('active');
            statusEl.textContent = 'GREEN';
            statusEl.style.color = '#28a745';
        }
    });
}

// Manual Signal Control
async function setSignal(state, index = null) {
    try {
        // If index is provided, we target that signal; otherwise this is a global command (all signals)
        const payload = { state: state };
        if (index !== null && index !== undefined) {
            payload.index = parseInt(index);
        }

        const response = await fetch('/signal_control', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (data.success) {
            // Update UI from server state when available
            if (data.signal_states) {
                updateSignals(data.signal_states);
            } else {
                // If server didn't return states, request a quick refresh to sync
                setTimeout(() => fetch('/traffic_stats').then(r => r.json()).then(d => { if (d.signal_states) updateSignals(d.signal_states); }), 250);
            }
            showStatus(`Signal set to ${state.toUpperCase()}`, 'success');
        } else {
            showStatus('Error setting signal: ' + (data.error || 'Unknown'), 'error');
        }
    } catch (error) {
        showStatus('Error setting signal: ' + error.message, 'error');
    }
}

// Toggle manual mode: when active, automatic updates to the signal UI are paused
function toggleManualMode() {
    manualMode = !manualMode;
    const btn = document.getElementById('manual-toggle');
    const indicator = document.getElementById('manual-mode-indicator');
    if (manualMode) {
        btn.textContent = 'Disable Manual Mode';
        btn.classList.add('active');
        setManualControlsEnabled(true);
        indicator.style.display = 'inline-block';
        showStatus('Manual Mode enabled: automatic signal updates paused', 'info');
    } else {
        btn.textContent = 'Enable Manual Mode';
        btn.classList.remove('active');
        setManualControlsEnabled(false);
        indicator.style.display = 'none';
        showStatus('Manual Mode disabled: automatic updates resumed', 'success');
        // Sync UI with server state after leaving manual mode
        fetch('/traffic_stats').then(r => r.json()).then(d => {
            if (d.signal_states) updateSignals(d.signal_states);
        }).catch(err => console.error('Error syncing signals:', err));
    }
}

// Update Statistics
function updateStatistics(data) {
    if (data.traffic_stats) {
        trafficStats.totalVehicles = data.traffic_stats.total_vehicles;
        trafficStats.emergencyVehicles = data.traffic_stats.emergency_vehicles;
        
        totalVehiclesEl.textContent = trafficStats.totalVehicles;
        emergencyCountEl.textContent = trafficStats.emergencyVehicles;
    }
    currentVehiclesEl.textContent = data.vehicle_count || 0;
}

// Display Vehicles
function displayVehicles(vehicles) {
    if (!vehicles || vehicles.length === 0) {
        vehiclesListEl.innerHTML = '<p class="no-vehicles">No vehicles detected</p>';
        return;
    }
    
    vehiclesListEl.innerHTML = vehicles.map(vehicle => {
        const emergencyBadge = vehicle.is_emergency 
            ? `<span class="emergency-badge">EMERGENCY</span>` 
            : '';
        const emergencyClass = vehicle.is_emergency ? 'emergency' : '';
        
        return `
            <div class="vehicle-item ${emergencyClass}">
                <div class="vehicle-type">
                    ${vehicle.class.toUpperCase()} ${emergencyBadge}
                </div>
                <div class="vehicle-confidence">
                    Confidence: ${vehicle.confidence}%
                    ${vehicle.emergency_type ? ` | Type: ${vehicle.emergency_type}` : ''}
                </div>
            </div>
        `;
    }).join('');
}

// Image Upload Elements
const imageUploadArea = document.getElementById('image-upload-area');
const imageInput = document.getElementById('image-input');
const processImageBtn = document.getElementById('process-image-btn');

// Image Upload
imageUploadArea.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        processImageBtn.disabled = false;
        showStatus('Image selected. Click "Process Image" to detect vehicles.', 'info');
    }
});

// Drag and drop for image
imageUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    imageUploadArea.classList.add('dragover');
});

imageUploadArea.addEventListener('dragleave', () => {
    imageUploadArea.classList.remove('dragover');
});

imageUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    imageUploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        imageInput.files = files;
        processImageBtn.disabled = false;
        showStatus('Image selected. Click "Process Image" to detect vehicles.', 'info');
    }
});

processImageBtn.addEventListener('click', async () => {
    const file = imageInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    processImageBtn.disabled = true;
    showStatus('Processing image...', 'info');
    
    // Stop camera if running
    if (cameraStream) {
        stopCamera();
    }
    
    try {
        const response = await fetch('/upload_image', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Display processed image
            const img = document.createElement('img');
            img.src = data.image;
            img.style.width = '100%';
            img.style.display = 'block';
            img.style.borderRadius = '10px';
            
            // Clear canvas and video - hide all camera elements
            cameraCanvas.style.display = 'none';
            cameraVideo.style.display = 'none';
            noCamera.style.display = 'none';
            
            // Clear any existing image or video stream
            const container = document.querySelector('.camera-container');
            const existingImg = container.querySelector('img');
            if (existingImg) {
                existingImg.remove();
            }
            // Remove any video elements from previous video uploads
            const existingVideo = container.querySelector('video');
            if (existingVideo && existingVideo !== cameraVideo) {
                existingVideo.remove();
            }
            
            // Display the processed image
            container.appendChild(img);
            container.style.background = '#000';
            
            // Update traffic signal(s) unless manual mode is active
            if (!manualMode) {
                if (data.signal_states) {
                    updateSignals(data.signal_states);
                } else {
                    updateSignal(data.signal_state);
                }
            }
            
            // Update statistics
            updateStatistics(data);
            
            // Display vehicles
            displayVehicles(data.vehicles);
            
            // Show emergency alert
            if (data.emergency_detected) {
                emergencyAlert.style.display = 'block';
            } else {
                emergencyAlert.style.display = 'none';
            }
            
            showStatus(`Detected ${data.vehicle_count} vehicle(s)! ${data.emergency_detected ? 'EMERGENCY VEHICLE DETECTED!' : ''}`, 'success');
        } else {
            showStatus('Error: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        showStatus('Error processing image: ' + error.message, 'error');
    } finally {
        processImageBtn.disabled = false;
    }
});

// Video Upload
videoUploadArea.addEventListener('click', () => videoInput.click());

videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        processVideoBtn.disabled = false;
        showStatus('Video selected. Click "Process Video" to analyze.', 'info');
    }
});

videoUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    videoUploadArea.classList.add('dragover');
});

videoUploadArea.addEventListener('dragleave', () => {
    videoUploadArea.classList.remove('dragover');
});

videoUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    videoUploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('video/')) {
        videoInput.files = files;
        processVideoBtn.disabled = false;
        showStatus('Video selected. Click "Process Video" to analyze.', 'info');
    }
});

processVideoBtn.addEventListener('click', async () => {
    const file = videoInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    processVideoBtn.disabled = true;
    showStatus('Processing video...', 'info');
    
    try {
        const response = await fetch('/upload_video', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Stop camera if running
            if (cameraStream) {
                stopCamera();
            }

            // Display video feed (MJPEG stream) provided by server
            noCamera.style.display = 'none';
            const videoUrl = data.video_path + '?t=' + new Date().getTime();

            // Create img element for video stream
            const img = document.createElement('img');
            img.src = videoUrl;
            img.style.width = '100%';
            img.style.display = 'block';

            // Clear canvas and show video
            cameraCanvas.style.display = 'none';
            cameraVideo.style.display = 'none';

            const container = document.querySelector('.camera-container');
            const existingImg = container.querySelector('img');
            if (existingImg) {
                existingImg.remove();
            }
            container.appendChild(img);

            showStatus('Video processing started!', 'success');
        } else {
            showStatus('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
    } finally {
        processVideoBtn.disabled = false;
    }
});

// Status Message
function showStatus(message, type) {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = 'status ' + (type || 'info');
    
    if (!message) {
        statusDiv.style.display = 'none';
    }
}

// Fetch traffic stats periodically
setInterval(async () => {
    try {
        const response = await fetch('/traffic_stats');
        const data = await response.json();
        updateStatistics({ traffic_stats: data.stats });
        // Only apply automatic signal updates when manual mode is OFF
        if (!manualMode && data.signal_states) {
            updateSignals(data.signal_states);
        }
    } catch (error) {
        console.error('Error fetching stats:', error);
    }
}, 5000); // Update every 5 seconds

// Check server health on load
window.addEventListener('load', async () => {
    try {
        // Reset traffic statistics on each page load so counts start fresh
        try {
            await fetch('/reset_stats', { method: 'POST' });
        } catch (err) {
            console.warn('Could not reset stats:', err);
        }

        const response = await fetch('/health');
        const data = await response.json();
        if (data.model_loaded) {
            showStatus('Traffic Control System Ready!', 'success');
        } else {
            showStatus('System loading... Model will be downloaded.', 'info');
        }
        // Ensure manual control UI is disabled by default (manualMode=false)
        setManualControlsEnabled(manualMode);
    } catch (error) {
        showStatus('Could not connect to server.', 'error');
    }
});

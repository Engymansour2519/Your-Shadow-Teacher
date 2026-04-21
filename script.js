// State Management
const state = {
    currentPage: 'welcome',
    tasks: JSON.parse(localStorage.getItem('shadowTasks')) || [],
    focusMinutes: 25,
    breakMinutes: 5,
    timerSecondsLeft: 0,
    timerInterval: null,
    isTimerRunning: false,
    webcamStream: null
};

// DOM Elements
const pages = document.querySelectorAll('.page');
const todoInput = document.getElementById('todo-input');
const todoList = document.getElementById('todo-list');
const sessionTodoList = document.getElementById('session-todo-list');
const timerDisplay = document.getElementById('session-timer');
const webcamElement = document.getElementById('webcam');
const cameraPlaceholder = document.getElementById('camera-placeholder');

// --- Navigation ---
function showPage(pageId) {
    state.currentPage = pageId;
    pages.forEach(page => {
        page.classList.remove('active');
        if (page.id === `page-${pageId}`) {
            page.classList.add('active');
        }
    });

    if (pageId === 'session') {
        renderSessionTasks();
        startFocusSession();
    }
}

// --- To-Do Logic ---
function saveTasks() {
    localStorage.setItem('shadowTasks', JSON.stringify(state.tasks));
    renderTasks();
}

function addTask() {
    const text = todoInput.value.trim();
    if (!text) return;

    state.tasks.push({
        id: Date.now(),
        text: text,
        completed: false
    });

    todoInput.value = '';
    saveTasks();
}

function toggleTask(id) {
    state.tasks = state.tasks.map(task =>
        task.id === id ? { ...task, completed: !task.completed } : task
    );
    saveTasks();
    renderSessionTasks();
}

function deleteTask(id) {
    state.tasks = state.tasks.filter(task => task.id !== id);
    saveTasks();
}

function renderTasks() {
    todoList.innerHTML = '';
    state.tasks.forEach(task => {
        const li = document.createElement('li');
        li.className = `todo-item ${task.completed ? 'completed' : ''}`;
        li.innerHTML = `
            <input type="checkbox" ${task.completed ? 'checked' : ''} onchange="toggleTask(${task.id})">
            <span class="todo-text">${task.text}</span>
            <button class="delete-btn" onclick="deleteTask(${task.id})">×</button>
        `;
        todoList.appendChild(li);
    });
}

function renderSessionTasks() {
    sessionTodoList.innerHTML = '';
    state.tasks.forEach(task => {
        const li = document.createElement('li');
        li.className = `todo-item ${task.completed ? 'completed' : ''}`;
        li.innerHTML = `
            <input type="checkbox" ${task.completed ? 'checked' : ''} onchange="toggleTask(${task.id})">
            <span class="todo-text">${task.text}</span>
        `;
        sessionTodoList.appendChild(li);
    });
}

// --- Timer Logic ---
function startFocusSession() {
    state.timerSecondsLeft = state.focusMinutes * 60;
    updateTimerDisplay();
    // Timer will wait for the user to press "Start"
}

function startTimer() {
    if (state.isTimerRunning) return;
    state.isTimerRunning = true;

    state.timerInterval = setInterval(() => {
        state.timerSecondsLeft--;
        updateTimerDisplay();

        if (state.timerSecondsLeft <= 0) {
            clearInterval(state.timerInterval);
            state.isTimerRunning = false;
            showPage('feedback');
        }
    }, 1000);
}

function togglePause() {
    const btn = document.getElementById('pause-timer-btn');
    if (state.isTimerRunning) {
        clearInterval(state.timerInterval);
        state.isTimerRunning = false;
        btn.textContent = 'Resume';
    } else {
        startTimer();
        btn.textContent = 'Pause';
    }
}

function resetTimer() {
    clearInterval(state.timerInterval);
    state.isTimerRunning = false;
    state.timerSecondsLeft = state.focusMinutes * 60;
    updateTimerDisplay();
    document.getElementById('pause-timer-btn').textContent = 'Pause';
}

function updateTimerDisplay() {
    const mins = Math.floor(state.timerSecondsLeft / 60);
    const secs = state.timerSecondsLeft % 60;
    timerDisplay.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatTimeString(seconds) {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
}


// --- Camera & AI Backend Logic ---
let aiInterval = null;
let drawLoop = null;

async function activateCamera() {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const captureCanvas = document.getElementById('capture-canvas');
    const ctx = canvas.getContext('2d');
    const captureCtx = captureCanvas.getContext('2d');

    if (state.webcamStream) return; // Already active

    try {
        state.webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        video.srcObject = state.webcamStream;
        
        canvas.style.display = 'block';
        cameraPlaceholder.style.display = 'none';

        // Draw video to canvas loop
        const render = () => {
            if (!state.webcamStream) return;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            drawLoop = requestAnimationFrame(render);
        };
        drawLoop = requestAnimationFrame(render);

        // Auto-start focus timer
        if (!state.isTimerRunning) {
            startTimer();
        }

        // Start AI detection loop
        if (!aiInterval) {
            aiInterval = setInterval(async () => {
                if (!state.webcamStream) return;

                // Capture frame to hidden canvas
                captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
                const imageData = captureCanvas.toDataURL('image/jpeg', 0.7);

                try {
                    const res = await fetch('/api/detect', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: imageData })
                    });
                    const result = await res.json();
                    if (result.status === 'success') {
                        updateUIWithStats(result.data);
                    }
                } catch (e) {
                    console.error("AI API Error:", e);
                }
            }, 800); // Process every 800ms to balance accuracy and server load
        }

    } catch (err) {
        console.error("Error accessing camera:", err);
        alert("Could not access camera. Please ensure you have given permission.");
    }
}

function deactivateCamera() {
    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(track => track.stop());
        state.webcamStream = null;
    }
    
    if (drawLoop) {
        cancelAnimationFrame(drawLoop);
        drawLoop = null;
    }

    if (aiInterval) {
        clearInterval(aiInterval);
        aiInterval = null;
    }

    document.getElementById('webcam-canvas').style.display = 'none';
    cameraPlaceholder.style.display = 'flex';
    document.getElementById('focus-alert-banner').style.display = 'none';
    
    if (state.isTimerRunning) {
        togglePause();
    }
}

function updateUIWithStats(data) {
    const banner = document.getElementById('focus-alert-banner');
    
    if (data.is_distracted) {
        banner.style.display = 'block';
        banner.textContent = data.distraction_type === 'phone' ? '⚠️ Phone Detected!' : '⚠️ Person Absent!';
    } else {
        banner.style.display = 'none';
    }

    // Update stats panel
    const stats = data.stats;
    document.getElementById('stat-distraction-time').textContent = formatTimeString(stats.total_distraction_time || 0);
    document.getElementById('stat-distractions').textContent = stats.distraction_count || 0;
    document.getElementById('stat-phone-held').textContent = stats.phone_held || 0;
    document.getElementById('stat-phone-near').textContent = stats.phone_near_face || 0;
}

// --- Event Listeners ---
document.getElementById('add-todo-btn').addEventListener('click', addTask);
todoInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') addTask(); });

document.getElementById('start-journey-btn').addEventListener('click', () => {
    if (state.tasks.length === 0) {
        alert("Please add at least one task before starting!");
        return;
    }
    showPage('selection');
});

document.querySelectorAll('.mode-card').forEach(btn => {
    btn.addEventListener('click', () => {
        state.focusMinutes = parseInt(btn.dataset.focus);
        state.breakMinutes = parseInt(btn.dataset.break);
        showPage('session');
    });
});

document.querySelectorAll('.back-btn').forEach(btn => {
    btn.addEventListener('click', () => showPage(btn.dataset.target));
});

document.getElementById('activate-camera-btn').addEventListener('click', activateCamera);
document.getElementById('start-model-btn').addEventListener('click', activateCamera);
document.getElementById('end-model-btn').addEventListener('click', deactivateCamera);
document.getElementById('pause-timer-btn').addEventListener('click', togglePause);
document.getElementById('reset-timer-btn').addEventListener('click', resetTimer);

document.getElementById('restart-yes-btn').addEventListener('click', () => {
    showPage('selection');
});

document.getElementById('restart-no-btn').addEventListener('click', () => {
    document.querySelector('.feedback-actions').classList.add('hidden');
    document.getElementById('thank-you-msg').classList.remove('hidden');
});

// Initialization
renderTasks();
window.toggleTask = toggleTask;
window.deleteTask = deleteTask;

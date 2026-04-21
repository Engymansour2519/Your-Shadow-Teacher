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
let statsInterval = null;

function activateCamera() {
    if (webcamElement.src.includes('/video_feed')) return; // Already active

    // Connect to Flask AI stream instead of raw local webcam
    webcamElement.src = '/video_feed';
    webcamElement.style.display = 'block';
    cameraPlaceholder.style.display = 'none';

    // Auto-start the focus timer as soon as the camera turns on
    if (!state.isTimerRunning) {
        startTimer();
    }

    // Start polling the API for distraction data
    if (!statsInterval) {
        statsInterval = setInterval(pollStats, 1000);
    }
}

function deactivateCamera() {
    // Disconnect stream
    webcamElement.src = '';
    webcamElement.style.display = 'none';
    cameraPlaceholder.style.display = 'flex';
    
    // Stop polling stats
    if (statsInterval) {
        clearInterval(statsInterval);
        statsInterval = null;
    }
    
    // Hide visual alerts
    document.getElementById('focus-alert-banner').style.display = 'none';
    
    // Pause timer
    if (state.isTimerRunning) {
        togglePause();
    }
}

async function pollStats() {
    try {
        const res = await fetch('/api/stats');
        const result = await res.json();
        if (result.status !== 'success') return;

        const data = result.data;
        const banner = document.getElementById('focus-alert-banner');

        if (data.is_distracted) {
            banner.style.display = 'block';
        } else {
            banner.style.display = 'none';
        }

        // Update Model Feedback Stats
        document.getElementById('stat-distraction-time').textContent = formatTimeString(data.total_distraction_time || 0);
        document.getElementById('stat-distractions').textContent = data.total_distractions || 0;
        document.getElementById('stat-phone-held').textContent = data.phone_held || 0;
        document.getElementById('stat-phone-near').textContent = data.phone_near_face || 0;
    } catch (e) {
        // Server not reachable — silently ignore
    }
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

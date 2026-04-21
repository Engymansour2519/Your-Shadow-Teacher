# 🌿 Shadow Teacher: AI-Powered Focus Journey

Shadow Teacher is an intelligent Pomodoro focus assistant that uses **YOLO26** (the latest state-of-the-art object detection) to monitor your productivity in real-time. It detects distractions like phone usage and person absence, helping you stay in the flow.

## ✨ Key Features
- **AI Distraction Detection**: Real-time monitoring for phones and desk presence.
- **Dynamic Pomodoro Timer**: Automatically pauses your session if you get distracted.
- **Cloud-Ready Architecture**: Browser-based camera capture for deployment on free cloud tiers.
- **Rich Aesthetics**: A cozy, nature-inspired UI designed for a peaceful focus experience.

---

## 🚀 Deployment Guide (GitHub + Render)

Follow these steps to host your graduation project online for free:

### 1. Push to GitHub
1. Create a new repository on GitHub.
2. Run these commands in your project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Cloud Ready"
   git branch -M main
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

### 2. Deploy the Backend on Render
1. Go to [Render.com](https://render.com/) and sign up for free.
2. Click **New +** > **Web Service**.
3. Connect your GitHub repository.
4. Set the following:
   - **Name**: `shadow-teacher`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Deploy Web Service**.

### 3. (Optional) Custom Domain via GitHub Pages
If you want to use your `github.io` domain:
1. Go to your GitHub Repo **Settings** > **Pages**.
2. Select the `main` branch and `/ (root)` folder.
3. *Note*: In `static/script.js`, you may need to update the `fetch('/api/detect')` URL to point to your Render URL if you host them separately.

---

## 🛠 Tech Stack
- **Backend**: Python, Flask, OpenCV
- **AI Engine**: YOLO26 (Ultralytics), DeepSORT
- **Frontend**: Vanilla JavaScript, CSS3 (Glassmorphism), HTML5
- **Deployment**: Gunicorn, Render/GitHub

## 👨‍🎓 Graduation Project Context
This project demonstrates the integration of real-time computer vision with modern web technologies to solve productivity challenges. It utilizes the **YOLO26** model for its high efficiency and accuracy on CPU-based cloud environments.

---
*Created with 💚 for the Graduation Project 2026.*

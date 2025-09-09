# AI Tutor (Offline)

This project is a fully offline AI-powered study tutor using a local LLM and Vite + React frontend.

## Setup Guide

### Backend

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend

Run these inside `frontend/`:

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start frontend:
   ```bash
   npm run dev
   ```

The app will run at [http://localhost:5173](http://localhost:5173)

This system works fully offline. Place your PDFs in the root directory to use them.

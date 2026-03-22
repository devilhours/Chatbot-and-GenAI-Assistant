# 🤖 GenAI Assistant with PostgreSQL

> 🚀 Production-ready AI system combining **LLMs, OCR, multimodal understanding, and database persistence**

An advanced **Generative AI Assistant** built using **Streamlit + Google Gemini + PostgreSQL**, capable of:

- 💬 Conversational AI (Chat)
- 📄 PDF Question Answering (with OCR support)
- 🖼️ Image-based Q&A (Multimodal AI)
- 🗂️ Persistent memory using PostgreSQL

## 🌟 Key Highlights

- 🧠 **Gemini LLM Integration** (`gemini-2.5-flash-lite`)
- 📄 **Smart Document Processing** (handles scanned PDFs using OCR)
- 🖼️ **Multimodal AI** (image + text reasoning)
- 🗄️ **Database-backed Memory** (PostgreSQL)
- ⚡ **Interactive UI** using Streamlit
- 🧩 **Modular & Scalable Architecture**

## 🎯 Problem Statement

Traditional AI tools often:
- ❌ Do not support multiple input formats (PDF, Image, Text)
- ❌ Lack persistent memory (no chat history)
- ❌ Fail on scanned PDFs (no OCR support)

## ✅ Solution

This project solves these challenges by:
- Integrating **Gemini multimodal AI**
- Supporting **PDF + OCR + Image understanding**
- Storing all interactions in **PostgreSQL**
- Providing a **single unified interface**

## 🏗️ Architecture
     ┌──────────────┐
     │  Streamlit UI│
     └──────┬───────┘
            │
    ┌───────▼────────┐
    │ Gemini API (LLM)│
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │ PostgreSQL DB   │
    └────────────────┘

## ⚙️ Tech Stack

| Layer        | Technology |
|-------------|-----------|
| Frontend     | Streamlit |
| LLM          | Google Gemini API |
| Framework    | LangChain |
| Database     | PostgreSQL |
| ORM          | SQLAlchemy |
| PDF Parsing  | pdfplumber |
| OCR          | pytesseract |
| Image        | Pillow (PIL) |

## 🚀 Features

### 💬 Chat Mode
- Real-time AI chat using Gemini
- Fast and cost-efficient responses
- Stored in database

### 📄 PDF Q&A
- Upload PDF files
- Extract text using:
  - `pdfplumber` (text PDFs)
  - `pytesseract` (scanned PDFs)
- Ask contextual questions

### 🖼️ Image Q&A
- Upload image (JPG/PNG)
- Ask questions using Gemini multimodal

### 📜 History Mode
- View previous conversations
- Stored with timestamps in PostgreSQL

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
git clone https://github.com/your-username/genai-assistant.git
cd genai-assistant

2️⃣ Create Virtual Environment
python -m venv venv

Activate:
Windows
venv\Scripts\activate

Mac/Linux
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Install Tesseract OCR
Download:
👉 https://github.com/tesseract-ocr/tesseract
Set path (if needed):
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

5️⃣ Setup Environment Variables
Create .env file:
GOOGLE_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://username:password@localhost:5432/dbname

6️⃣ Run PostgreSQL
Ensure PostgreSQL is installed and running.

7️⃣ Run Application
streamlit run app6.py

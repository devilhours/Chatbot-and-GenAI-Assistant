import os
import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import datetime
import psycopg2

from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ If Tesseract is not in PATH, set it manually like this:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================== Setup ==================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
db_url = os.getenv("DATABASE_URL")

if not api_key:
    st.error("⚠️ Please set your GOOGLE_API_KEY in the .env file")
else:
    genai.configure(api_key=api_key)

# ================== Database Setup ==================
engine = None
SessionLocal = None
Base = declarative_base()

if db_url:
    try:
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        st.sidebar.success("✅ Connected to PostgreSQL!")
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to connect to DB: {e}")

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    context_type = Column(String, nullable=True)  # chat / pdf / image
    source_name = Column(String, nullable=True)   # pdf filename / image filename
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Create table if DB is available
if engine:
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as e:
        st.sidebar.error(f"⚠️ Could not create table: {e}")

def save_chat(user_message, ai_response, context_type, source_name=None):
    """Save chat history into Postgres"""
    if not SessionLocal:
        return
    try:
        db = SessionLocal()
        chat = ChatHistory(
            user_message=user_message,
            ai_response=ai_response,
            context_type=context_type,
            source_name=source_name,
        )
        db.add(chat)
        db.commit()
    except SQLAlchemyError as e:
        st.error(f"⚠️ Failed to save chat: {e}")
    finally:
        db.close()

# ================== LangChain LLM ==================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key)

# ================== Streamlit UI ==================
st.set_page_config(page_title="GenAI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 GenAI Assistant with PostgreSQL")
st.caption("Chat with LLM • Ask about PDFs • Ask about Images • Store history in DB")

# Sidebar
mode = st.sidebar.radio("Choose Mode", ["💬 Chat", "📄 PDF Q&A", "🖼️ Image Q&A", "📜 History"])

# ================== Chat Mode ==================
if mode == "💬 Chat":
    st.subheader("💬 Chat with Gemini")
    user_input = st.text_input("You:", "")
    if st.button("Send") and user_input:
        response = llm.invoke(user_input)
        st.success("AI Response:")
        st.write(response.content)

        # Save to DB
        save_chat(user_input, response.content, context_type="chat")

# ================== PDF Q&A ==================
elif mode == "📄 PDF Q&A":
    st.subheader("📄 Upload a PDF and ask questions")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    pdf_text = ""
    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"
                else:
                    pil_image = page.to_image(resolution=300).original
                    pdf_text += pytesseract.image_to_string(pil_image) + "\n"

        # if pdf_text.strip():
        #     st.success(f"✅ Extracted text from {len(pdf.pages)} pages.")
        #     st.text_area("Extracted PDF Content (sample):", pdf_text[:1000] + "...", height=200)
        # else:
        #     st.error("⚠️ Could not extract text from PDF.")

    question = st.text_input("Ask a question about the PDF:")
    if st.button("Ask PDF") and uploaded_file:
        if pdf_text.strip():
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            prompt = f"Here is the PDF content:\n\n{pdf_text}\n\nAnswer: {question}"
            response = model.generate_content(prompt)
            st.subheader("Answer from PDF:")
            st.write(response.text)

            # Save to DB
            save_chat(question, response.text, context_type="pdf", source_name=uploaded_file.name)
        else:
            st.error("⚠️ No text extracted from PDF.")

# ================== Image Q&A ==================
elif mode == "🖼️ Image Q&A":
    st.subheader("🖼️ Upload an Image and ask questions")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if image_file:
        img = Image.open(image_file)
        st.image(img, caption="Uploaded Image", width=150)

        query = st.text_input("Ask a question about this image:")
        if st.button("Ask Image") and query:
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            response = model.generate_content([query, img])
            st.success("Answer from Image:")
            st.write(response.text)

            # Save to DB
            save_chat(query, response.text, context_type="image", source_name=image_file.name)

# ================== History ==================
elif mode == "📜 History":
    st.subheader("📜 Previous Conversations (from Postgres)")
    if SessionLocal:
        db = SessionLocal()
        history = db.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).all()
        db.close()

        for chat in history:
            st.markdown(f"**🧑 You:** {chat.user_message}")
            st.markdown(f"**🤖 AI:** {chat.ai_response}")
            st.caption(f"[{chat.context_type}] {chat.source_name or ''} • {chat.timestamp}")
            st.markdown("---")
    else:
        st.error("⚠️ Database not connected.")

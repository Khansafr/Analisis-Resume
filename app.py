# ============================================================
# AI RESUME SCREENING & CANDIDATE RANKING SYSTEM
# HR Friendly Bilingual Version + Candidate Analysis Aspects
# ============================================================

import os
import re
from typing import List, Tuple, Dict

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "model_artifacts/resume_classifier_model.pkl"
LABEL_ENCODER_PATH = "model_artifacts/label_encoder.pkl"
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# SCORING WEIGHT
# ============================================================

SEMANTIC_WEIGHT = 0.60
SKILL_WEIGHT = 0.30
EXPERIENCE_WEIGHT = 0.05
PROJECT_WEIGHT = 0.05
ORGANIZATION_WEIGHT = 0.00
CERTIFICATION_WEIGHT = 0.00


# ============================================================
# LANGUAGE TEXT
# ============================================================

TEXT = {
    "Indonesia": {
        "home": "Beranda",
        "analysis": "Analisis Resume",
        "about": "Tentang Sistem",
        "navigation": "Navigasi",

        "app_title": "AI Resume Screening System",
        "app_caption": "Sistem AI untuk membantu HR menganalisis resume dan memberi ranking kandidat secara otomatis.",
        "title": "Analisis Resume & Ranking Kandidat",
        "subtitle": "Bantu HR menemukan kandidat paling sesuai berdasarkan job description dan isi resume.",

        "resume_input": "Format Resume",
        "match_analysis": "Analisis Kecocokan",
        "candidate_ranking": "Ranking Kandidat",

        "features": "Fitur Utama",
        "feature_items": """
        <ul>
            <li>Menganalisis beberapa resume kandidat dalam format PDF.</li>
            <li>Membandingkan resume kandidat dengan job description.</li>
            <li>Menampilkan skor kecocokan kandidat secara otomatis.</li>
            <li>Menampilkan skill yang cocok dan skill yang belum sesuai.</li>
            <li>Menampilkan analisis kandidat berdasarkan pengalaman, project, organisasi, dan sertifikasi.</li>
            <li>Mengurutkan kandidat berdasarkan skor terbaik.</li>
            <li>Menyediakan hasil ranking dalam format CSV.</li>
        </ul>
        """,

        "input_jd": "1. Masukkan Job Description",
        "upload_resume": "2. Masukkan Resume Kandidat",
        "jd_label": "Tempel job description di sini",
        "jd_placeholder": "Contoh: Kami mencari Data Analyst yang menguasai Python, SQL, machine learning, pandas, dan data visualization...",
        "upload_label": "Masukkan satu atau lebih resume PDF",
        "pdf_note": "Pastikan resume berbentuk PDF teks. PDF hasil scan gambar mungkin tidak terbaca tanpa OCR.",
        "start": "🚀 Mulai Screening",

        "warning_jd": "Silakan masukkan job description terlebih dahulu.",
        "warning_file": "Silakan upload minimal satu file resume PDF.",
        "detected": "Kebutuhan Pekerjaan yang Terdeteksi",
        "no_skill": "Tidak ada skill yang terdeteksi. Skor akhir akan lebih bergantung pada kecocokan isi resume.",
        "processing": "Memproses",
        "done": "Screening selesai.",

        "summary": "📊 Ringkasan Screening",
        "total_candidate": "Total Kandidat",
        "highest_score": "Skor Tertinggi",
        "average_score": "Rata-rata Skor",
        "top_candidate": "Kandidat Teratas",

        "ranking": "🏆 Ranking Kandidat",
        "visualization": "📈 Visualisasi Ranking",
        "detail": "🔍 Candidate Analysis",

        "matched": "✅ Matched Skills",
        "missing": "❌ Missing Skills",
        "preview": "📄 Resume Preview",
        "download": "📥 Download Hasil Ranking CSV",

        "file_name": "Nama File",
        "predicted_category": "Kategori Prediksi",
        "semantic_score": "Skor Kecocokan Isi",
        "skill_score": "Skor Skill",
        "experience_score": "Skor Pengalaman",
        "organization_score": "Indikator Organisasi",
        "project_score": "Skor Project",
        "certification_score": "Indikator Sertifikasi",
        "final_match_score": "Skor Akhir",
        "recommendation": "Rekomendasi",
        "candidate_resume": "Resume Kandidat",
        "category_distribution": "Distribusi Kategori Kandidat",
        "recommendation_distribution": "Distribusi Rekomendasi",
        "resume_content": "Isi Resume",

        "scoring_weight": "Bobot Penilaian",
        "semantic_weight": "Kecocokan Isi Resume",
        "skill_weight": "Kecocokan Skill",

        "supported_categories": "Kategori Pekerjaan",
        "category_note": "Model klasifikasi dilatih pada 25 kategori pekerjaan berikut.",
        "category_limit": "Catatan: kategori prediksi terbatas pada daftar ini. Namun, analisis kecocokan dan ranking kandidat tetap dapat digunakan untuk resume di luar kategori tersebut.",

        "about_title": "📘 Tentang Sistem",
        "about_desc": """
        Sistem ini membantu HR melakukan proses screening awal resume secara lebih cepat dan objektif.
        HR cukup memasukkan job description dan mengunggah resume kandidat, kemudian sistem akan
        memberikan ranking berdasarkan tingkat kecocokan kandidat.
        """,
        "how_it_works": "Cara Kerja Singkat",
        "workflow_items": [
            "HR memasukkan job description.",
            "HR mengunggah satu atau lebih resume kandidat.",
            "Sistem membaca dan menganalisis isi resume.",
            "Sistem menghitung kecocokan kandidat berdasarkan isi resume, skill, pengalaman, dan project. Organisasi dan sertifikasi ditampilkan sebagai indikator tambahan.",
            "Kandidat diurutkan berdasarkan skor terbaik."
        ],
        "score_info": "Penjelasan Skor",
        "score_desc": "Skor akhir dihitung berdasarkan aspek utama kandidat, yaitu kecocokan isi resume, skill, pengalaman, dan project.",
        "formula": "Skor Akhir = 60% Isi Resume + 30% Skill + 5% Pengalaman + 5% Project. Organisasi dan sertifikasi hanya sebagai indikator tambahan.",

        "threshold_title": "Batas Rekomendasi",
        "final_score": "Skor Akhir",
        "highly_recommended": "Sangat Direkomendasikan",
        "recommended": "Direkomendasikan",
        "consider": "Dipertimbangkan",
        "not_suitable": "Tidak Sesuai",

        "candidate_analysis": "📌 Candidate Analysis",
        "aspect_table": "Tabel Aspek Penilaian",
        "aspect_chart": "Grafik Skor per Aspek",
        "content_match": "Kecocokan Isi Resume",
        "skill": "Skill",
        "experience": "Pengalaman Kerja",
        "organization": "Indikator Organisasi",
        "project": "Project / Portfolio",
        "certification": "Indikator Sertifikasi",
        "aspect": "Aspek Penilaian",
        "score": "Skor",
    },

    "English": {
        "home": "Home",
        "analysis": "Resume Analysis",
        "about": "About System",
        "navigation": "Navigation",

        "app_title": "AI Resume Screening System",
        "app_caption": "AI system to help HR analyze resumes and rank candidates automatically.",
        "title": "Resume Analysis & Candidate Ranking",
        "subtitle": "Help HR find the most suitable candidates based on job description and resume content.",

        "resume_input": "Resume Input",
        "match_analysis": "Match Analysis",
        "candidate_ranking": "Candidate Ranking",

        "features": "Main Features",
        "feature_items": """
        <ul>
            <li>Analyze multiple candidate resumes in PDF format.</li>
            <li>Compare candidate resumes with the job description.</li>
            <li>Generate candidate match scores automatically.</li>
            <li>Show matched and missing skills.</li>
            <li>Show candidate analysis based on experience, project, organization, and certification.</li>
            <li>Rank candidates based on the best score.</li>
            <li>Export ranking results to CSV.</li>
        </ul>
        """,

        "input_jd": "1. Input Job Description",
        "upload_resume": "2. Upload Candidate Resume",
        "jd_label": "Paste job description here",
        "jd_placeholder": "Example: We are looking for a Data Analyst skilled in Python, SQL, machine learning, pandas, and data visualization...",
        "upload_label": "Upload one or more PDF resumes",
        "pdf_note": "Make sure the resume is a text-based PDF. Scanned image PDFs may not be readable without OCR.",
        "start": "🚀 Start Screening",

        "warning_jd": "Please input the job description first.",
        "warning_file": "Please upload at least one resume PDF.",
        "detected": "Detected Job Requirements",
        "no_skill": "No skills detected. The final score will rely more on resume content matching.",
        "processing": "Processing",
        "done": "Screening completed.",

        "summary": "📊 Screening Summary",
        "total_candidate": "Total Candidate",
        "highest_score": "Highest Score",
        "average_score": "Average Score",
        "top_candidate": "Top Candidate",

        "ranking": "🏆 Candidate Ranking",
        "visualization": "📈 Ranking Visualization",
        "detail": "🔍 Candidate Analysis",

        "matched": "✅ Matched Skills",
        "missing": "❌ Missing Skills",
        "preview": "📄 Resume Preview",
        "download": "📥 Download Ranking CSV",

        "file_name": "File Name",
        "predicted_category": "Predicted Category",
        "semantic_score": "Content Match Score",
        "skill_score": "Skill Score",
        "experience_score": "Experience Score",
        "organization_score": "Organization Indicator",
        "project_score": "Project Score",
        "certification_score": "Certification Indicator",
        "final_match_score": "Final Score",
        "recommendation": "Recommendation",
        "candidate_resume": "Candidate Resume",
        "category_distribution": "Candidate Category Distribution",
        "recommendation_distribution": "Recommendation Distribution",
        "resume_content": "Resume Content",

        "scoring_weight": "Scoring Weight",
        "semantic_weight": "Resume Content Match",
        "skill_weight": "Skill Match",

        "supported_categories": "Job Categories",
        "category_note": "The classification model was trained on the following 25 job categories.",
        "category_limit": "Note: predicted categories are limited to this list. However, resume matching and candidate ranking can still be used for resumes outside these categories.",

        "about_title": "📘 About System",
        "about_desc": """
        This system helps HR recruiters perform initial resume screening faster and more objectively.
        HR only needs to input the job description and upload candidate resumes, then the system will
        rank candidates based on their match level.
        """,
        "how_it_works": "How It Works",
        "workflow_items": [
            "HR inputs the job description.",
            "HR uploads one or more candidate resumes.",
            "The system reads and analyzes resume content.",
            "The system calculates candidate match scores based on resume content, skills, experience, and projects. Organization and certification are shown as additional indicators.",
            "Candidates are ranked based on the best score."
        ],
        "score_info": "Score Explanation",
        "score_desc": "The final score is calculated from the main candidate assessment aspects: resume content match, skill, experience, and project.",
        "formula": "Final Score = 60% Resume Content + 30% Skill + 5% Experience + 5% Project. Organization and certification are additional indicators only.",

        "threshold_title": "Recommendation Threshold",
        "final_score": "Final Score",
        "highly_recommended": "Highly Recommended",
        "recommended": "Recommended",
        "consider": "Consider",
        "not_suitable": "Not Suitable",

        "candidate_analysis": "📌 Candidate Analysis",
        "aspect_table": "Assessment Aspect Table",
        "aspect_chart": "Score per Aspect Chart",
        "content_match": "Resume Content Match",
        "skill": "Skill",
        "experience": "Work Experience",
        "organization": "Organization Indicator",
        "project": "Project / Portfolio",
        "certification": "Certification Indicator",
        "aspect": "Assessment Aspect",
        "score": "Score",
    },
}


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    padding-top: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1200px;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #0f172a !important;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 12px;
}

.subtitle {
    font-size: 17px;
    color: #475569;
    margin-bottom: 30px;
}

.card,
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

.metric-card {
    text-align: center;
}

.metric-title {
    font-size: 14px;
    color: #64748b;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #2563eb;
}

.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 14px;
    height: 52px;
    padding: 0 22px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}

textarea {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-radius: 14px !important;
    border: 1px solid #cbd5e1 !important;
}

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] span {
    color: #0f172a !important;
}

div[data-baseweb="select"] svg {
    fill: #0f172a !important;
}

[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 18px !important;
    padding: 18px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #f8fafc !important;
    border: 1px dashed #94a3b8 !important;
    border-radius: 14px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #0f172a !important;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 14px;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder tidak ditemukan: {LABEL_ENCODER_PATH}")

    classifier = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    sbert = SentenceTransformer(SBERT_MODEL_NAME)

    return classifier, label_encoder, sbert


try:
    classifier_model, label_encoder, sbert_model = load_models()
except Exception as e:
    st.error(f"Gagal memuat model / Failed to load model: {e}")
    st.stop()


# ============================================================
# FUNCTIONS
# ============================================================

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(uploaded_file) -> str:
    text = ""

    try:
        pdf_reader = PdfReader(uploaded_file)

        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + " "

    except Exception as e:
        st.warning(f"Gagal membaca file / Failed to read file {uploaded_file.name}: {e}")
        return ""

    return text.encode("utf-8", errors="ignore").decode().strip()


COMMON_SKILLS = [
    # ============================================================
    # GENERAL PROGRAMMING
    # ============================================================
    "python", "java", "javascript", "typescript", "php", "c++", "c#", "r",
    "html", "css", "sql", "scala", "go", "ruby", "kotlin", "swift",
    "matlab", "bash", "shell scripting",

    # ============================================================
    # DATA SCIENCE / DATA ANALYST / BUSINESS ANALYST
    # ============================================================
    "data science", "data analysis", "data analytics", "data analyst",
    "business analysis", "business analyst", "business intelligence",
    "data cleaning", "data preprocessing", "data processing",
    "data transformation", "data wrangling", "data mining",
    "data visualization", "data visualisation", "visualization",
    "dashboard", "interactive dashboard", "reporting", "business report",
    "business insight", "business insights", "kpi", "metrics",
    "performance analysis", "trend analysis", "pattern analysis",
    "predictive analytics", "descriptive analytics", "statistical analysis",
    "statistics", "excel", "power bi", "tableau", "looker",
    "google data studio", "pandas", "numpy", "matplotlib", "seaborn",
    "plotly", "scikit learn", "scikit-learn", "sklearn",

    # ============================================================
    # MACHINE LEARNING / AI
    # ============================================================
    "machine learning", "deep learning", "artificial intelligence", "ai",
    "supervised learning", "unsupervised learning", "classification",
    "regression", "clustering", "predictive modeling",
    "predictive modelling", "modeling", "modelling",
    "natural language processing", "nlp", "text mining",
    "text classification", "sentiment analysis", "recommendation system",
    "computer vision", "image classification", "neural network",
    "tensorflow", "tensorflow.js", "pytorch", "keras", "opencv",

    # ============================================================
    # DATABASE
    # ============================================================
    "database", "database management", "mysql", "postgresql", "mongodb",
    "oracle", "oracle sql", "sql server", "microsoft sql server",
    "sqlite", "nosql", "query optimization", "data warehouse",
    "data warehousing", "bigquery", "snowflake", "redshift",

    # ============================================================
    # ETL DEVELOPER / HADOOP / BIG DATA
    # ============================================================
    "etl", "etl developer", "etl pipeline", "etl pipelines",
    "data pipeline", "data pipelines", "data extraction",
    "data integration", "data loading", "data ingestion",
    "data modeling", "data modelling", "data architecture",
    "apache airflow", "airflow", "dbt", "hadoop", "apache hadoop",
    "spark", "apache spark", "hive", "apache hive", "pig",
    "kafka", "apache kafka", "big data", "mapreduce",

    # ============================================================
    # PYTHON DEVELOPER
    # ============================================================
    "python developer", "flask", "django", "fastapi", "streamlit",
    "dash", "api", "rest api", "backend", "web scraping",
    "beautifulsoup", "selenium", "automation", "scripting",

    # ============================================================
    # JAVA DEVELOPER
    # ============================================================
    "java developer", "spring boot", "spring framework", "hibernate",
    "maven", "gradle", "junit", "microservices", "object oriented programming",
    "oop",

    # ============================================================
    # DOTNET DEVELOPER
    # ============================================================
    ".net", "dotnet", "dotnet developer", "asp.net", "asp.net mvc",
    "asp.net core", "c#", "visual studio", "entity framework",
    "linq", "web api",

    # ============================================================
    # WEB DESIGNING
    # ============================================================
    "web designing", "web design", "frontend", "front end",
    "html", "css", "javascript", "typescript", "react", "vue",
    "angular", "bootstrap", "tailwind", "ui design", "ux design",
    "responsive design", "figma", "adobe xd",

    # ============================================================
    # DEVOPS ENGINEER
    # ============================================================
    "devops", "devops engineer", "docker", "kubernetes", "linux",
    "ubuntu", "aws", "azure", "google cloud", "gcp",
    "cloud computing", "jenkins", "git", "github", "gitlab",
    "bitbucket", "ci cd", "ci/cd", "deployment", "monitoring",
    "terraform", "ansible", "nginx",

    # ============================================================
    # TESTING / AUTOMATION TESTING
    # ============================================================
    "testing", "software testing", "manual testing", "automation testing",
    "test automation", "quality assurance", "qa", "selenium",
    "junit", "testng", "cypress", "postman", "api testing",
    "unit testing", "integration testing", "functional testing",
    "regression testing", "performance testing", "bug tracking",

    # ============================================================
    # BLOCKCHAIN
    # ============================================================
    "blockchain", "smart contract", "solidity", "ethereum",
    "web3", "cryptocurrency", "crypto", "decentralized application",
    "dapp", "nft", "metamask",

    # ============================================================
    # NETWORK SECURITY ENGINEER
    # ============================================================
    "network security", "cybersecurity", "information security",
    "firewall", "vpn", "penetration testing", "vulnerability assessment",
    "wireshark", "ids", "ips", "network monitoring",
    "network administration", "security analysis", "risk assessment",

    # ============================================================
    # SAP DEVELOPER
    # ============================================================
    "sap", "sap developer", "abap", "sap hana", "sap fico",
    "sap mm", "sap sd", "sap basis", "erp", "business process",

    # ============================================================
    # PMO / PROJECT MANAGEMENT / OPERATIONS MANAGER
    # ============================================================
    "pmo", "project management", "project manager", "operations manager",
    "operations management", "agile", "scrum", "kanban",
    "jira", "trello", "notion", "wrike", "risk management",
    "stakeholder management", "requirement analysis",
    "requirements gathering", "documentation", "process improvement",
    "workflow management", "team management", "planning",
    "coordination", "leadership",

    # ============================================================
    # HR
    # ============================================================
    "hr", "human resources", "recruitment", "talent acquisition",
    "employee relations", "payroll", "training and development",
    "performance management", "interviewing", "onboarding",
    "hr administration", "people management",

    # ============================================================
    # SALES
    # ============================================================
    "sales", "sales executive", "sales management", "crm",
    "customer relationship management", "lead generation",
    "business development", "negotiation", "customer service",
    "market analysis", "target achievement", "sales strategy",

    # ============================================================
    # ADVOCATE / LEGAL
    # ============================================================
    "advocate", "legal", "law", "legal research", "legal drafting",
    "litigation", "contract drafting", "compliance", "legal advice",
    "case management", "court", "regulation",

    # ============================================================
    # ARTS
    # ============================================================
    "arts", "graphic design", "illustration", "adobe photoshop",
    "adobe illustrator", "adobe premiere", "adobe after effects",
    "creative design", "visual design", "content creation",
    "photography", "video editing", "animation",

    # ============================================================
    # CIVIL ENGINEER
    # ============================================================
    "civil engineer", "civil engineering", "autocad", "staad pro",
    "sketchup", "revit", "construction management",
    "structural analysis", "surveying", "site engineering",
    "project estimation", "quantity surveying",

    # ============================================================
    # ELECTRICAL ENGINEERING
    # ============================================================
    "electrical engineering", "electrical engineer", "circuit design",
    "power system", "plc", "microcontroller", "arduino",
    "matlab", "simulink", "electronics", "control system",

    # ============================================================
    # MECHANICAL ENGINEER
    # ============================================================
    "mechanical engineer", "mechanical engineering", "solidworks",
    "autocad", "catia", "ansys", "manufacturing",
    "machine design", "thermodynamics", "maintenance",
    "production engineering",

    # ============================================================
    # HEALTH AND FITNESS
    # ============================================================
    "health and fitness", "fitness", "nutrition", "personal trainer",
    "health coaching", "exercise", "wellness", "diet planning",
    "physical training", "sports science",

    # ============================================================
    # SOFT SKILLS
    # ============================================================
    "communication", "teamwork", "collaboration", "leadership",
    "problem solving", "problem-solving", "analytical thinking",
    "critical thinking", "adaptability", "creativity",
    "time management", "attention to detail", "detail oriented",
    "detail-oriented", "initiative", "presentation",
    "public speaking", "decision making", "independent work",
    "work independently", "meet deadlines"
]

SEMANTIC_ALIAS = {
    "teamwork": [
        "collaboration", "team player", "team collaboration",
        "cross functional", "cross-functional", "working with team"
    ],

    "communication": [
        "presentation", "stakeholder", "stakeholder communication",
        "reporting", "public speaking", "business communication"
    ],

    "problem solving": [
        "problem-solving", "solving problems", "troubleshooting",
        "analytical problem solving"
    ],

    "analytical thinking": [
        "analytical mindset", "critical thinking", "analysis",
        "analytical skill"
    ],

    "data analysis": [
        "data analytics", "analyze data", "analysing data",
        "data-driven analysis", "business analysis"
    ],

    "data visualization": [
        "dashboard", "interactive dashboard", "visualisation",
        "visualization", "chart", "graph", "plot"
    ],

    "data cleaning": [
        "data cleansing", "cleaning data", "cleaned data",
        "data preprocessing", "preprocessing"
    ],

    "data preprocessing": [
        "data cleaning", "data cleansing", "data wrangling",
        "preprocess data"
    ],

    "business intelligence": [
        "bi", "bi tools", "power bi", "tableau",
        "dashboard", "business insight"
    ],

    "reporting": [
        "report", "reports", "generate reports",
        "business report", "analytical report"
    ],

    "dashboard": [
        "interactive dashboard", "dashboard development",
        "power bi dashboard", "tableau dashboard"
    ],

    "machine learning": [
        "ml", "predictive analytics", "classification",
        "clustering", "regression", "modeling", "modelling",
        "supervised learning", "unsupervised learning"
    ],

    "predictive analytics": [
        "predictive modeling", "predictive modelling",
        "forecasting", "prediction", "regression model"
    ],

    "statistics": [
        "statistical analysis", "statistical methods",
        "probability", "hypothesis testing"
    ],

    "natural language processing": [
        "nlp", "text mining", "text classification",
        "sentiment analysis"
    ],

    "python": [
        "python programming", "python-based", "python based"
    ],

    "sql": [
        "structured query language", "query", "database query",
        "sql server", "postgresql", "mysql", "oracle sql"
    ],

    "pandas": [
        "python pandas", "pandas dataframe", "dataframe"
    ],

    "numpy": [
        "numerical python", "python numpy"
    ],

    "power bi": [
        "microsoft power bi", "powerbi", "bi dashboard"
    ],

    "tableau": [
        "tableau dashboard", "tableau desktop"
    ],

    "excel": [
        "microsoft excel", "spreadsheet", "pivot table",
        "power query"
    ],

    "etl": [
        "etl pipeline", "etl pipelines", "data pipeline",
        "data extraction", "data transformation", "data loading"
    ],

    "rest api": [
        "api development", "web service", "endpoint", "api"
    ],

    "project management": [
        "pmo", "scrum", "agile", "kanban", "manage project"
    ],

    "requirement analysis": [
        "requirements gathering", "business requirements",
        "user requirement", "system requirement"
    ],

    "automation testing": [
        "test automation", "automated testing", "selenium testing"
    ],

    "manual testing": [
        "software testing", "qa testing", "quality assurance"
    ],

    "devops": [
        "ci cd", "ci/cd", "deployment", "docker", "kubernetes"
    ],

    "blockchain": [
        "smart contract", "ethereum", "web3", "crypto"
    ],

    "network security": [
        "cybersecurity", "information security", "firewall",
        "penetration testing"
    ],

    "human resources": [
        "hr", "recruitment", "talent acquisition", "employee relations"
    ],

    "sales": [
        "business development", "lead generation", "crm",
        "customer relationship management"
    ],

    "legal": [
        "advocate", "law", "legal research", "litigation",
        "contract drafting"
    ],

    "graphic design": [
        "visual design", "creative design", "photoshop",
        "illustrator"
    ],

    "civil engineering": [
        "civil engineer", "construction", "structural analysis",
        "site engineering"
    ],

    "electrical engineering": [
        "electrical engineer", "circuit design", "power system",
        "electronics"
    ],

    "mechanical engineering": [
        "mechanical engineer", "machine design", "manufacturing",
        "solidworks"
    ]
}

def extract_job_skills(job_desc: str) -> List[str]:
    job_desc = clean_text(job_desc)
    detected = []

    for skill in COMMON_SKILLS:
        skill_clean = clean_text(skill)
        pattern = r"\b" + re.escape(skill_clean) + r"\b"

        if re.search(pattern, job_desc):
            detected.append(skill)

    return sorted(list(set(detected)))


def analyze_skill_match(job_skills: List[str], resume_text: str) -> Tuple[List[str], List[str], float]:
    resume_text = clean_text(resume_text)

    matched_skills = []
    missing_skills = []

    for skill in job_skills:
        skill_clean = clean_text(skill)
        pattern = r"\b" + re.escape(skill_clean) + r"\b"
        found = bool(re.search(pattern, resume_text))

        if not found and skill in SEMANTIC_ALIAS:
            for alias in SEMANTIC_ALIAS[skill]:
                alias_pattern = r"\b" + re.escape(clean_text(alias)) + r"\b"

                if re.search(alias_pattern, resume_text):
                    found = True
                    break

        if found:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    skill_score = 0 if len(job_skills) == 0 else (len(matched_skills) / len(job_skills)) * 100

    return matched_skills, missing_skills, round(skill_score, 2)


def extract_aspect_score(keywords: List[str], resume_text: str) -> float:
    resume_text = clean_text(resume_text)
    matched = []

    for keyword in keywords:
        keyword_clean = clean_text(keyword)
        pattern = r"\b" + re.escape(keyword_clean) + r"\b"

        if re.search(pattern, resume_text):
            matched.append(keyword)

    if len(keywords) == 0:
        return 0

    return round((len(matched) / len(keywords)) * 100, 2)


def analyze_resume_aspects(resume_text: str) -> Dict:
    experience_keywords = [
        "experience", "work experience", "employment", "internship",
        "pengalaman", "magang", "kerja", "staff", "analyst",
        "developer", "engineer", "officer", "specialist", "manager",
        "freelance", "full time", "part time"
    ]

    organization_keywords = [
        "organization", "organisasi", "committee", "volunteer",
        "leadership", "community", "club", "association",
        "panitia", "anggota", "ketua", "divisi", "himpunan",
        "bem", "ukm", "osis"
    ]

    project_keywords = [
        "project", "portfolio", "dashboard", "application",
        "system", "website", "machine learning", "data analysis",
        "proyek", "aplikasi", "sistem", "visualization",
        "github", "repository", "case study"
    ]

    certification_keywords = [
        "certification", "certificate", "course", "training",
        "bootcamp", "sertifikat", "pelatihan", "kursus",
        "dicoding", "coursera", "udemy"
    ]

    return {
        "Experience Score": extract_aspect_score(experience_keywords, resume_text),
        "Organization Score": extract_aspect_score(organization_keywords, resume_text),
        "Project Score": extract_aspect_score(project_keywords, resume_text),
        "Certification Score": extract_aspect_score(certification_keywords, resume_text),
    }


def predict_category(cleaned_resume: str) -> str:
    embedding = sbert_model.encode(
        [cleaned_resume],
        normalize_embeddings=True
    )

    prediction = classifier_model.predict(embedding)[0]
    category = label_encoder.inverse_transform([prediction])[0]

    return category


def calculate_semantic_score(cleaned_jd: str, cleaned_resume: str) -> float:
    jd_embedding = sbert_model.encode(
        cleaned_jd,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    resume_embedding = sbert_model.encode(
        cleaned_resume,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    score = util.cos_sim(jd_embedding, resume_embedding).item() * 100

    return round(score, 2)


def calculate_final_score(
    semantic_score: float,
    skill_score: float,
    experience_score: float,
    organization_score: float,
    project_score: float,
    certification_score: float
) -> float:
    final_score = (
        (SEMANTIC_WEIGHT * semantic_score) +
        (SKILL_WEIGHT * skill_score) +
        (EXPERIENCE_WEIGHT * experience_score) +
        (PROJECT_WEIGHT * project_score)
    )

    return round(final_score, 2)


def get_recommendation(score: float) -> str:
    if score >= 75:
        return "Highly Recommended"
    elif score >= 60:
        return "Recommended"
    elif score >= 45:
        return "Consider"
    else:
        return "Not Suitable"


def analyze_resume_file(uploaded_file, cleaned_jd: str, job_skills: List[str]) -> Dict:
    raw_resume_text = extract_text_from_pdf(uploaded_file)

    if not raw_resume_text:
        return {
            "File Name": uploaded_file.name,
            "Predicted Category": "Unreadable PDF",
            "Content Match Score": 0,
            "Skill Score": 0,
            "Experience Score": 0,
            "Organization Score": 0,
            "Project Score": 0,
            "Certification Score": 0,
            "Final Match Score": 0,
            "Recommendation": "Not Suitable",
            "Matched Skills": "-",
            "Missing Skills": ", ".join(job_skills) if job_skills else "-",
            "Resume Text": "",
        }

    cleaned_resume = clean_text(raw_resume_text)

    category = predict_category(cleaned_resume)
    semantic_score = calculate_semantic_score(cleaned_jd, cleaned_resume)

    matched_skills, missing_skills, skill_score = analyze_skill_match(
        job_skills,
        cleaned_resume
    )

    aspect_scores = analyze_resume_aspects(cleaned_resume)

    final_score = calculate_final_score(
        semantic_score,
        skill_score,
        aspect_scores["Experience Score"],
        aspect_scores["Organization Score"],
        aspect_scores["Project Score"],
        aspect_scores["Certification Score"],
    )

    recommendation = get_recommendation(final_score)

    return {
        "File Name": uploaded_file.name,
        "Predicted Category": category,
        "Content Match Score": semantic_score,
        "Skill Score": skill_score,
        "Experience Score": aspect_scores["Experience Score"],
        "Organization Score": aspect_scores["Organization Score"],
        "Project Score": aspect_scores["Project Score"],
        "Certification Score": aspect_scores["Certification Score"],
        "Final Match Score": final_score,
        "Recommendation": recommendation,
        "Matched Skills": ", ".join(matched_skills) if matched_skills else "-",
        "Missing Skills": ", ".join(missing_skills) if missing_skills else "-",
        "Resume Text": raw_resume_text,
    }


# ============================================================
# SIDEBAR
# ============================================================

language = st.sidebar.selectbox(
    "Language / Bahasa",
    ["Indonesia", "English"]
)

t = TEXT[language]

st.sidebar.title("📄 AI Resume Screening")
st.sidebar.caption(t["app_caption"])

menu = st.sidebar.radio(
    t["navigation"],
    [t["home"], t["analysis"], t["about"]]
)

st.sidebar.divider()

with st.sidebar.expander(t["supported_categories"]):
    st.caption(t["category_note"])

    for category in label_encoder.classes_:
        st.write(f"• {category}")

    st.info(t["category_limit"])


# ============================================================
# HOME PAGE
# ============================================================

if menu == t["home"]:
    st.markdown(
        f"""
        <div class="main-title">🤖 {t["app_title"]}</div>
        <div class="subtitle">{t["subtitle"]}</div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{t["resume_input"]}</div>
                <div class="metric-value">PDF</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{t["match_analysis"]}</div>
                <div class="metric-value">AI</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{t["candidate_ranking"]}</div>
                <div class="metric-value">Rank</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        f"""
        <div class="card">
        <h3>{t["features"]}</h3>
        {t["feature_items"]}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RESUME ANALYSIS PAGE
# ============================================================

elif menu == t["analysis"]:
    st.markdown(f"<div class='main-title'>📄 {t['title']}</div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.subheader(t["input_jd"])
        job_desc = st.text_area(
            t["jd_label"],
            height=280,
            placeholder=t["jd_placeholder"]
        )

    with right_col:
        st.subheader(t["upload_resume"])
        uploaded_files = st.file_uploader(
            t["upload_label"],
            type=["pdf"],
            accept_multiple_files=True
        )

        st.info(t["pdf_note"])

    if st.button(t["start"]):
        if not job_desc.strip():
            st.warning(t["warning_jd"])
            st.stop()

        if not uploaded_files:
            st.warning(t["warning_file"])
            st.stop()

        cleaned_jd = clean_text(job_desc)
        job_skills = extract_job_skills(cleaned_jd)

        st.subheader(t["detected"])

        if job_skills:
            st.write(", ".join(job_skills))
        else:
            st.warning(t["no_skill"])

        results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.write(f"{t['processing']}: {uploaded_file.name}")

            result = analyze_resume_file(uploaded_file, cleaned_jd, job_skills)
            results.append(result)

            progress_bar.progress((i + 1) / len(uploaded_files))

        status_text.write(t["done"])

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(
            by="Final Match Score",
            ascending=False
        ).reset_index(drop=True)

        result_df.insert(0, "Rank", range(1, len(result_df) + 1))

        st.markdown("---")
        st.header(t["summary"])

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(t["total_candidate"], len(result_df))
        c2.metric(t["highest_score"], f"{result_df['Final Match Score'].max()}%")
        c3.metric(t["average_score"], f"{round(result_df['Final Match Score'].mean(), 2)}%")
        c4.metric(t["top_candidate"], result_df.iloc[0]["File Name"])

        st.header(t["ranking"])

        display_df = result_df[
            [
                "Rank",
                "File Name",
                "Predicted Category",
                "Content Match Score",
                "Skill Score",
                "Experience Score",
                "Organization Score",
                "Project Score",
                "Certification Score",
                "Final Match Score",
                "Recommendation",
            ]
        ].rename(
            columns={
                "File Name": t["file_name"],
                "Predicted Category": t["predicted_category"],
                "Content Match Score": t["semantic_score"],
                "Skill Score": t["skill_score"],
                "Experience Score": t["experience_score"],
                "Organization Score": t["organization_score"],
                "Project Score": t["project_score"],
                "Certification Score": t["certification_score"],
                "Final Match Score": t["final_match_score"],
                "Recommendation": t["recommendation"],
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.header(t["visualization"])

        fig = px.bar(
            result_df,
            x="File Name",
            y="Final Match Score",
            color="Recommendation",
            text="Final Match Score",
            hover_data=[
                "Predicted Category",
                "Content Match Score",
                "Skill Score",
                "Experience Score",
                "Organization Score",
                "Project Score",
                "Certification Score",
            ]
        )

        fig.update_layout(
            xaxis_title=t["candidate_resume"],
            yaxis_title=t["final_match_score"],
            yaxis_range=[0, 100]
        )

        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            recommendation_count = result_df["Recommendation"].value_counts().reset_index()
            recommendation_count.columns = ["Recommendation", "Total"]

            fig_pie = px.pie(
                recommendation_count,
                names="Recommendation",
                values="Total",
                hole=0.45,
                title=t["recommendation_distribution"]
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            category_count = result_df["Predicted Category"].value_counts().reset_index()
            category_count.columns = ["Predicted Category", "Total"]

            fig_category = px.bar(
                category_count,
                x="Predicted Category",
                y="Total",
                title=t["category_distribution"]
            )

            st.plotly_chart(fig_category, use_container_width=True)

        st.header(t["detail"])

        for _, row in result_df.iterrows():
            with st.expander(
                f"Rank #{row['Rank']} — {row['File Name']} — {row['Final Match Score']}%"
            ):
                if row["Final Match Score"] >= 75:
                    st.success(row["Recommendation"])
                elif row["Final Match Score"] >= 60:
                    st.info(row["Recommendation"])
                elif row["Final Match Score"] >= 45:
                    st.warning(row["Recommendation"])
                else:
                    st.error(row["Recommendation"])

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(t["predicted_category"], row["Predicted Category"])
                m2.metric(t["semantic_score"], f"{row['Content Match Score']}%")
                m3.metric(t["skill_score"], f"{row['Skill Score']}%")
                m4.metric(t["final_match_score"], f"{row['Final Match Score']}%")

                st.markdown(f"### {t['candidate_analysis']}")
                st.caption("Final Score dihitung dari Content Match, Skill, Pengalaman, dan Project. Organisasi serta Sertifikasi hanya indikator tambahan.")

                a1, a2, a3 = st.columns(3)

                a1.metric(t["experience"], f"{row['Experience Score']}%")
                a2.metric(t["organization"], f"{row['Organization Score']}%")
                a3.metric(t["project"], f"{row['Project Score']}%")

                a4, a5 = st.columns(2)

                a4.metric(t["certification"], f"{row['Certification Score']}%")
                a5.metric(t["skill"], f"{row['Skill Score']}%")

                aspect_df = pd.DataFrame({
                    t["aspect"]: [
                        t["content_match"],
                        t["skill"],
                        t["experience"],
                        t["organization"],
                        t["project"],
                        t["certification"],
                    ],
                    t["score"]: [
                        row["Content Match Score"],
                        row["Skill Score"],
                        row["Experience Score"],
                        row["Organization Score"],
                        row["Project Score"],
                        row["Certification Score"],
                    ]
                })

                st.dataframe(
                    aspect_df,
                    use_container_width=True,
                    hide_index=True
                )

                fig_aspect = px.bar(
                    aspect_df,
                    x=t["aspect"],
                    y=t["score"],
                    text=t["score"],
                    title=t["aspect_chart"]
                )

                fig_aspect.update_layout(
                    yaxis_range=[0, 100],
                    xaxis_title=t["aspect"],
                    yaxis_title=t["score"]
                )

                st.plotly_chart(fig_aspect, use_container_width=True)

                st.markdown(f"### {t['matched']}")
                st.write(row["Matched Skills"])

                st.markdown(f"### {t['missing']}")
                st.write(row["Missing Skills"])

                st.markdown(f"### {t['preview']}")
                st.text_area(
                    t["resume_content"],
                    row["Resume Text"][:5000],
                    height=280,
                    key=f"resume_preview_{row['Rank']}"
                )

        export_df = result_df.drop(columns=["Resume Text"], errors="ignore")
        csv = export_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label=t["download"],
            data=csv,
            file_name="candidate_ranking.csv",
            mime="text/csv"
        )


# ============================================================
# ABOUT SYSTEM PAGE
# ============================================================

elif menu == t["about"]:
    st.title(t["about_title"])

    workflow_html = "".join(
        [f"<li>{item}</li>" for item in t["workflow_items"]]
    )

    st.markdown(
        f"""
<div class="card">
<p>{t["about_desc"]}</p>

<h3>{t["how_it_works"]}</h3>
<ol>
{workflow_html}
</ol>

<h3>{t["score_info"]}</h3>
<p>{t["score_desc"]}</p>
<p><b>{t["formula"]}</b></p>
</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
### {t["threshold_title"]}

| {t["final_score"]} | {t["recommendation"]} |
|---:|---|
| ≥ 75 | {t["highly_recommended"]} |
| 60 - 74 | {t["recommended"]} |
| 45 - 59 | {t["consider"]} |
| < 45 | {t["not_suitable"]} |
        """
    )

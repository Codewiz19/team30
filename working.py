import os
import logging
import fitz  # PyMuPDF
import json
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Initialize session state
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

# --- Configuration ---
# Get API key from either st.secrets or environment variables
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default company profile
COMPANY_PROFILE = {
    "legal_name": "FirstStaff Workforce Solutions, LLC",
    "address": "3105 Maple Avenue, Suite 1200, Dallas, TX 75201",
    "employees": 250,
    "annual_revenue": "$25M",
    "years_in_business": 12,
    "certifications": ["ISO 9001", "SOC 2"],
    "industry_sectors": ["IT Services", "Healthcare Staffing", "Government Contracting"],
    "past_performance": ["Texas HHS IT Staffing (2020-2023)", "USDA IT Support (2021-Present)"],
    "capabilities": ["Staff Augmentation", "Managed Services", "IT Consulting", "Healthcare Staffing"]
}

# --- Helper Functions ---
def initialize_groq():
    """Initialize Groq Chat model"""
    if not GROQ_API_KEY:
        st.error("Groq API key not found!")
        return None
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0.2
    )

def load_pdf_text(uploaded_file):
    """Extract text from uploaded PDF"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
       
        doc = fitz.open(tmp_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        # Make sure to close the document before deletion
        doc.close()
        
        try:
            os.unlink(tmp_path)
        except Exception as e:
            logging.warning(f"Could not delete temporary file {tmp_path}: {str(e)}")
            # Continue execution even if file deletion fails
        
        return full_text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return ""

def split_text_into_chunks(text, chunk_size=4000, overlap=200):
    """Split text into overlapping chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def analyze_with_groq(prompt: str) -> str:
    """Execute analysis using Groq AI"""
    try:
        llm = initialize_groq()
        if not llm:
            return "Error: API key not configured"
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logging.error(f"Groq API error: {str(e)}")
        return f"Error: {str(e)}"

# --- Analysis Functions ---
def evaluate_eligibility(rfp_text: str, company_profile: dict) -> str:
    """Evaluate if the company is eligible to bid on the RFP"""
    # Split text into chunks if too long
    chunks = split_text_into_chunks(rfp_text)
    rfp_summary = rfp_text[:8000] if len(chunks) <= 1 else "\n\n".join(chunks[:3])
    
    prompt = f"""
    You are a expert RFP analyst in legal compliance. Analyze the following RFP and evaluate
    if the company described in the company profile meets the eligibility and requirements.
    
    RFP EXCERPT:
    
    {rfp_summary}
    
    
    COMPANY PROFILE:
    
    {json.dumps(company_profile, indent=2)}
    
    
    Provide a clear analysis with the following sections:
    1. Eligibility Assessment (Yes/No)
    2. Key Requirements Analysis
    3. Company Qualification Matching
    4. Recommendations for Eligibility Improvement
    
    Format your response in Markdown.
    """
    return analyze_with_groq(prompt)

def generate_submission_checklist(rfp_text: str) -> str:
    """Generate a submission checklist based on the RFP requirements"""
    # Split text into chunks if too long
    chunks = split_text_into_chunks(rfp_text)
    rfp_summary = rfp_text[:8000] if len(chunks) <= 1 else "\n\n".join(chunks[:3])
    
    prompt = f"""
    You are an expert RFP analyst. Analyze the following RFP excerpt and generate
    a comprehensive submission checklist for the bidder.
    
    RFP EXCERPT:
    
    {rfp_summary}
    
    
    Create a detailed checklist that includes:
    1. Required documentation
    2. Submission deadlines and important dates
    3. Format requirements
    4. Technical response requirements
    5. Pricing/cost proposal requirements
    6. Past performance/references needed
    
    Format your response as a Markdown checklist with clear categories and checkboxes.
    """
    return analyze_with_groq(prompt)

def assess_risks(rfp_text: str) -> str:
    """Assess risks in the RFP"""
    # Split text into chunks if too long
    chunks = split_text_into_chunks(rfp_text)
    rfp_summary = rfp_text[:8000] if len(chunks) <= 1 else "\n\n".join(chunks[:3])
    
    prompt = f"""
    You are an expert risk analyst specializing in government contracting. Analyze the following
    RFP excerpt and identify potential risks for the bidder.
    
    RFP EXCERPT:
    
    {rfp_summary}
    
    
    Provide a comprehensive risk assessment including:
    1. Financial risks
    2. Technical delivery risks
    3. Timeline/schedule risks
    4. Compliance risks
    5. Resource allocation risks
    6. Contract terms risks
    
    For each identified risk, provide:
    - Risk description
    - Potential impact (High/Medium/Low)
    - Suggested mitigation strategy
    
    Format your response in Markdown with clear sections and tables for readability.
    """
    return analyze_with_groq(prompt)

def create_vector_store(text):
    """Create a vector store from text for semantic search"""
    try:
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Split text into chunks
        chunks = split_text_into_chunks(text)
        
        # Create vector store
        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store
    except Exception as e:
        logging.error(f"Error creating vector store: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="RFP Analyzer",
        page_icon="📄",
        layout="wide"
    )
    
    # Sidebar Configuration
    st.sidebar.title("Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload RFP PDF", type="pdf")
   
    # Company Profile Editor
    st.sidebar.subheader("Company Profile")
    company_profile_json = st.sidebar.text_area(
        "Edit Company Profile (JSON)",
        value=json.dumps(COMPANY_PROFILE, indent=2),
        height=300
    )
    
    # Parse company profile with error handling
    try:
        company_profile = json.loads(company_profile_json)
    except json.JSONDecodeError as e:
        st.sidebar.error(f"Error parsing JSON: {str(e)}")
        company_profile = COMPANY_PROFILE
    
    # Main Interface
    st.title("Government RFP Analysis Tool")
   
    if uploaded_file:
        if st.button("Analyze RFP"):
            with st.spinner("Processing PDF..."):
                # Process PDF
                rfp_text = load_pdf_text(uploaded_file)
               
                if not rfp_text:
                    st.error("Failed to extract text from PDF. Please check the file and try again.")
                    return
                
                # Create vector store for potential future use
                with st.spinner("Creating vector database for advanced querying..."):
                    vector_store = create_vector_store(rfp_text)
                
                # Run analyses
                with st.spinner("Analyzing eligibility..."):
                    eligibility_result = evaluate_eligibility(rfp_text, company_profile)
                
                with st.spinner("Generating submission checklist..."):
                    checklist_result = generate_submission_checklist(rfp_text)
                
                with st.spinner("Assessing risks..."):
                    risk_result = assess_risks(rfp_text)
                
                # Store results
                st.session_state.analysis_results = {
                    "eligibility": eligibility_result,
                    "checklist": checklist_result,
                    "risks": risk_result
                }
    
    # Display Results
    if st.session_state.analysis_results:
        tab1, tab2, tab3 = st.tabs(["Eligibility", "Submission Checklist", "Risk Analysis"])
       
        with tab1:
            st.markdown(st.session_state.analysis_results["eligibility"])
       
        with tab2:
            st.markdown(st.session_state.analysis_results["checklist"])
       
        with tab3:
            st.markdown(st.session_state.analysis_results["risks"])

if __name__ == "__main__":  # Fixed the main function call with proper double underscores
    main()
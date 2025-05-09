# RFP Analysis Chatbot

An AI-powered tool for analyzing government Request for Proposal (RFP) documents, built with Streamlit and LLaMA 3 via Groq API.

## Features

- PDF RFP document processing
- Eligibility analysis with detailed assessments
- Scoring assessment with visualization
- Risk analysis with prioritized factors
- Clarification questions generation
- Interactive submission checklist
- Chatbot for Q&A
- Company profile customization
- Executive summary generation

## Prerequisites

- Python 3.8 or higher
- PDF documents for analysis
- Groq API key

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rfp-analysis-chatbot.git
cd rfp-analysis-chatbot
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API key - either:
   - Create a `.env` file with `GROQ_API_KEY=your_api_key_here`
   - Or create `.streamlit/secrets.toml` with `GROQ_API_KEY="your_api_key_here"`

5. Run the application:
```bash
streamlit run chtint.py
```

## Usage

1. Upload an RFP document in PDF format
2. (Optional) Customize company profile in the sidebar
3. Click "Process RFP Document" to extract text
4. Click "Check Eligibility First" to determine bid eligibility 
5. If eligible, click "Perform Full Analysis" for:
   - Executive Summary
   - Scoring Assessment
   - Risk Analysis
   - Clarification Questions
   - Submission Checklist
6. Use the chatbot for additional questions

## Security Notes

- Keep your API keys secure
- Don't commit `.env` or `secrets.toml` files
- Be careful with sensitive RFP documents

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.#   t e a m 3 0  
 #   t e a m 3 0  
 #   t e a m 3 0  
 
# AIDE Python - AI-Assisted Data Extraction

A Python implementation of the AIDE (AI-Assisted Data Extraction) tool for systematic review and meta-analysis.

**Inspired by** [AIDE-Web](https://github.com/noah-schroeder/AIDE-Web) by Schroeder et al. (2025).

## Features

- **Universal LLM Support**: Works with any OpenAI-compatible API endpoint
- **Local Processing**: All data stays on your machine
- **Spreadsheet Support**: Upload coding forms in .csv, .xls, or .xlsx format
- **Structured JSON Output**: Clean JSON responses from LLM
- **Session-Based Storage**: Data stored in memory only (cleared on restart)
- **Human-in-the-Loop**: Every data point requires human validation

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

## How to Use

1. **Configure API Settings** (Setup page)
   - Enter your OpenAI-compatible API endpoint URL
   - Provide your API key
   - Choose the LLM model

2. **Upload Coding Form** (Setup page)
   - Upload your coding form (.csv, .xls, or .xlsx)
   - First row should contain your LLM prompts

3. **Analyze PDFs** (Analyze page)
   - Upload PDF files to analyze
   - Review LLM responses with source information
   - Validate each data point before recording

4. **Export Results** (Final Coding Form page)
   - Download completed coding form as Excel or CSV

## Citation

If you use this tool in your research, please cite the original AIDE paper:

```bibtex
@misc{schroeder2025largelanguagemodelshumanintheloop,
    title={Large Language Models with Human-In-The-Loop Validation for Systematic Review Data Extraction},
    author={Noah L. Schroeder and Chris Davis Jaldi and Shan Zhang},
    year={2025},
    eprint={2501.11840},
    archivePrefix={arXiv},
    primaryClass={cs.HC},
    url={https://arxiv.org/abs/2501.11840},
}
```

## License

This project is licensed under the GNU General Public License v3.0 (GPL v3).

## Disclaimer

- LLM-extracted data must be validated by humans
- Do not rely on LLM extraction without validation
- API keys are stored only in memory and cleared on exit

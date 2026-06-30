# AIDE - AI-Assisted Data Extraction

A PyQt6 desktop application for systematic review data extraction using Large Language Models.

**Inspired by** [AIDE-Web](https://github.com/noah-schroeder/AIDE-Web) by Schroeder et al. (2025).

## Features

- **AI-Generated Templates**: Describe your research topic (PICOS), AI generates extraction fields automatically
- **Study Type Detection**: Auto-identifies RCT, Cohort, Case-Control, Diagnostic studies
- **Built-in Quality Assessment**: ROB2 (RCT), NOS (Cohort/Case-Control), QUADAS-2 (Diagnostic)
- **Study-Level & Arm-Level Fields**: RCT/Cohort fields expand per study arm for long-format extraction
- **Per-Arm Source Tracking**: Each arm gets independent source quotes and page numbers
- **Universal LLM Support**: Works with any OpenAI-compatible API (OpenAI, DeepSeek, MiMo, etc.)
- **Local Processing**: All data stays on your machine
- **Long & Wide Export**: Export one-row-per-arm (long) or one-row-per-study (wide) format
- **Human-in-the-Loop**: Every data point requires human validation

## Versions

| Version | File | Description |
|---------|------|-------------|
| **AIDE Pro v2** | `aide.py` | Full version with unlimited extractions |
| **AIDE Free v2** | `aide_trial.py` | Free version with daily usage limit |

## Installation

```bash
# Clone the repository
git clone https://github.com/silvester1205/AIDE_PRO.git
cd AIDE_PRO

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
# Pro version
python aide.py

# Free version
python aide_trial.py
```

## How to Use

### 1. Setup
- Configure API settings (endpoint, API key, model)
- Describe your research topic using PICOS framework
- Click **AI Generate Template** to auto-create extraction fields
- Edit fields as needed, then click **Apply Template**

### 2. Analyze
- Load PDF articles
- AI extracts data for each field with source quotes
- Review and edit each field, click **Source** to verify against PDF
- Click **Record All** to save extracted data

### 3. Export
- **Export Excel**: One row per study arm (long format, for meta-analysis)
- **Export Wide**: One row per study (pivoted format)

## Quality Assessment Tools

| Tool | Study Type | Domains |
|------|-----------|---------|
| **ROB2** | RCT | 5 domains + Overall (Low / Some concerns / High risk) |
| **NOS** | Cohort | Selection (0-4) + Comparability (0-2) + Outcome (0-3) |
| **NOS** | Case-Control | Selection (0-4) + Comparability (0-2) + Exposure (0-3) |
| **QUADAS-2** | Diagnostic | 4 domains Risk of Bias + 3 domains Applicability |

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

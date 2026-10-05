# 🤖 CatPriceAI – AI-Powered Reinsurance Pricing Workbench

> **An AI-enabled decision support platform for Catastrophe Excess of Loss (Cat XL) Reinsurance Pricing**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![OpenAI](https://img.shields.io/badge/LLM-GPT--4o--mini-purple)
![SQLite](https://img.shields.io/badge/Database-SQLite-orange)

---

## 📌 Overview

CatPriceAI is an AI-powered actuarial pricing workbench designed to modernize catastrophe reinsurance pricing workflows.

The platform combines **traditional actuarial pricing methodologies**, **machine learning**, **Retrieval-Augmented Generation (RAG)**, and an **AI Pricing Agent** to help actuaries and underwriters analyze pricing outputs through natural language instead of manually navigating reports and spreadsheets.

This project was developed as part of the **AI in Multi-Industry Setup Internship** under the **Sri Sathya Sai Institute of Actuaries**.

---

#  Key Features

*  Synthetic Cat XL portfolio generation
*  Exposure and loss analysis
*  Multiple actuarial pricing methodologies
*  Machine Learning pricing model
*  AI-powered Pricing Agent
*  RAG-based treaty document assistant
*  FastAPI backend
*  Streamlit dashboard
*  SQLite-based pricing database

---

#  System Architecture

```
Portfolio Generation
        │
        ▼
Exposure Analysis
        │
        ▼
Loss Analysis
        │
        ▼
Actuarial Pricing Engine
(Burning Cost
Frequency-Severity
Exposure Rating)
        │
        ▼
Pricing Outputs
        │
 ┌──────┴───────────┐
 ▼                  ▼
SQLite DB      FastAPI APIs
                    │
                    ▼
             AI Pricing Agent
                    │
                    ▼
           Streamlit Dashboard
```

---

# 📂 Project Structure

```
CatPriceAI/

├── data/
│   ├── 01_generate_data.py
│   ├── portfolio.csv
│   ├── pricing_output.csv
│   ├── catprice.db
│   └── ...
│
├── analysis/
│   ├── 02_exposure_analysis.py
│   ├── 03_loss_analysis.py
│   └── 04_actuarial_pricing.py
│
├── ml/
│   └── 05_ml_pricing_model.py
│
├── visualization/
│   ├── 06_dashboard.py
│   └── figures/
│
├── rag/
│   ├── 07_rag_engine.py
│   ├── docs/
│   └── vectorstore/
│
├── agent/
│   └── 08_pricing_agent.py
│
├── api/
│   └── api_main.py
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

# 📊 Pricing Methodologies

The pricing engine combines multiple actuarial techniques:

* Burning Cost Method
* Frequency-Severity Modelling
* Exposure Rating
* Machine Learning Pricing

These methodologies produce:

* Technical Rate on Line (ROL)
* Technical Premium
* Pricing Recommendation

---

# 🤖 AI Pricing Agent

One of the core capabilities of the project is the AI Pricing Agent.

Instead of manually reviewing pricing reports, users can ask questions such as:

* What is the technical premium?
* Compare market ROL with technical ROL.
* Which pricing methodology contributed the most?
* Summarize the exposure profile.
* Explain the pricing recommendation.

The agent retrieves validated pricing outputs from the database and analysis modules before generating natural-language explanations using an LLM.

---

# 📚 Retrieval-Augmented Generation (RAG)

The project includes a RAG pipeline that enables users to query treaty wording and underwriting guidelines.

Example queries:

* What are the treaty exclusions?
* What is the attachment point?
* Explain the retention clause.

---

# 🛠️ Technology Stack

### Languages

* Python

### AI & ML

* OpenAI GPT-4o-mini
* LangChain
* FAISS
* BM25

### Backend

* FastAPI

### Frontend

* Streamlit

### Database

* SQLite

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib

---

# 💡 My Contribution

As part of the project team, I was responsible for building the **AI Pricing Agent**.

My work included:

* Developing the conversational Pricing Agent
* Integrating the LLM with pricing outputs
* Connecting the agent to SQLite and CSV-based pricing data
* Designing prompt workflows for pricing explanations
* Implementing retrieval-first responses to minimize hallucinations
* Supporting explainable AI for actuarial pricing
* Contributing to the integration with the FastAPI-based architecture

---

# 📖 What I Learned

This project strengthened my understanding of:

* AI Agent Architecture
* Enterprise AI Systems
* Retrieval-Augmented Generation (RAG)
* FastAPI development
* Prompt Engineering
* LLM integration
* Explainable AI
* Actuarial Pricing Concepts
* Reinsurance Analytics
* Modular Software Architecture

---

# 🎯 Future Enhancements

* PostgreSQL support for large-scale datasets
* Multi-agent orchestration
* Real-time pricing APIs
* Cloud deployment
* Authentication & role-based access
* Model monitoring
* Support for additional actuarial pricing methodologies

---

# 🙏 Acknowledgements

This project was developed as part of the **AI in Multi-Industry Setup Internship** under the **Sri Sathya Sai Institute of Actuaries**.

Special thanks to our mentors and the entire project team for their guidance, collaboration, and continuous support throughout the project.

---
# **Files Overview:**
# CatPriceAI

A synthetic Cat XL treaty pricing workbench with Streamlit front-end, FastAPI backend, pricing analytics, and optional OpenAI-enhanced agent/RAG support.

## Project structure

- `01_generate_data.py` - build synthetic cedant portfolio, loss events, treaty terms, and CSV/SQLite outputs
- `app/10_streamlit_app.py` - main Streamlit UI logic
- `app/streamlit_app.py` - Streamlit entrypoint wrapper
- `api/09_api.py` - FastAPI backend exposing health, exposure, pricing, top losses, treaty Q&A, and pricing query
- `rag/07_rag_engine.py` - treaty document retrieval engine with optional OpenAI prompt support
- `agent/08_pricing_agent.py` - local pricing agent for underwriting queries
- `data/` - generated CSV and SQLite data assets

## Setup

From the project root:

```bash
python -m pip install -r requirements.txt
```

## Generate data

Generate the synthetic dataset required by the app:

```bash
python 01_generate_data.py
```

## Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Then open the browser at:

- `http://localhost:8501`

## Run the FastAPI backend

```bash
uvicorn api.09_api:app --host 127.0.0.1 --port 8000 --reload
```

API endpoints:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/exposure`
- `http://127.0.0.1:8000/pricing`
- `http://127.0.0.1:8000/top-losses`
- `http://127.0.0.1:8000/treaty/ask`
- `http://127.0.0.1:8000/pricing/query`

## Run both together

Use the helper script:

```bash
./run.sh
```

This starts Streamlit at `http://localhost:8501` and FastAPI at `http://127.0.0.1:8000`.

## OpenAI API key support

The Streamlit app sidebar accepts an optional OpenAI API key.

- If provided, the underwriting agent and treaty Q&A use OpenAI for richer responses.
- If blank, the app falls back to local rule-based answers and document retrieval.

To run with an environment key:

```bash
export OPENAI_API_KEY="your_api_key_here"
./run.sh
```

## Quick terminal commands

```bash
# install dependencies
python -m pip install -r requirements.txt

# generate data
python 01_generate_data.py

# run Streamlit
streamlit run app/streamlit_app.py

# run FastAPI backend
uvicorn api.09_api:app --host 127.0.0.1 --port 8000 --reload

# run both together
./run.sh
```

## Notes

- The app uses generated CSV files in `data/`.
- `app/streamlit_app.py` imports the UI code from `app/10_streamlit_app.py`.
- The backend and UI are designed to work with the current local project dataset.

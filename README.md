# Regulatory Knowledge Graph Pipeline (reg_kg) 🚀

## 1. Project Vision

This project's mission is to build a **multimodal Knowledge Graph (KG)** from financial regulatory documents (e.g., from BaFin, ECB, Basel).

We aim to create a "digital twin" of the regulatory landscape. Instead of just modeling the text, our graph will also incorporate and describe the charts, tables, and formulas within these documents. This will allow us to ask complex questions and find not just relevant text snippets, but the *exact artifacts* (visuals, data, calculations) that provide evidence for an answer.

Our ultimate goal is to build a **GraphRAG** (Graph Retrieval-Augmented Generation) system. This advanced technique uses the structured knowledge in our KG to improve the accuracy and relevance of Large Language Models (LLMs). It will allow us to ask natural language questions and get answers that are "grounded" in the verified facts and relationships from our graph, reducing hallucinations and providing traceable, verifiable responses.

---

## 2. Quick Start (How to Develop)

Follow these steps to set up your development environment:

1. **Clone the Repo:** Get the project code onto your machine (or Azure environment).

    ```bash
    git clone [your-repo-url]
    cd reg_kg
    ```

    * **What is Git?**  
      Git is the industry standard for *version control*. It lets us track changes to our code, collaborate without conflicts, and revert mistakes easily.

2. **Create the Environment:** We use **Conda** for managing Python packages and environments.

    ```bash
    conda env create -f environment.yml
    conda activate reg_kg_env
    ```

    * **What is Conda?**  
      Conda is a package and environment manager. It creates isolated spaces for projects so dependencies do not conflict.

3. **Run the Pipeline:** We use **Typer** to create a clean command-line interface (CLI) for running pipeline steps. All commands are executed from `src/main.py`.

    ```bash
    # See all available commands and their descriptions
    python src/main.py --help

    # Test environment & CLI
    python src/main.py hello "YourName"

    # Run a specific pipeline step (placeholder)
    python src/main.py run-epic-1
    ```

    * **What is Typer?**  
      Typer simplifies building professional CLIs, allowing us to trigger pipeline components cleanly.

---

## 3. Project Structure (The "Project Tour") 🗺️

This project is structured as a modular pipeline (assembly line). Each "Epic" is a major processing stage.

```
reg_kg/                      ← Project Root
├─ .git/                     ← Git internals (hidden)
├─ .gitignore                ← Files Git should ignore
├─ data/                     ← All project data
│  ├─ raw_pdfs/              ← INPUT: Original regulatory PDFs
│  ├─ processed/             ← INTERMEDIATE: Structured outputs like JSONL
│  └─ artifacts/             ← INTERMEDIATE: Extracted images/tables
├─ ontology/                 ← Knowledge Graph blueprints (e.g., FIBO)
│  └─ fibo.owl
├─ src/                      ← All pipeline code
│  ├─ __init__.py
│  ├─ main.py                ← CLI entrypoint (run this!)
│  └─ pipeline/
│     ├─ __init__.py
│     └─ epic_1_parser.py    ← Epic 1: PDF parsing
├─ utils/                    ← Shared helpers (logging, DB, etc.)
├─ notebooks/                ← Jupyter experiments / exploratory work
├─ tests/                    ← Automated tests (pytest)
├─ environment.yml           ← Conda environment spec
└─ README.md                 ← This file
```

---

## 4. Key Libraries (What They Are & Why We Use Them)

| Library | Purpose | Used In |
|--------|---------|---------|
| **PyMuPDF (`fitz`)** | High-performance PDF text + layout + image extraction | Epics 1 & 7 |
| **spaCy** | NLP processing for entity and relation extraction | Epics 4 & 5 |
| **rdflib** | RDF triple + ontology management for Knowledge Graphs | Epics 6 & 9 |
| **pandas** | Tabular data manipulation (e.g., PDF tables → KG text) | Epics 7 & 8 |
| **Typer** | CLI orchestration | Project-wide |
| **JupyterLab** | Experimental analysis environment | `/notebooks` |
| **pytest** | Test automation | `/tests` |

---

### Definition of Done (Checklist)

- [ ] A new Git repository is created and the `README.md` content above is committed.
- [ ] The `environment.yml` file is in the project root.
- [ ] A new team member can clone the repo and build the environment using `conda env create -f environment.yml`.
- [ ] The member can activate the environment (`conda activate reg_kg_env`) and run:
      ```bash
      python src/main.py hello "Test"
      ```
- [ ] The complete folder structure (including `.gitkeep` files for empty folders) is committed to `main`.
- [ ] `.gitignore` includes standard Python ignores and environment-specific ignores.


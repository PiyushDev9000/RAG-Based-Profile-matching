# RAG-Based Resume Matcher

Match job descriptions to resumes using semantic search.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

## Run

**Step 1 — Generate resumes:**

```bash
python generate_resumes.py
```

**Step 2 — Build RAG pipeline:**

```bash
python resume_rag.py
```

**Step 3 — Match jobs:**

```bash
python job_matcher.py
```

## Project Structure

```
├── generate_resumes.py   # Creates synthetic PDF resumes
├── resume_rag.py         # Loads, chunks, embeds, stores in ChromaDB
├── job_matcher.py        # Matches job descriptions to resumes
├── job_descriptions/     # Add .txt job description files here
├── resumes/              # Auto-generated after Step 1
└── chroma_db/            # Auto-generated after Step 2
```

## Add Your Own Resumes

Drop any real PDF resume into `/resumes` before running `resume_rag.py`.

## Add Your Own Job Description

Add a `.txt` file to `/job_descriptions/` with the role and required skills.

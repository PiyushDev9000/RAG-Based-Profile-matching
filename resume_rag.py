import os
import re
import pypdf
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Configuration ───────────────────────────────────────────────────────────
RESUMES_DIR = "resumes"
JD_DIR = "job_descriptions"
CHROMA_DB_DIR = "chroma_db"        # ChromaDB will create this folder
COLLECTION_NAME = "resumes"        # Name of our collection inside ChromaDB
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Extended section headers — covers both synthetic & real resume formats
SECTION_HEADERS = [
    # Standard
    "SKILLS", "EXPERIENCE", "EDUCATION", "SUMMARY", "OBJECTIVE",
    # Real resume variants
    "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "TECHNICAL SKILLS",
    "CORE SKILLS", "KEY SKILLS", "PROJECTS", "PERSONAL PROJECTS",
    "CERTIFICATIONS", "ACHIEVEMENTS", "AWARDS", "PUBLICATIONS",
    "VOLUNTEER", "LANGUAGES", "INTERESTS", "ABOUT ME", "PROFILE"
]

# ── PART 1: PDF LOADER ───────────────────────────────────────────────────────

def load_pdf(pdf_path):
    """
    Extract raw text from a PDF file.
    Handles both synthetic and real-world PDFs.
    Returns a single string of all text across all pages.
    """
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        
        if reader.is_encrypted:
            print(f"  🔒 Skipping encrypted PDF: {pdf_path}")
            return None
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                
    except Exception as e:
        print(f"  ⚠️  Error reading {pdf_path}: {e}")
        return None
    
    cleaned = text.strip()
    
    if len(cleaned) < 100:
        print(f"  ⚠️  Very little text extracted from {pdf_path}")
        print(f"      This might be a scanned/image-based PDF.")
        return None
    
    return cleaned


def load_all_resumes(resumes_dir=RESUMES_DIR):
    """
    Load all PDF resumes from the resumes directory.
    Returns a list of dicts with path and raw text.
    """
    resumes = []
    pdf_files = list(Path(resumes_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in /{resumes_dir}")
        return []
    
    print(f"📂 Found {len(pdf_files)} resumes in /{resumes_dir}\n")
    
    skipped = 0
    for pdf_path in pdf_files:
        print(f"  📄 Loading: {pdf_path.name}")
        raw_text = load_pdf(pdf_path)
        
        if raw_text:
            resumes.append({
                "path": str(pdf_path),
                "filename": pdf_path.name,
                "raw_text": raw_text
            })
        else:
            skipped += 1
    
    print(f"\n✅ Successfully loaded : {len(resumes)} resumes")
    if skipped > 0:
        print(f"⚠️  Skipped            : {skipped} resumes")
    print()
    return resumes

# ── PART 2: SMART CHUNKER ────────────────────────────────────────────────────

def normalize_header(line):
    return line.strip().upper()


def detect_sections(text):
    sections = {}
    current_section = "HEADER"
    current_content = []
    
    for line in text.split("\n"):
        line_stripped = line.strip()
        normalized = normalize_header(line_stripped)
        
        matched_header = None
        for header in SECTION_HEADERS:
            if (normalized == header or
                normalized.startswith(header + ":") or
                normalized.startswith(header + " ")):
                matched_header = header
                break
        
        if matched_header:
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = matched_header
            current_content = []
        else:
            if line_stripped:
                current_content.append(line_stripped)
    
    if current_content:
        sections[current_section] = "\n".join(current_content).strip()
    
    return sections


def split_long_section(text, section_name, max_length=800):
    words = text.split()
    chunks = []
    chunk_size = max_length // 5
    overlap = 20
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    
    return chunks


def chunk_resume(resume):
    sections = detect_sections(resume["raw_text"])
    chunks = []
    
    for section_name, section_content in sections.items():
        if not section_content.strip():
            continue
        
        if len(section_content) > 1000:
            sub_chunks = split_long_section(section_content, section_name)
            for i, sub in enumerate(sub_chunks):
                chunks.append({
                    "section": section_name,
                    "content": sub,
                    "source_file": resume["filename"],
                    "source_path": resume["path"],
                    "chunk_id": f"{resume['filename']}_{section_name}_{i}"
                })
        else:
            chunks.append({
                "section": section_name,
                "content": section_content,
                "source_file": resume["filename"],
                "source_path": resume["path"],
                "chunk_id": f"{resume['filename']}_{section_name}"
            })
    
    return chunks


def chunk_all_resumes(resumes):
    all_chunks = []
    for resume in resumes:
        chunks = chunk_resume(resume)
        all_chunks.extend(chunks)
        sections_found = list(dict.fromkeys([c['section'] for c in chunks]))
        print(f"  ✂️  {resume['filename']:<35} → {len(chunks)} chunks {sections_found}")
    
    print(f"\n✅ Total chunks created: {len(all_chunks)}\n")
    return all_chunks

# ── PART 3: METADATA EXTRACTOR ───────────────────────────────────────────────

def extract_name(text):
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "@" in line:
            continue
        if re.match(r'^\+?[\d\s\-\(\)]+$', line):
            continue
        if len(line) > 60:
            continue
        if len(line.split()) < 2:
            continue
        if any(header in line.upper() for header in SECTION_HEADERS):
            continue
        return line.strip()
    return "Unknown"


def extract_skills(sections):
    skills_text = (
        sections.get("SKILLS") or
        sections.get("TECHNICAL SKILLS") or
        sections.get("CORE SKILLS") or
        sections.get("KEY SKILLS") or
        ""
    )
    if not skills_text:
        return []
    cleaned = skills_text.replace("•", ",").replace("\n", ",").replace("|", ",")
    skills = [s.strip() for s in cleaned.split(",") if s.strip() and len(s.strip()) > 1]
    return skills


def extract_experience_years(sections):
    exp_text = (
        sections.get("EXPERIENCE") or
        sections.get("WORK EXPERIENCE") or
        sections.get("PROFESSIONAL EXPERIENCE") or
        ""
    )
    if not exp_text:
        return 0
    year_pattern = r'(\d{4})\s*[-–]\s*(\d{4}|Present|Current|Now)'
    matches = re.findall(year_pattern, exp_text, re.IGNORECASE)
    total_years = 0
    for start, end in matches:
        start_year = int(start)
        end_year = 2024 if end.lower() in ["present", "current", "now"] else int(end)
        total_years += max(0, end_year - start_year)
    return total_years


def extract_education(sections):
    edu_text = sections.get("EDUCATION", "")
    if not edu_text:
        return "Unknown"
    for line in edu_text.split("\n"):
        line = line.strip()
        if line and len(line) > 5:
            return line
    return "Unknown"


def extract_metadata(resume):
    text = resume["raw_text"]
    sections = detect_sections(text)
    return {
        "name": extract_name(text),
        "skills": extract_skills(sections),
        "skills_str": ", ".join(extract_skills(sections)),
        "experience_years": extract_experience_years(sections),
        "education": extract_education(sections),
        "resume_path": resume["path"],
        "filename": resume["filename"]
    }


def extract_all_metadata(resumes):
    metadata_store = {}
    for resume in resumes:
        metadata = extract_metadata(resume)
        metadata_store[resume["filename"]] = metadata
        print(f"  🏷️  {metadata['name']:<28} | "
              f"Skills: {len(metadata['skills']):<3} | "
              f"Exp: {metadata['experience_years']} yrs | "
              f"Edu: {metadata['education'][:30]}")
    print(f"\n✅ Metadata extracted for {len(metadata_store)} resumes\n")
    return metadata_store

# ── PART 4: EMBEDDING MODEL ──────────────────────────────────────────────────

def load_embedding_model():
    """
    Load the sentence transformer embedding model.
    First run downloads ~80MB model, then cached locally.
    all-MiniLM-L6-v2 converts any text → 384 dimensional vector
    """
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL}")
    print(f"   (First run downloads ~80MB — please wait...)")
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"✅ Embedding model loaded successfully\n")
    return model


def generate_embedding(model, text):
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def generate_embeddings_for_chunks(model, chunks):
    """Generate embeddings for all chunks"""
    
    texts = [chunk["content"] for chunk in chunks]
    
    print(f"🔢 Generating embeddings for {len(texts)} chunks...")
    print(f"   Model: all-MiniLM-L6-v2 → 384 dimensions per chunk\n")
    
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    # Convert numpy array to Python list for ChromaDB
    embeddings = embeddings.tolist()
    
    # Attach embedding to each chunk
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]
    
    print(f"\n✅ Embeddings generated: {len(embeddings)} vectors")
    print(f"   Each vector has {len(embeddings[0])} dimensions")
    
    return chunks

# ── PART 5: CHROMADB SETUP & STORAGE ─────────────────────────────────────────

def setup_chromadb():
    """
    Initialize ChromaDB client and create/load the resumes collection.
    
    ChromaDB stores data in a local folder (chroma_db/).
    If the collection already exists, it loads it.
    If not, it creates a fresh one.
    """
    print(f"🗄️  Setting up ChromaDB...")
    print(f"   Storage location: /{CHROMA_DB_DIR}\n")
    
    # Create persistent client — data saved to disk, survives restarts
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # Get or create collection
    # Think of a collection like a table in a regular database
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity for matching
        # Cosine similarity measures the ANGLE between vectors
        # Perfect for semantic similarity — ignores length, focuses on direction/meaning
    )
    
    print(f"✅ ChromaDB ready!")
    print(f"   Collection: '{COLLECTION_NAME}'")
    print(f"   Existing records: {collection.count()}\n")
    
    return client, collection


def store_in_chromadb(collection, chunks, metadata_store):
    """
    Store all chunks with their embeddings and metadata in ChromaDB.
    
    ChromaDB needs 4 things per record:
    1. ids        → unique string ID per chunk
    2. embeddings → the 384-number vector
    3. documents  → original text content
    4. metadatas  → dict of filterable fields (name, skills, exp years etc.)
    """
    print(f"💾 Storing {len(chunks)} chunks in ChromaDB...")
    
    # Check if already populated — avoid duplicates on re-runs
    existing_count = collection.count()
    if existing_count > 0:
        print(f"   ⚠️  Collection already has {existing_count} records.")
        print(f"   🗑️  Clearing and re-indexing for fresh start...\n")
        collection.delete(where={"source_file": {"$ne": ""}})
    
    # Prepare data in the format ChromaDB expects
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        filename = chunk["source_file"]
        
        # Get metadata for this resume
        resume_meta = metadata_store.get(filename, {})
        
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        documents.append(chunk["content"])
        
        # ChromaDB metadata must be flat dict with string/int/float values only
        # No lists allowed — that's why we have skills_str
        metadatas.append({
            "source_file"      : filename,
            "source_path"      : chunk["source_path"],
            "section"          : chunk["section"],
            "candidate_name"   : resume_meta.get("name", "Unknown"),
            "skills_str"       : resume_meta.get("skills_str", ""),
            "experience_years" : resume_meta.get("experience_years", 0),
            "education"        : resume_meta.get("education", "Unknown"),
        })
    
    # Store in batches of 50 to avoid memory issues
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
        print(f"   ✅ Stored batch {i//batch_size + 1}: chunks {i+1} to {batch_end}")
    
    print(f"\n✅ All chunks stored in ChromaDB!")
    print(f"   Total records in DB: {collection.count()}\n")


def verify_retrieval(model, collection):
    """
    Quick test to confirm ChromaDB retrieval is working.
    Runs a sample query and prints top 3 results.
    """
    print("🔍 VERIFICATION: Testing retrieval with sample query...")
    print("-" * 50)
    
    test_query = "Python developer with machine learning experience"
    print(f"   Query: '{test_query}'\n")
    
    # Convert query to embedding
    query_embedding = generate_embedding(model, test_query)
    
    # Search ChromaDB for top 3 most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    print("   Top 3 Results:")
    for i in range(len(results["ids"][0])):
        candidate = results["metadatas"][0][i]["candidate_name"]
        section   = results["metadatas"][0][i]["section"]
        distance  = results["distances"][0][i]
        similarity = round((1 - distance) * 100, 1)  # Convert distance to similarity %
        content_preview = results["documents"][0][i][:80]
        
        print(f"\n   [{i+1}] {candidate} — {section} section")
        print(f"       Similarity : {similarity}%")
        print(f"       Preview    : {content_preview}...")
    
    print("\n✅ Retrieval working correctly!\n")

# ── PART 6: MAIN PIPELINE ────────────────────────────────────────────────────

def process_resumes():
    """
    Full pipeline — Checkpoints 2 & 3 combined:
    1. Load PDFs
    2. Chunk by sections
    3. Extract metadata
    4. Generate embeddings
    5. Store in ChromaDB
    6. Verify retrieval
    """
    print("=" * 65)
    print("📋  RESUME RAG PIPELINE")
    print("=" * 65)
    
    # ── Checkpoint 2 ────────────────────────────────────────
    print("\n🔄 STEP 1: Loading PDFs...")
    resumes = load_all_resumes()
    if not resumes:
        print("❌ No resumes loaded. Check your /resumes folder.")
        return None, None, None, None
    
    print("🔄 STEP 2: Chunking Resumes by Section...")
    all_chunks = chunk_all_resumes(resumes)
    
    print("🔄 STEP 3: Extracting Metadata...")
    metadata_store = extract_all_metadata(resumes)
    
    # ── Checkpoint 3 ────────────────────────────────────────
    print("🔄 STEP 4: Loading Embedding Model...")
    model = load_embedding_model()
    
    print("🔄 STEP 5: Generating Embeddings...")
    all_chunks = generate_embeddings_for_chunks(model, all_chunks)
    
    print("🔄 STEP 6: Setting Up ChromaDB...")
    client, collection = setup_chromadb()
    
    print("🔄 STEP 7: Storing in ChromaDB...")
    store_in_chromadb(collection, all_chunks, metadata_store)
    
    # ── Verification ─────────────────────────────────────────
    verify_retrieval(model, collection)
    
    # ── Final Summary ─────────────────────────────────────────
    print("=" * 65)
    print("📊  FINAL PIPELINE SUMMARY")
    print("=" * 65)
    print(f"  Resumes Loaded       : {len(resumes)}")
    print(f"  Chunks Created       : {len(all_chunks)}")
    print(f"  Embeddings Generated : {len(all_chunks)} × 384 dimensions")
    print(f"  ChromaDB Records     : {collection.count()}")
    print(f"  DB Location          : /{CHROMA_DB_DIR}")
    print("=" * 65)
    print("\n🎉 RAG Pipeline Ready! You can now run job_matcher.py")
    
    return model, collection, all_chunks, metadata_store


if __name__ == "__main__":
    model, collection, chunks, metadata = process_resumes()

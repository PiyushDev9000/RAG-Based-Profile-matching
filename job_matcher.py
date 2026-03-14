import os
import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ── Import our existing pipeline ────────────────────
from resume_rag import (
    load_all_resumes,
    chunk_all_resumes,
    extract_all_metadata,
    RESUMES_DIR
)

# ── Configuration ────────────────────────────────────
JD_DIR              = "job_descriptions"
CHROMA_DB_PATH      = "chroma_db"
COLLECTION_NAME     = "resumes"
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
TOP_K               = 10        # retrieve top 10 chunks
OUTPUT_DIR          = "results"
SEMANTIC_WEIGHT     = 0.70      # 70% semantic score
KEYWORD_WEIGHT      = 0.30      # 30% keyword score


# ══════════════════════════════════════════════════════
# PART 1: SETUP & LOADERS
# ══════════════════════════════════════════════════════

def load_embedding_model():
    """Load the same embedding model used in resume_rag.py"""
    print("🤖 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Model loaded\n")
    return model


def load_chromadb():
    """Connect to existing ChromaDB collection"""
    print("🗄️  Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"✅ Connected! Collection has {collection.count()} records\n")
        return collection
    except Exception:
        print("❌ ChromaDB collection not found!")
        print("   Please run resume_rag.py first to build the database.")
        return None


def load_job_description(jd_path):
    """
    Load a job description from a .txt file.
    Returns the raw text content.
    """
    try:
        with open(jd_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"❌ Error loading JD: {e}")
        return None


def list_job_descriptions(jd_dir=JD_DIR):
    """List all available job description files"""
    jd_files = list(Path(jd_dir).glob("*.txt"))
    return jd_files


# ══════════════════════════════════════════════════════
# PART 2: SKILL EXTRACTION FROM JD
# ══════════════════════════════════════════════════════

# Common tech & business skills to look for in JDs
KNOWN_SKILLS = [
    # Programming
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "Node.js", "React", "Angular", "Vue",
    # Data & ML
    "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch",
    "Scikit-learn", "Pandas", "NumPy", "SQL", "Spark", "Tableau",
    "Data Visualization", "Statistics", "Keras",
    # Backend & DevOps
    "Docker", "Kubernetes", "AWS", "GCP", "Azure", "CI/CD", "Jenkins",
    "Terraform", "Ansible", "Linux", "REST APIs", "Microservices",
    "PostgreSQL", "MongoDB", "Redis", "Kafka",
    # Marketing
    "SEO", "Content Marketing", "Google Analytics", "HubSpot",
    "Email Marketing", "Social Media", "Copywriting", "Campaign Management",
    # Finance
    "Financial Modeling", "Excel", "Bloomberg Terminal", "Risk Analysis",
    "Forecasting", "Valuation", "PowerBI", "Accounting",
    # Product
    "Agile", "Scrum", "Jira", "Product Roadmap", "A/B Testing",
    "User Research", "Wireframing", "Stakeholder Management",
]


def extract_skills_from_jd(jd_text):
    """
    Extract required skills mentioned in the job description.
    Matches against our known skills list (case-insensitive).
    Returns a list of matched skills.
    """
    jd_lower = jd_text.lower()
    matched_skills = []
    
    for skill in KNOWN_SKILLS:
        if skill.lower() in jd_lower:
            matched_skills.append(skill)
    
    return matched_skills


def extract_required_experience(jd_text):
    """
    Extract minimum years of experience from JD.
    Looks for patterns like '3+ years', '2 years', '5+ years experience'
    Returns integer (0 if not found)
    """
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'(\d+)\+?\s*years?\s*experience',
        r'experience\s*of\s*(\d+)\+?\s*years?',
        r'minimum\s*(\d+)\+?\s*years?',
        r'at\s*least\s*(\d+)\+?\s*years?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, jd_text.lower())
        if match:
            return int(match.group(1))
    
    return 0  # No requirement found


# ══════════════════════════════════════════════════════
# PART 3: SEMANTIC SEARCH
# ══════════════════════════════════════════════════════

def embed_query(model, text):
    """
    Convert job description text into an embedding vector.
    Uses the same model as resume embeddings — this is what
    makes comparison possible.
    """
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def semantic_search(model, collection, jd_text, top_k=TOP_K):
    """
    Search ChromaDB for resume chunks most similar to the JD.
    
    How it works:
    1. Convert JD to embedding
    2. ChromaDB finds chunks with closest vectors
    3. Returns top_k most similar chunks with similarity scores
    """
    print(f"   🔍 Running semantic search (top {top_k} chunks)...")
    
    # Convert JD to embedding
    jd_embedding = embed_query(model, jd_text)
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[jd_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # ChromaDB returns distances (lower = more similar)
    # Convert to similarity scores (higher = more similar)
    chunks = []
    for i in range(len(results["ids"][0])):
        distance   = results["distances"][0][i]
        similarity = round((1 - distance) * 100, 2)  # Convert to 0-100 scale
        
        chunks.append({
            "chunk_id"    : results["ids"][0][i],
            "content"     : results["documents"][0][i],
            "metadata"    : results["metadatas"][0][i],
            "similarity"  : similarity,
            "source_file" : results["metadatas"][0][i].get("source_file", ""),
            "section"     : results["metadatas"][0][i].get("section", ""),
            "name"        : results["metadatas"][0][i].get("name", "Unknown"),
        })
    
    return chunks


# ══════════════════════════════════════════════════════
# PART 4: KEYWORD BOOSTING
# ══════════════════════════════════════════════════════

def keyword_score(candidate_text, required_skills):
    """
    Calculate keyword match score for a candidate.
    
    How it works:
    - Check how many required JD skills appear in candidate's resume text
    - Score = (matched skills / total required skills) × 100
    
    Why needed?
    - Semantic search understands meaning but might miss exact skill names
    - "Must have Docker" → we explicitly check if Docker is mentioned
    """
    if not required_skills:
        return 0, []
    
    candidate_lower = candidate_text.lower()
    matched = []
    
    for skill in required_skills:
        if skill.lower() in candidate_lower:
            matched.append(skill)
    
    score = round((len(matched) / len(required_skills)) * 100, 2)
    return score, matched


# ══════════════════════════════════════════════════════
# PART 5: DEDUPLICATION & CANDIDATE AGGREGATION
# ══════════════════════════════════════════════════════

def aggregate_by_candidate(chunks, metadata_store, required_skills):
    """
    Multiple chunks may belong to the same candidate.
    This function:
    1. Groups chunks by candidate (source_file)
    2. Takes the best semantic score per candidate
    3. Combines all chunk text for keyword scoring
    4. Returns one entry per candidate
    """
    candidates = {}
    
    for chunk in chunks:
        source_file = chunk["source_file"]
        
        if source_file not in candidates:
            # Get metadata for this candidate
            meta = metadata_store.get(source_file, {})
            
            candidates[source_file] = {
                "source_file"      : source_file,
                "name"             : meta.get("name", chunk["name"]),
                "resume_path"      : meta.get("resume_path", source_file),
                "skills"           : meta.get("skills", []),
                "experience_years" : meta.get("experience_years", 0),
                "education"        : meta.get("education", "Unknown"),
                "best_similarity"  : chunk["similarity"],
                "all_text"         : chunk["content"],
                "relevant_chunks"  : [chunk],
            }
        else:
            # Update best similarity score
            if chunk["similarity"] > candidates[source_file]["best_similarity"]:
                candidates[source_file]["best_similarity"] = chunk["similarity"]
            
            # Accumulate all text for keyword matching
            candidates[source_file]["all_text"] += " " + chunk["content"]
            candidates[source_file]["relevant_chunks"].append(chunk)
    
    # Now compute keyword score using all accumulated text
    for source_file, candidate in candidates.items():
        kw_score, matched_skills = keyword_score(
            candidate["all_text"],
            required_skills
        )
        candidate["keyword_score"]   = kw_score
        candidate["matched_skills"]  = matched_skills
    
    return candidates


# ══════════════════════════════════════════════════════
# PART 6: FINAL SCORING
# ══════════════════════════════════════════════════════

def compute_final_score(candidate, required_experience):
    """
    Compute final 0-100 match score for a candidate.
    
    Formula:
    Final Score = (Semantic × 70%) + (Keyword × 30%)
    
    Penalty:
    If candidate experience < required → small penalty applied
    """
    semantic = candidate["best_similarity"]
    keyword  = candidate["keyword_score"]
    
    # Weighted combination
    final = (semantic * SEMANTIC_WEIGHT) + (keyword * KEYWORD_WEIGHT)
    
    # Experience penalty (max -10 points)
    if required_experience > 0:
        candidate_exp = candidate.get("experience_years", 0)
        if candidate_exp < required_experience:
            gap     = required_experience - candidate_exp
            penalty = min(gap * 2, 10)  # 2 points per year gap, max 10
            final   = max(0, final - penalty)
    
    return round(final, 2)


def generate_reasoning(candidate, required_skills, required_experience):
    """
    Generate a human-readable explanation of why this candidate matched.
    """
    reasons = []
    
    # Skill match reasoning
    matched  = candidate["matched_skills"]
    total_req = len(required_skills)
    
    if matched:
        reasons.append(
            f"Matches {len(matched)}/{total_req} required skills: "
            f"{', '.join(matched[:5])}"
            f"{'...' if len(matched) > 5 else ''}"
        )
    
    # Experience reasoning
    exp = candidate.get("experience_years", 0)
    if exp > 0:
        if required_experience > 0 and exp >= required_experience:
            reasons.append(
                f"Has {exp} years experience (meets {required_experience}+ requirement)"
            )
        elif required_experience > 0 and exp < required_experience:
            reasons.append(
                f"Has {exp} years experience (below {required_experience}+ requirement)"
            )
        else:
            reasons.append(f"Has {exp} years of experience")
    
    # Semantic match reasoning
    sim = candidate["best_similarity"]
    if sim >= 50:
        reasons.append(f"Strong semantic similarity ({sim}%) to job description")
    elif sim >= 35:
        reasons.append(f"Moderate semantic similarity ({sim}%) to job description")
    else:
        reasons.append(f"Low semantic similarity ({sim}%) to job description")
    
    # Education
    edu = candidate.get("education", "")
    if edu and edu != "Unknown":
        reasons.append(f"Education: {edu}")
    
    return ". ".join(reasons)


# ══════════════════════════════════════════════════════
# PART 7: MAIN MATCHING ENGINE
# ══════════════════════════════════════════════════════

def match_job_description(model, collection, metadata_store, jd_path):
    """
    Full matching pipeline for one job description.
    
    Steps:
    1. Load JD text
    2. Extract required skills & experience
    3. Semantic search → top 10 chunks
    4. Aggregate by candidate
    5. Score each candidate
    6. Sort & return top results
    """
    print(f"\n{'='*65}")
    print(f"🎯 MATCHING: {Path(jd_path).name}")
    print(f"{'='*65}")
    
    # Step 1: Load JD
    jd_text = load_job_description(jd_path)
    if not jd_text:
        return None
    
    print(f"\n📋 Job Description Preview:")
    print(f"   {jd_text[:150]}...\n")
    
    # Step 2: Extract requirements
    required_skills     = extract_skills_from_jd(jd_text)
    required_experience = extract_required_experience(jd_text)
    
    print(f"   📌 Required Skills Found  : {required_skills}")
    print(f"   📅 Required Experience    : {required_experience}+ years\n")
    
    # Step 3: Semantic search
    chunks = semantic_search(model, collection, jd_text, top_k=TOP_K)
    print(f"   ✅ Retrieved {len(chunks)} chunks\n")
    
    # Step 4: Aggregate by candidate
    print("   🔀 Aggregating by candidate & computing keyword scores...")
    candidates = aggregate_by_candidate(chunks, metadata_store, required_skills)
    print(f"   ✅ {len(candidates)} unique candidates found\n")
    
    # Step 5: Score each candidate
    print("   🏆 Computing final scores...")
    scored = []
    for source_file, candidate in candidates.items():
        final_score = compute_final_score(candidate, required_experience)
        reasoning   = generate_reasoning(candidate, required_skills, required_experience)
        
        # Get relevant excerpts (best matching chunk content)
        excerpts = []
        for chunk in candidate["relevant_chunks"][:2]:
            excerpts.append(chunk["content"][:200])
        
        scored.append({
            "candidate_name"    : candidate["name"],
            "resume_path"       : candidate["resume_path"],
            "match_score"       : final_score,
            "semantic_score"    : candidate["best_similarity"],
            "keyword_score"     : candidate["keyword_score"],
            "matched_skills"    : candidate["matched_skills"],
            "experience_years"  : candidate["experience_years"],
            "education"         : candidate["education"],
            "relevant_excerpts" : excerpts,
            "reasoning"         : reasoning
        })
    
    # Step 6: Sort by final score
    scored.sort(key=lambda x: x["match_score"], reverse=True)
    
    return {
        "job_description"       : jd_text,
        "jd_file"               : Path(jd_path).name,
        "required_skills"       : required_skills,
        "required_experience"   : required_experience,
        "total_candidates"      : len(scored),
        "top_matches"           : scored
    }


# ══════════════════════════════════════════════════════
# PART 8: OUTPUT & DISPLAY
# ══════════════════════════════════════════════════════

def display_results(results, top_n=5):
    """Pretty print top N matches to terminal"""
    
    if not results:
        return
    
    print(f"\n{'='*65}")
    print(f"🏆 TOP {top_n} MATCHES FOR: {results['jd_file']}")
    print(f"{'='*65}")
    print(f"Required Skills : {', '.join(results['required_skills'][:6])}")
    print(f"Required Exp    : {results['required_experience']}+ years")
    print(f"Total Candidates: {results['total_candidates']}")
    print(f"{'='*65}\n")
    
    for i, match in enumerate(results["top_matches"][:top_n], 1):
        print(f"[#{i}] {match['candidate_name']}")
        print(f"      Final Score    : {match['match_score']}/100")
        print(f"      Semantic       : {match['semantic_score']}%")
        print(f"      Keyword Match  : {match['keyword_score']}%")
        print(f"      Experience     : {match['experience_years']} years")
        print(f"      Education      : {match['education'][:50]}")
        print(f"      Matched Skills : {', '.join(match['matched_skills'][:5])}")
        print(f"      Reasoning      : {match['reasoning'][:120]}...")
        print(f"      Resume         : {match['resume_path']}")
        print()


def save_results(results, output_dir=OUTPUT_DIR):
    """Save results as JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    jd_name    = Path(results["jd_file"]).stem
    output_path = os.path.join(output_dir, f"{jd_name}_matches.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved → {output_path}")
    return output_path


# ══════════════════════════════════════════════════════
# PART 9: MAIN — RUN ALL JDs
# ══════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("🚀 JOB MATCHER — RAG Based Profile Matching")
    print("=" * 65)
    
    # ── Setup ────────────────────────────────────────
    model      = load_embedding_model()
    collection = load_chromadb()
    
    if not collection:
        return
    
    # ── Load metadata from resumes ───────────────────
    print("📋 Loading resume metadata...")
    resumes        = load_all_resumes()
    metadata_store = extract_all_metadata(resumes)
    
    # ── Get all JD files ─────────────────────────────
    jd_files = list_job_descriptions()
    
    if not jd_files:
        print(f"❌ No .txt files found in /{JD_DIR}")
        return
    
    print(f"\n📂 Found {len(jd_files)} job descriptions:")
    for jd in jd_files:
        print(f"   - {jd.name}")
    
    # ── Process each JD ──────────────────────────────
    all_results = []
    
    for jd_path in jd_files:
        results = match_job_description(
            model, collection, metadata_store, jd_path
        )
        
        if results:
            display_results(results, top_n=5)
            output_path = save_results(results)
            all_results.append(results)
    
    # ── Final Summary ─────────────────────────────────
    print("\n" + "=" * 65)
    print("✅ ALL JOB DESCRIPTIONS PROCESSED")
    print("=" * 65)
    print(f"   JDs Processed : {len(all_results)}")
    print(f"   Results saved : /{OUTPUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
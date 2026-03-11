import chromadb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Get all embeddings
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("resumes")
data = collection.get(include=["embeddings", "metadatas"])

embeddings = np.array(data["embeddings"])
sections   = [m.get("section", "") for m in data["metadatas"]]
files      = [m.get("source_file", "") for m in data["metadatas"]]

# Filter only SKILLS chunks for cleaner visualization
idx = [i for i, s in enumerate(sections) if s == "SKILLS"]
skill_embeddings = embeddings[idx]
skill_files      = [files[i] for i in idx]

# Reduce 384 dimensions → 2 dimensions using t-SNE
print("Running t-SNE... (takes ~10 seconds)")
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
reduced = tsne.fit_transform(skill_embeddings)

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=100)

# Label each dot with filename
for i, fname in enumerate(skill_files):
    label = fname.replace(".pdf", "").replace("_", " ").title()
    plt.annotate(label, (reduced[i, 0], reduced[i, 1]),
                 fontsize=7, alpha=0.8)

plt.title("Resume Embeddings Visualized (t-SNE)\nEach dot = one resume's SKILLS section")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.savefig("embeddings_visualization.png", dpi=150)
plt.show()
print("✅ Saved as embeddings_visualization.png")
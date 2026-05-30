import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
col = client.get_or_create_collection("papers")
count = col.count()
print(f"Papers in ChromaDB: {count}")

if count > 0:
    results = col.get(limit=count, include=["metadatas"])
    for meta in results["metadatas"]:
        source = meta.get("source", "?")
        title = meta.get("title", "?")[:80]
        year = meta.get("year", "?")
        doi = meta.get("doi", "")
        print(f"  [{source}] {title} ({year}) doi={doi}".encode("ascii", errors="replace").decode())

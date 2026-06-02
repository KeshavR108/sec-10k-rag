def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks


def chunk_sections(sections, company, doc_id):
    all_chunks = []
    chunk_index = 0

    for section_name, section_text in sections.items():
        if not section_text.strip():
            continue

        text_chunks = chunk_text(section_text)

        for text in text_chunks:
            all_chunks.append({
                "id": f"{doc_id}_chunk_{chunk_index}",
                "doc_id": doc_id,
                "company": company,
                "section": section_name,
                "text": text
            })
            chunk_index += 1

    return all_chunks

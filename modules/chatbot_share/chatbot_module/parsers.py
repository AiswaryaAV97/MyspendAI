# modules/parser.py
import pdfplumber, pandas as pd, os, re

def clean_text(s): return re.sub(r'\s+', ' ', s or '').strip()

def split_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks=[]
    i=0
    while i < len(words):
        chunks.append(clean_text(' '.join(words[i:i+chunk_size])))
        i += chunk_size - overlap
    return chunks

def parse_pdf(path):
    chunks=[]
    with pdfplumber.open(path) as pdf:
        for pnum, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            for idx, c in enumerate(split_text(txt)):
                chunks.append({"text": c, "page": pnum, "chunk_index": idx, "source": os.path.basename(path)})
    return chunks

def parse_excel(path):
    df = pd.read_excel(path, engine="openpyxl")
    chunks=[]
    for r_idx, row in df.iterrows():
        merged = ' '.join([str(x) for x in row.values if pd.notna(x)])
        for idx,c in enumerate(split_text(merged)):
            chunks.append({"text": c, "row": r_idx, "chunk_index": idx, "source": os.path.basename(path)})
    return chunks

def parse_document(file_obj):
    """
    Parse a document (PDF, Excel, or text file) from a file object.
    Returns a list of text chunks.
    """
    if isinstance(file_obj, str):
        # If it's a file path string
        path = file_obj
    else:
        # If it's a file upload object from Flask
        filename = file_obj.filename
        path = os.path.join("/tmp", filename)  # Save to temp location
        file_obj.save(path)
    
    # Determine file type and parse accordingly
    if path.endswith('.pdf'):
        return parse_pdf(path)
    elif path.endswith(('.xlsx', '.xls')):
        return parse_excel(path)
    elif path.endswith('.txt'):
        # Simple text file parsing
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            chunks = split_text(text)
            return [{"text": c, "chunk_index": idx, "source": os.path.basename(path)} for idx, c in enumerate(chunks)]
    else:
        raise ValueError(f"Unsupported file type: {path}")

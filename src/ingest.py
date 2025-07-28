import fitz  # PyMuPDF
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import unicodedata
import re
from typing import List, Dict, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentIngestionPipeline:
    """Optimized document ingestion pipeline."""

    def __init__(self, collection_path: str, model_path: str = './models/intfloat-multilingual-e5-small'):
        self.collection_path = Path(collection_path)
        self.pdf_dir = self.collection_path / "PDFs"
        self.embeddings_file = self.collection_path / "collection_embeddings.npy"
        self.metadata_file = self.collection_path / "collection_metadata.json"
        self.faiss_index_file = self.collection_path / "faiss_index.bin"

        # ✅ Load model from local path for offline use
        self.model = SentenceTransformer(model_path)
        self.chunks = []

        self.collection_path.mkdir(exist_ok=True)

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text)
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        return text.strip()

    def is_likely_heading(self, text: str, font_size: float, is_bold: bool, avg_font_size: float) -> bool:
        if len(text.split()) > 15:
            return False
        heading_patterns = [
            r'^[A-Z][A-Z\s]+$', r'^\d+\.?\s+[A-Z]', r'^[IVX]+\.?\s+[A-Z]', r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$'
        ]
        for pattern in heading_patterns:
            if re.match(pattern, text.strip()):
                return True
        return font_size > avg_font_size * 1.1 or is_bold

    def extract_and_chunk_content(self, pdf_path: Path) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        logger.info(f"Processing: {pdf_path.name}")

        all_text_blocks = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict", sort=True)["blocks"]
            for block_idx, block in enumerate(blocks):
                if not block.get("lines"):
                    continue
                text_parts = []
                font_size = 12
                is_bold = False
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_parts.append(span.get("text", ""))
                        font_size = span.get("size", 12)
                        is_bold = span.get("flags", 0) & 2**4 > 0
                text = self.clean_text(" ".join(text_parts))
                if len(text.strip()) >= 10:
                    all_text_blocks.append({
                        "text": text,
                        "page_num": page_num + 1,
                        "block_idx": block_idx,
                        "font_size": font_size,
                        "is_bold": is_bold,
                        "word_count": len(text.split())
                    })
        doc.close()
        return self._create_intelligent_chunks(pdf_path, all_text_blocks)

    def _create_intelligent_chunks(self, pdf_path: Path, text_blocks: List[Dict]) -> List[Dict[str, Any]]:
        if not text_blocks:
            return []

        avg_font_size = np.mean([block["font_size"] for block in text_blocks])
        chunks = []
        current_section = "Introduction"
        current_chunk_text = []
        current_chunk_pages = set()
        chunk_word_count = 0

        MAX_CHUNK_WORDS = 300
        MIN_CHUNK_WORDS = 50

        for block in text_blocks:
            text = block["text"]
            page_num = block["page_num"]
            is_heading = self.is_likely_heading(text, block["font_size"], block["is_bold"], avg_font_size)

            if is_heading:
                if current_chunk_text and chunk_word_count >= MIN_CHUNK_WORDS:
                    chunk = self._create_chunk(pdf_path, current_chunk_text, current_section, current_chunk_pages, len(chunks))
                    chunks.append(chunk)
                current_section = text[:100]
                current_chunk_text = []
                current_chunk_pages = set()
                chunk_word_count = 0
                continue

            current_chunk_text.append(text)
            current_chunk_pages.add(page_num)
            chunk_word_count += block["word_count"]

            if chunk_word_count >= MAX_CHUNK_WORDS:
                combined_text = " ".join(current_chunk_text)
                sentences = combined_text.split('. ')
                if len(sentences) > 1:
                    mid_point = len(sentences) // 2
                    first_half = '. '.join(sentences[:mid_point]) + '.'
                    second_half = '. '.join(sentences[mid_point:])
                    chunk = self._create_chunk_from_text(pdf_path, first_half, current_section, current_chunk_pages, len(chunks))
                    chunks.append(chunk)
                    current_chunk_text = [second_half] if second_half.strip() else []
                    chunk_word_count = len(second_half.split()) if second_half.strip() else 0
                else:
                    chunk = self._create_chunk(pdf_path, current_chunk_text, current_section, current_chunk_pages, len(chunks))
                    chunks.append(chunk)
                    current_chunk_text = []
                    current_chunk_pages = set()
                    chunk_word_count = 0

        if current_chunk_text and chunk_word_count >= MIN_CHUNK_WORDS:
            chunk = self._create_chunk(pdf_path, current_chunk_text, current_section, current_chunk_pages, len(chunks))
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} intelligent chunks from {pdf_path.name}")
        return chunks

    def _create_chunk(self, pdf_path: Path, text_parts: List[str], section_title: str, page_numbers: set, chunk_index: int) -> Dict[str, Any]:
        combined_text = " ".join(text_parts)
        return self._create_chunk_from_text(pdf_path, combined_text, section_title, page_numbers, chunk_index)

    def _create_chunk_from_text(self, pdf_path: Path, text: str, section_title: str, page_numbers: set, chunk_index: int) -> Dict[str, Any]:
        primary_page = min(page_numbers) if page_numbers else 1
        return {
            "document": pdf_path.name,
            "page_number": primary_page,
            "section_title": section_title,
            "content": text.strip(),
            "word_count": len(text.split()),
            "chunk_id": f"{pdf_path.stem}_chunk_{chunk_index}",
            "pages_spanned": sorted(list(page_numbers)) if len(page_numbers) > 1 else [primary_page]
        }

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        embeddings_normalized = embeddings.astype('float32')
        faiss.normalize_L2(embeddings_normalized)
        index.add(embeddings_normalized)
        logger.info(f"Created FAISS index with {index.ntotal} vectors")
        return index

    def process_documents(self) -> Dict[str, Any]:
        start_time = datetime.now()
        logger.info(f"Starting document ingestion for: {self.collection_path.name}")

        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.pdf_dir}")
        logger.info(f"Found {len(pdf_files)} PDF files")

        all_chunks = []
        for pdf_path in pdf_files:
            chunks = self.extract_and_chunk_content(pdf_path)
            all_chunks.extend(chunks)

        logger.info(f"Total chunks extracted: {len(all_chunks)}")
        self.chunks = all_chunks

        logger.info("Generating embeddings...")
        content_texts = [chunk["content"] for chunk in self.chunks]
        embeddings = self.model.encode(content_texts, show_progress_bar=False, convert_to_numpy=True, batch_size=32).astype('float32')

        logger.info("Creating FAISS index...")
        faiss_index = self.create_faiss_index(embeddings)

        self._save_artifacts(embeddings, faiss_index, pdf_files)

        processing_time = (datetime.now() - start_time).total_seconds()
        summary = {
            "total_documents": len(pdf_files),
            "total_chunks": len(self.chunks),
            "processing_time_seconds": processing_time,
            "average_chunk_length": np.mean([chunk["word_count"] for chunk in self.chunks]),
            "embedding_dimension": embeddings.shape[1]
        }
        logger.info(f"✅ Ingestion completed in {processing_time:.2f} seconds")
        return summary

    def _save_artifacts(self, embeddings: np.ndarray, faiss_index: faiss.Index, pdf_files: List[Path]):
        np.save(self.embeddings_file, embeddings)
        faiss.write_index(faiss_index, str(self.faiss_index_file))
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved artifacts to {self.collection_path}")

def main():
    COLLECTION_PATH = "collection1"
    MODEL_PATH = "./models/intfloat-multilingual-e5-small"

    try:
        pipeline = DocumentIngestionPipeline(COLLECTION_PATH, model_path=MODEL_PATH)
        summary = pipeline.process_documents()

        print(f"\n📊 Ingestion Summary:")
        print(f"Documents: {summary['total_documents']}")
        print(f"Chunks: {summary['total_chunks']}")
        print(f"Processing Time: {summary['processing_time_seconds']:.2f}s")
        print(f"Avg Chunk Length: {summary['average_chunk_length']:.1f} words")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()

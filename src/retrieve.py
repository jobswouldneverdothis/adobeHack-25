import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePersonaQueryPipeline:
    def __init__(
        self,
        collection_path: str = "collection1",
        retriever_path: str = "./models/intfloat-multilingual-e5-small",
        reranker_path: str = "./models/qnli-distilroberta-base"
    ):
        self.collection_path = Path(collection_path)
        self.retriever_path = retriever_path
        self.reranker_path = reranker_path

        self.input_file = self.collection_path / "challenge1b_input.json"
        self.output_file = self.collection_path / "challenge1b_output.json"
        self.metadata_file = self.collection_path / "collection_metadata.json"
        self.faiss_index_file = self.collection_path / "faiss_index.bin"

        self.retriever_model = None
        self.reranker_model = None
        self.faiss_index = None
        self.metadata = []

        self.top_k_retrieval = 150
        self.top_n_final = 20

    def load_everything(self):
        logger.info("Loading models and data from local paths...")


        self.retriever_model = SentenceTransformer(self.retriever_path)
        self.reranker_model = CrossEncoder(self.reranker_path)


        self.faiss_index = faiss.read_index(str(self.faiss_index_file))
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded {len(self.metadata)} chunks, FAISS index ready")

    def create_persona_job_query(self, persona_role: str, job_task: str) -> str:
        query = (
            f"As a {persona_role}, I need to {job_task}. "
            "Please include attractions, cultural sites, local cuisine, group-friendly activities, accommodations, and essential tips."
        )
        logger.info(f"Created query: {query}")
        return query

    def search_with_faiss(self, query: str, top_k: int = 150) -> List[Dict]:
        query_embedding = self.retriever_model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype='float32')

        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.metadata):
                doc = self.metadata[idx].copy()
                doc['faiss_score'] = float(score)
                results.append(doc)

        logger.info(f"FAISS found {len(results)} relevant chunks")

        for i, doc in enumerate(results[:5]):
            logger.info(f"FAISS #{i+1}: {doc['document']} - {doc['section_title']} (Score: {doc['faiss_score']:.4f})")

        return results

    def rerank_results(self, query: str, faiss_results: List[Dict], top_n: int = 20) -> List[Dict]:
        if not faiss_results:
            return []

        rerank_pairs = [[query, doc['content']] for doc in faiss_results]
        rerank_scores = self.reranker_model.predict(rerank_pairs, show_progress_bar=True)

        for doc, score in zip(faiss_results, rerank_scores):
            doc['rerank_score'] = float(score)
            doc['final_score'] = float(score)

        reranked = sorted(faiss_results, key=lambda x: x['final_score'], reverse=True)
        seen_documents = set()
        final_results = []

        for doc in reranked:
            if doc['document'] not in seen_documents:
                final_results.append(doc)
                seen_documents.add(doc['document'])
            if len(final_results) == top_n:
                break

        if len(final_results) < top_n:
            logger.warning(f"Only {len(final_results)} unique documents found out of requested top {top_n}.")

        logger.info(f"Reranked to {len(final_results)} final results")
        return final_results

    def generate_output(self, input_data: Dict, final_results: List[Dict]) -> Dict:
        persona_role = input_data["persona"]["role"]
        job_task = input_data["job_to_be_done"]["task"]
        input_docs = [doc["filename"] for doc in input_data["documents"]]

        output = {
            "metadata": {
                "input_documents": input_docs,
                "persona": persona_role,
                "job_to_be_done": job_task,
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_analyzed": len(self.metadata),
                "sections_returned": len(final_results)
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

        for rank, result in enumerate(final_results, 1):
            output["extracted_sections"].append({
                "document": result["document"],
                "section_title": result["section_title"],
                "importance_rank": rank,
                "page_number": result["page_number"],
                "relevance_score": round(result["final_score"], 4)
            })

            output["subsection_analysis"].append({
                "document": result["document"],
                "refined_text": result["content"],
                "page_number": result["page_number"],
                "section_title": result["section_title"],
                "importance_rank": rank
            })

        return output

    def run_pipeline(self) -> Dict:
        start_time = datetime.now()
        logger.info("Starting persona-driven document analysis...")

        with open(self.input_file, 'r') as f:
            input_data = json.load(f)

        self.load_everything()
        persona_role = input_data["persona"]["role"]
        job_task = input_data["job_to_be_done"]["task"]

        query = self.create_persona_job_query(persona_role, job_task)
        faiss_results = self.search_with_faiss(query, top_k=self.top_k_retrieval)
        final_results = self.rerank_results(query, faiss_results, top_n=self.top_n_final)
        output_data = self.generate_output(input_data, final_results)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Pipeline completed in {processing_time:.2f} seconds")

        return output_data

def main():
    COLLECTION_PATH = "collection1"
    RETRIEVER_PATH = "./models/intfloat-multilingual-e5-small"
    RERANKER_PATH = "./models/qnli-distilroberta-base"

    try:
        pipeline = SimplePersonaQueryPipeline(COLLECTION_PATH, RETRIEVER_PATH, RERANKER_PATH)
        results = pipeline.run_pipeline()

        metadata = results["metadata"]
        sections = results["extracted_sections"]

        print(f"\n\U0001F4CA Results Summary:")
        print(f"Persona: {metadata['persona']}")
        print(f"Task: {metadata['job_to_be_done']}")
        print(f"Documents: {len(metadata['input_documents'])}")
        print(f"Sections Found: {len(sections)}")

        print(f"\n🏆 Top 3 Results:")
        for section in sections[:3]:
            print(f"  {section['importance_rank']}. {section['document']} - "
                  f"{section['section_title']} (Score: {section['relevance_score']:.3f})")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

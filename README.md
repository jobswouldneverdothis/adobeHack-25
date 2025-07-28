# Persona-Based Document Analysis Pipeline

## Project Overview
This project implements a sophisticated document analysis system that uses advanced NLP techniques to process and retrieve relevant information based on specific persona requirements. The system utilizes a two-stage retrieval pipeline combining FAISS-based semantic search and cross-encoder reranking.

## Technical Architecture

### Core Components
1. **Document Retrieval System**
   - FAISS index for efficient similarity search
   - Sentence Transformer model for document embedding
   - Cross-encoder for result reranking

2. **Processing Pipeline**
   - Initial document chunking and embedding
   - Semantic search using FAISS
   - Result reranking using cross-encoder
   - Final output generation with metadata

### Models Used
- Primary Retriever: `intfloat-multilingual-e5-small`
- Reranker: `qnli-distilroberta-base`

## Directory Structure
```
bottle-turtle/
├── src/
│   ├── retrieve.py
│   ├── ingest.py
│   └── utils/
├── data/
│   └── collection1/
│       ├── PDFs/
│       ├── collection_metadata.json
│       └── faiss_index.bin
├── models/
│   ├── intfloat-multilingual-e5-small/
│   └── qnli-distilroberta-base/
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

## Prerequisites
- Docker installed on your system
- Python 3.9+
- Minimum 8GB RAM
- Downloaded model files in `models/` directory

## Setup and Installation

### Using Docker

1. Build the Docker image:
```bash
docker build -t persona-analysis -f docker/Dockerfile .
```

2. Run the ingestion process:
```bash
docker-compose -f docker/docker-compose.yml up ingestion
```

3. Run the retrieval process:
```bash
docker-compose -f docker/docker-compose.yml up retrieval
```

### Manual Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the pipeline:
```bash
python src/ingest.py  # For document ingestion
python src/retrieve.py  # For information retrieval
```

## Usage Guide

### Document Ingestion
1. Place PDF documents in `data/collection1/PDFs/`
2. Run the ingestion process
3. Verify generated files in `data/collection1/`:
   - `collection_metadata.json`
   - `faiss_index.bin`

### Information Retrieval
1. Ensure all required files are present
2. Run the retrieval process
3. Check output in the specified location

## Input/Output Specifications

### Input Format
```json
{
    "persona": "string",
    "requirements": ["string"],
    "context": "string"
}
```

### Output Format
```json
{
    "results": [
        {
            "document_id": "string",
            "section": "string",
            "relevance_score": float,
            "metadata": {}
        }
    ]
}
```

## Performance Considerations
- Processing time scales with document size
- RAM usage depends on collection size
- Batch processing recommended for large collections

## Troubleshooting

### Common Issues
1. **Memory Errors**
   - Reduce batch size in ingest.py
   - Increase Docker container memory limit

2. **Model Loading Errors**
   - Verify model files in `models/` directory
   - Check model compatibility

3. **Docker Issues**
   - Verify volume mounts
   - Check Docker memory allocation

### Error Messages
- `ModuleNotFoundError`: Install missing dependencies
- `RuntimeError: CUDA out of memory`: Reduce batch size
- `FileNotFoundError`: Check file paths and permissions

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For support or queries, please open an issue in the repository.

## Acknowledgments
- FAISS library by Facebook Research
- Sentence Transformers by UKPLab
- HuggingFace
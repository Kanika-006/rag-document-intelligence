
# RAG-Based Document Intelligence System

An end-to-end Retrieval-Augmented Generation (RAG) system that enables users to interact with PDF documents through natural-language questions. The system combines semantic search, keyword-based retrieval, reciprocal rank fusion, and CrossEncoder reranking to retrieve relevant information before generating grounded answers using Google Gemini.

## Key Features

- **PDF Processing:** Extracts text from PDF documents while preserving page-level metadata.
- **Document Chunking:** Splits extracted text into overlapping chunks for effective retrieval.
- **Semantic Search:** Uses Sentence Transformers embeddings and FAISS vector search.
- **Hybrid Retrieval:** Combines dense retrieval with BM25 keyword search.
- **Reciprocal Rank Fusion:** Merges results from multiple retrieval methods.
- **CrossEncoder Reranking:** Reorders retrieved passages based on query-document relevance.
- **Grounded Answer Generation:** Uses Google Gemini to generate responses based on retrieved document context.
- **Source Attribution:** Includes page references to help users verify generated answers.
- **Conversation Support:** Maintains conversation history and handles follow-up questions.
- **Out-of-Context Handling:** Instructs the model to abstain when the retrieved document context does not support an answer.
- **Interactive Interface:** Provides a Streamlit application for uploading PDFs and asking questions.

## System Architecture

```text
                PDF Document
                     |
                     v
                PDF Loader
                     |
                     v
               Text Extraction
                     |
                     v
                Text Chunking
                     |
                     v
              Embedding Model
                     |
                     v
                FAISS Index
                     |
          +----------+----------+
          |                     |
          v                     v
     FAISS Search           BM25 Search
          |                     |
          +----------+----------+
                     |
                     v
          Reciprocal Rank Fusion
                     |
                     v
             Candidate Retrieval
                     |
                     v
             CrossEncoder Reranking
                     |
                     v
             Relevant Context
                     |
                     v
              Gemini LLM
                     |
                     v
         Grounded Answer + Sources
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| PDF Extraction | PyPDF |
| Text Chunking | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Keyword Retrieval | BM25 |
| Ranking Fusion | Reciprocal Rank Fusion |
| Reranking | CrossEncoder |
| Language Model | Google Gemini |
| User Interface | Streamlit |
| Evaluation | Precision@K, Recall@K, MRR, Hit Rate |

## Project Structure

```text
rag-document-intelligence/
|
├── data/
│   └── sample_documents/
│       └── sample.pdf
|
├── src/
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── query_preprocessor.py
│   └── rag_pipeline.py
|
├── tests/
│   ├── test_pdf_loader.py
│   ├── test_embeddings.py
│   ├── test_gemini.py
│   ├── test_rag_pipeline.py
│   ├── test_retrieval_evaluation.py
│   ├── test_retriever.py
│   ├── test_retriever_comparison.py
│   ├── test_text_splitter.py
│   ├── test_vector_store.py
│   └── test_answer_evaluation.py
|
├── app.py
├── evaluation_dataset.py
├── evaluate_metrics.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Retrieval Evaluation

The retrieval pipeline was evaluated using 14 question-page relevance pairs from the sample PDF.

Three retrieval configurations were compared:

1. FAISS-only retrieval
2. Hybrid retrieval using FAISS and BM25
3. Hybrid retrieval with CrossEncoder reranking

### Evaluation Results

| Metric | FAISS Only | Hybrid | Hybrid + Reranking |
|---|---:|---:|---:|
| Precision@3 | 0.3571 | 0.3333 | 0.3571 |
| Recall@3 | 0.8571 | 0.8214 | 0.8571 |
| MRR | 0.7381 | 0.7500 | 1.0000 |
| Hit Rate@3 | 0.8571 | 0.9286 | 1.0000 |

**Key observation:** Hybrid retrieval with CrossEncoder reranking achieved an MRR of 1.00 and a Hit Rate@3 of 1.00 on this evaluation set, indicating that a relevant page appeared first for every evaluated query and at least one relevant page appeared in the top three for every query.

These are preliminary results from a small evaluation set and a single PDF. They should not be interpreted as general performance guarantees.

## Answer Quality Evaluation

The answer-generation pipeline was also tested using in-document and out-of-document questions.

The evaluation checks:

- Answer correctness
- Faithfulness to retrieved context
- Answer completeness
- Citation accuracy
- Appropriate refusal of unsupported questions

The initial evaluation uses a small manually reviewed set of questions. Results are preliminary and should be expanded with additional documents and test cases.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Kanika-006/rag-document-intelligence.git
cd rag-document-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_configured_gemini_model
```

Never upload your `.env` file or expose your API key.

### 5. Run the application

```bash
streamlit run app.py
```

Upload a PDF and start asking questions about its contents.

## Running Tests

Run individual tests from the project root:

```bash
python -m tests.test_pdf_loader
python -m tests.test_retrieval_evaluation
python -m tests.test_answer_evaluation
```

Run the evaluation metrics script:

```bash
python evaluate_metrics.py
```


## Author

Kanika 

Electronics and Computer Engineering  
Thapar Institute of Engineering and Technology

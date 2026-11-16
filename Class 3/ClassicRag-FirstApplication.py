"""
CLASSIC RAG - Simple Educational Implementation

Architecture:

User Question
      ↓
Embedding model
      ↓
Query vector
      ↓
Similarity search
      ↓
Retrieve relevant DOCUMENT TEXT
      ↓
Question + Document TEXT
      ↓
LLM
      ↓
Answer

Dependencies:
    pip install ollama


Models:
    Embedding: nomic-embed-text
    LLM:       mistral
"""
# Dependencies you should install: 
# python -m pip install ollama
# ollama pull nomic-embed-text
#If you don't have Mistral yet, also run: ollama pull mistral

import math
import ollama
import openrouter


# ============================================================
# CONFIGURATION
# ============================================================
# Version1
EMBEDDING_MODEL = "nomic-embed-text"
# Version2
#EMBEDDING_MODEL = "all-minilm"
# Version3
#EMBEDDING_MODEL = "bge-m3"
# Version4
#EMBEDDING_MODEL = "mxbai-embed-large"




LLM_MODEL = "mistral"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

TOP_K = 3


# ============================================================
# 1. KNOWLEDGE BASE
# ============================================================

def create_knowledge_base():
    """
    Creates two small documents.

    In a real application these could come from:
    - PDF files
    - Word documents
    - Web pages
    - Database records
    - Markdown files
    """

    documents = {

        "smart_energy.txt": """
Smart energy systems use digital technologies to monitor,
control and optimize energy consumption.

A smart meter measures electricity consumption at regular
intervals and communicates the measurements to an information
system. The collected data can be used to understand consumer
behavior and identify unusual consumption patterns.

Smart buildings can combine smart meters with temperature
sensors, occupancy sensors and photovoltaic systems. These
components allow the building management system to optimize
energy consumption while maintaining user comfort.

Machine learning can also be used in smart energy systems.
For example, an anomaly detection model can identify unusual
electricity consumption and help detect equipment failures.
""",

        "rag.txt": """
Retrieval-Augmented Generation, commonly called RAG, combines
information retrieval with a generative language model.

In a classical RAG system, documents are first divided into
smaller chunks. An embedding model converts each chunk into a
vector representation. These vectors are stored in a vector
database or another vector store.

When a user asks a question, the question is also converted
into a vector. A similarity search compares the question vector
with the document vectors and retrieves the most relevant
chunks.

The retrieved document text is then added to the user question
as context. The language model receives both the question and
the retrieved context and generates an answer.

The main advantage of RAG is that the language model can use
external knowledge without requiring that knowledge to be
included in the model's original training data.
"""
    }

    return documents


# ============================================================
# 2. CHUNKING
# ============================================================

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split a document into overlapping chunks.

    Example:

        Document
        ↓
        Chunk 1
        Chunk 2
        Chunk 3

    The overlap helps preserve information between chunks.
    IT helps avoid losing meaning when an important sentence/concept is split between two chunks.
    
    Bigger chunks are not necessarily better :

    Too small:

    You lose context.

    Too large:

    Retrieval becomes less precise because the chunk contains several different topics.

    So the objective is:

    The smallest chunk that contains enough information to answer the intended question.
    """

    text = " ".join(text.split())

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        # Move forward while keeping an overlap
        start = end - overlap

    return chunks


def build_chunks(documents):
    """
    Create chunks for all documents.

    Returns a list containing:
        - document name
        - chunk id
        - chunk text
    """

    all_chunks = []

    for document_name, text in documents.items():

        chunks = create_chunks(text)

        for i, chunk in enumerate(chunks):

            all_chunks.append({
                "document": document_name,
                "chunk_id": i,
                "text": chunk
            })

    return all_chunks


# ============================================================
# 3. EMBEDDING
# ============================================================

def create_embedding(text):
    """
    Convert text into a numerical vector using Ollama.
    """

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    # Ollama returns a list of embeddings.
    return response["embeddings"][0]


# ============================================================
# 4. BUILD VECTOR STORE
# ============================================================

def build_vector_store(chunks):
    """
    Create embeddings for every chunk.

    We keep everything in Python memory.

    In a production system this could be replaced by:
        Chroma
        FAISS
        Qdrant
        Milvus
        etc.
    """

    vector_store = []

    for chunk in chunks:

        print(
            f"Embedding {chunk['document']} "
            f"- chunk {chunk['chunk_id']}"
        )

        vector = create_embedding(chunk["text"])

        vector_store.append({
            "document": chunk["document"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "vector": vector
        })

    return vector_store


# ============================================================
# 5. COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.

    cosine_similarity =

        A · B
        -------
        |A| |B|

    The value is generally between -1 and 1.
    Higher = more semantically similar.
    """

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


# ============================================================
# 6. RETRIEVAL
# ============================================================

def retrieve(query, vector_store, top_k=TOP_K):
    """
    Classic RAG retrieval:

        User question
             ↓
        Query embedding
             ↓
        Similarity search
             ↓
        Top-K chunks
    """

    print("\nCreating query embedding...")

    query_vector = create_embedding(query)

    results = []

    for item in vector_store:

        score = cosine_similarity(
            query_vector,
            item["vector"]
        )

        results.append({
            "document": item["document"],
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "score": score
        })

    # Sort by similarity
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# 7. AUGMENTATION
# ============================================================

def build_augmented_prompt(question, retrieved_chunks):
    """
    This is the AUGMENTATION step.

    IMPORTANT:

    We do NOT add the vectors together.

    Instead:

        Question
             +
        Retrieved DOCUMENT TEXT
             ↓
        Augmented prompt
    """

    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are a helpful assistant.

Answer the user's question using ONLY the
information contained in the context below.

If the answer cannot be found in the context,
say:

"I don't have enough information in the knowledge base."

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

USER QUESTION:

{question}

Answer:
"""

    return prompt


# ============================================================
# 8. GENERATION
# ============================================================

def generate_answer(prompt):
    """
    Send the augmented prompt to the local Mistral model.
    """

    response = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt
    )

    return response["response"]


# ============================================================
# 9. DISPLAY RETRIEVED DOCUMENTS
# ============================================================

def display_retrieved_chunks(results):
    """
    Display what the RAG system actually retrieved.
    """

    print("\n")
    print("=" * 70)
    print("RETRIEVED DOCUMENT CHUNKS")
    print("=" * 70)

    for i, result in enumerate(results):

        print(f"\nResult #{i + 1}")
        print(f"Document : {result['document']}")
        print(f"Chunk    : {result['chunk_id']}")
        print(f"Score    : {result['score']:.4f}")

        print("\nText:")
        print(result["text"])


# ============================================================
# 10. MAIN RAG PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("CLASSIC RAG DEMONSTRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1 - Create documents
    # --------------------------------------------------------

    documents = create_knowledge_base()

    print("\nKnowledge base:")
    print(f"{len(documents)} documents")

    # --------------------------------------------------------
    # STEP 2 - Chunk documents
    # --------------------------------------------------------

    print("\nCreating chunks...")

    chunks = build_chunks(documents)

    print(f"Created {len(chunks)} chunks.")

    # Show chunks to students
    print("\n" + "=" * 70)
    print("CHUNKS")
    print("=" * 70)

    for chunk in chunks:

        print(
            f"\n[{chunk['document']}] "
            f"Chunk {chunk['chunk_id']}"
        )

        print(chunk["text"])

    # --------------------------------------------------------
    # STEP 3 - Embed documents
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING DOCUMENT EMBEDDINGS")
    print("=" * 70)

    vector_store = build_vector_store(chunks)

    print(
        f"\nVector store contains "
        f"{len(vector_store)} vectors."
    )

    # --------------------------------------------------------
    # STEP 4 - Ask a question
    # --------------------------------------------------------

    question = input(
        "\nAsk a question about the knowledge base: "
    )

    # --------------------------------------------------------
    # STEP 5 - Retrieval
    # --------------------------------------------------------

    results = retrieve(
        question,
        vector_store,
        TOP_K
    )

    # --------------------------------------------------------
    # STEP 6 - Display retrieved text
    # --------------------------------------------------------

    display_retrieved_chunks(results)

    # --------------------------------------------------------
    # STEP 7 - AUGMENTATION
    # --------------------------------------------------------

    augmented_prompt = build_augmented_prompt(
        question,
        results
    )

    print("\n")
    print("=" * 70)
    print("AUGMENTED PROMPT SENT TO MISTRAL")
    print("=" * 70)

    print(augmented_prompt)

    # --------------------------------------------------------
    # STEP 8 - GENERATION
    # --------------------------------------------------------

    print("\nGenerating answer with Mistral...")

    answer = generate_answer(
        augmented_prompt
    )

    # --------------------------------------------------------
    # STEP 9 - FINAL ANSWER
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(answer)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
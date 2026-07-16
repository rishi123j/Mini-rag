import numpy as np

# ---- Document class ----
class Document:
    def __init__(self, id, title, content, source):
        self.id = id
        self.title = title
        self.content = content
        self.source = source
        self.vector = np.random.rand(5)  # fake embedding

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source": self.source
        }

# ---- Fake document database ----
DOCUMENTS = [
    Document(
        id=1,
        title="Introduction to RAG",
        content="RAG combines retrieval with generation for accurate answers",
        source="AI_paper.pdf"
    ),
    Document(
        id=2,
        title="LangChain Framework",
        content="LangChain simplifies building LLM applications with tools",
        source="LangChain_docs.pdf"
    ),
    Document(
        id=3,
        title="Vector Databases",
        content="Vector databases store embeddings for semantic search",
        source="VectorDB_guide.pdf"
    ),
    Document(
        id=4,
        title="Fine Tuning LLMs",
        content="Fine tuning adapts pretrained models for specific tasks",
        source="ML_guide.pdf"
    ),
    Document(
        id=5,
        title="AI Agents",
        content="AI agents use tools and reasoning to complete complex tasks",
        source="Agents_paper.pdf"
    )
]

def get_all_documents():
    return DOCUMENTS

def get_document_by_id(doc_id):
    for doc in DOCUMENTS:
        if doc.id == doc_id:
            return doc
    return None

import os
import uuid
import logging
import requests
import certifi
import psycopg
import nltk
import json

from transformers import AutoTokenizer
from langchain.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain_postgres import PostgresChatMessageHistory
from langchain.vectorstores.pgvector import PGVector

# === SSL FIX ===
os.environ['SSL_CERT_FILE'] = certifi.where()

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configuration ===
PDF_PATH = "testpdf.pdf"
LOCAL_TOKENIZER_PATH = r"C:\Users\maabdelr\Downloads\downloaded\downloaded"
EMBEDDING_URL = "http://localhost:1234/v1/embeddings"
MODEL_NAME = "qwen3-4b"
SESSION_ID = str(uuid.uuid4())
PG_CONN_STRING = "postgresql+psycopg://postgres:20102004@localhost:5432/chat_history"
TABLE_NAME = "chat_history"

# === NLTK Setup ===
nltk.download('punkt')
tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH)

# === Hierarchical Chunking ===
def hierarchical_chunking(text, tokenizer):
    sections = text.split("\n\n")
    hierarchy = []
    for section_id, section in enumerate(sections):
        paragraphs = section.strip().split("\n")
        for para_id, para in enumerate(paragraphs):
            sentences = nltk.sent_tokenize(para)
            for sent_id, sentence in enumerate(sentences):
                tokens = tokenizer.tokenize(sentence)
                hierarchy.append({
                    "section_id": section_id,
                    "paragraph_id": para_id,
                    "sentence_id": sent_id,
                    "text": sentence,
                    "tokens": tokens,
                    "token_count": len(tokens)
                })
    return hierarchy

# === Load PDF ===
loader = PyMuPDFLoader(PDF_PATH)
documents = loader.load()
raw_text = "\n\n".join([doc.page_content for doc in documents])
chunked_data = hierarchical_chunking(raw_text, tokenizer)
print(f"chunked data.len: {len(chunked_data)}" )
texts = [chunk["text"] for chunk in chunked_data]

# === Embed Texts via LM Studio ===
embedding_payload = {
    "input": texts,
    "model": "nomic-embed-text-v1"
}
response = requests.post(
    EMBEDDING_URL,
    headers={"Content-Type": "application/json"},
    json=embedding_payload
)
response.raise_for_status()
embedding_data = response.json()["data"]
vectors = [item["embedding"] for item in embedding_data]

# === Custom Embedding Class ===
class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return vectors
    def embed_query(self, text):
        query_response = requests.post(
            EMBEDDING_URL,
            headers={"Content-Type": "application/json"},
            json={"input": [text], "model": "nomic-embed-text-v1"}
        )
        query_response.raise_for_status()
        return query_response.json()["data"][0]["embedding"]

embedding_function = CustomEmbeddings()

# === Store in Custom Table ===
with psycopg.connect("dbname=chat_history user=postgres password=20102004 host=localhost port=5432") as conn:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY,
                text TEXT,
                embedding VECTOR(768),
                metadata JSONB
            )
        """)
        conn.commit()

        for text, embedding in zip(texts, vectors):
            doc_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO documents (id, text, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                """,
                (doc_id, text, embedding, json.dumps({}))
            )
        conn.commit()

# === Convert to LangChain Documents ===
docs = [Document(page_content=text) for text in texts]

# === Store in PGVector for Retrieval ===
vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=embedding_function,
    collection_name="documents",
    connection_string=PG_CONN_STRING
)

# === Connect to LM Studio Chat Model ===
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed",
    temperature=0.7
)

# === PostgreSQL Chat History ===
sync_connection = psycopg.connect(
    conninfo="dbname=chat_history user=postgres password=20102004 host=localhost port=5432"
)
chat_history = PostgresChatMessageHistory(
    TABLE_NAME,
    SESSION_ID,
    sync_connection=sync_connection
)

# === Contextualization Prompt ===
contextualize_q_system_prompt = (
    "You are a helpful assistant that reformulates and decomposes user queries. "
    "Given a chat history and the latest user question, which might reference context in the chat history, "
    "perform the following steps:\n"
    "1. Reformulate the question into a standalone version that can be understood without chat history.\n"
    "2. If the question is complex, decompose it into simpler sub-questions.\n"
    "3. Return a list of strings in the following format:\n"
    "   - First item: Original user query\n"
    "   - Second item: Refined standalone version\n"
    "   - Remaining items (up to 8): Decomposed sub-questions\n"
    "Limit the total number of items in the list to 10.\n"
    "Return only the list in valid JSON format."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
retriever = vectorstore.as_retriever()
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# === QA Prompt ===
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise if the user asks a question not in the context reply with idk.\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain.schema import HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

import json
from langchain.schema import HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

import json
from langchain.schema import HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)

def get_response(user_query):
    user_chat_history = chat_history.messages

    # Step 1: Get structured output from contextualization
    structured_output = llm.invoke(contextualize_q_prompt.format_messages(
        chat_history=user_chat_history,
        input=user_query
    ))

    try:
        raw = structured_output.content.strip()
        parsed = json.loads(raw)

        if not isinstance(parsed, list) or len(parsed) < 2:
            raise ValueError("Parsed output is not a valid list with at least two items.")

        refined_query = parsed[1]
        decomposed_queries = parsed[2:]

        print("\nStructured Query JSON:")
        print(json.dumps(parsed, indent=4))

    except Exception as e:
        logger.error(f"Failed to parse structured output: {e}")
        print("Raw LLM Output:", getattr(structured_output, "content", "No content"))
        refined_query = user_query
        decomposed_queries = []

    # Step 2: Retrieve documents for refined + decomposed queries
    all_queries = [refined_query] + decomposed_queries
    retrieved_docs = []
    for q in all_queries:
        retrieved_docs.extend(history_aware_retriever.invoke({
            "input": q,
            "chat_history": user_chat_history
        }))

    # Step 3: Remove duplicates
    unique_docs = {doc.page_content: doc for doc in retrieved_docs}.values()

    # Step 4: Run QA chain
    result = question_answer_chain.invoke({
        "input": refined_query,
        "chat_history": user_chat_history,
        "context": list(unique_docs)
    })

    # Step 5: Save to history
    chat_history.add_message(HumanMessage(content=user_query))
    chat_history.add_message(AIMessage(content=result))

    return result


def get_chat_history():
    return chat_history.messages

# === Test Run ===
if __name__ == "__main__":
    print(get_response("Summarize it /no_think"))

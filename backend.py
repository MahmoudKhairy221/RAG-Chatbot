# test.py

import requests
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
import logging 

# === CONFIGURATION ===
PDF_PATH = "testpdf.pdf"
EMBEDDING_URL = "http://localhost:1234/v1/embeddings"
MODEL_NAME = "qwen3-4b"
SESSION_ID = "user-session-001"
COLLECTION_NAME = "pdf_embeddings"
PG_CONN_STRING = "postgresql+psycopg2://postgres:20102004@localhost:5432/chat_history"

# === STEP 1: Load PDF ===
loader = PyMuPDFLoader(PDF_PATH)
documents = loader.load()
texts = [doc.page_content for doc in documents]

# === STEP 2: Embed Texts via LM Studio ===
embedding_payload = {
    "input": texts,
    "model": "nomic-embed-text-v1"
}
try:
    response = requests.post(
        EMBEDDING_URL,
        headers={"Content-Type": "application/json"},
        json=embedding_payload
    )
    response.raise_for_status()
    embedding_data = response.json()["data"]
    vectors = [item["embedding"] for item in embedding_data]
except Exception as e:
    raise RuntimeError(f"Embedding failed: {e}")

# === STEP 3: Custom Embedding Class ===
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

# === STEP 4: Store Vectors in pgvector ===
vectorstore = PGVector.from_texts(
    texts=texts,
    embedding=embedding_function,
    collection_name=COLLECTION_NAME,
    connection_string=PG_CONN_STRING
)

# === STEP 5: Connect to LM Studio Chat Model ===
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed",
    temperature=0.7
)

# === STEP 6: PostgreSQL Chat History ===
chat_history = SQLChatMessageHistory(
    session_id=SESSION_ID,
    connection_string=PG_CONN_STRING
)

# === STEP 7: Custom Memory to Store Only 'answer' ===
class AnswerOnlyMemory(ConversationBufferMemory):
    def save_context(self, inputs, outputs):
        if "answer" in outputs:
            outputs = {"answer": outputs["answer"]}
        super().save_context(inputs, outputs)

memory = ConversationBufferMemory(
    # memory_key="chat_history",
    chat_memory=chat_history,
    return_messages=True
)

print(str(memory))


# === STEP 8: Create Conversational Retrieval Chain ===
def get_qa_chain():
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

# === STEP 9: Expose Functions ===
def get_response(user_query):
    chain = get_qa_chain()
    return chain.invoke({"question": user_query})["answer"]

def get_chat_history():
    return chat_history.messages

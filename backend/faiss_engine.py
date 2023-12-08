from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def get_vectorstore(text_chunks):
    """
    Create the vectorstore
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    return vectorstore

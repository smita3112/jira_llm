from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

import time

embeddings_model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
folder_path=""
target_db_path=""


embeddings = HuggingFaceEmbeddings(
    model_name=embeddings_model_name,
    model_kwargs={'device': 'cuda'}
)

def buildDB_from_folder(folder_path, target_db_path, chunk_size=1000, chunk_overlap=100):
    st = time.time()
    loader = DirectoryLoader(folder_path, glob="**/*.htm")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(docs)
    db = FAISS.from_documents(texts, self.hf)
    et = time.time()
    print(et - st)
    db.save_local(target_db_path)

buildDB_from_folder(folder_path, target_db_path)
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.embeddings import HuggingFaceEmbeddings
import requests
import logging

logging.basicConfig(
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)


from config import INFERENCE_API_URL

db_path=""
embeddings_model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


class CustomLLM(LLM):
    temperature: float = 0.1
    min_new_tokens: int = 2
    max_new_tokens: int = 1024
    do_sample: bool = False
    num_beams: int = 1
    repetition_penalty: float = 1.2
    renormalize_logits: bool = True

    @property
    def _llm_type(self) -> str:
        return "h2o_oa_falcon_7b"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        r = requests.post(
            INFERENCE_API_URL,
            json={
                "prompt": prompt,
                "temperature": self.temperature,
                "min_new_tokens": self.min_new_tokens,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "num_beams": self.num_beams,
                "repetition_penalty": self.repetition_penalty,
                "renormalize_logits": self.renormalize_logits
            }
        )
        return r.json()["response"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "temperature": self.temperature,
            "min_new_tokens": self.min_new_tokens,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "renormalize_logits": self.renormalize_logits
        }

embeddings = HuggingFaceEmbeddings(
    model_name=embeddings_model_name,
    model_kwargs={'device': 'cuda'}
)

llm = CustomLLM()
vectordb=FAISS.load_local(db_path, embeddings)

def read_from_db(db_name, query, similarity_threshold, broad_k, k):
    logger.info("Reading from db")
    docs = db_name.similarity_search(query, broad_k)
    ef = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=similarity_threshold, k=k)
    doc_compressed = ef.compress_documents(documents=docs, query=query)
    filtered_docs = [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(doc_compressed)]
    logger.info("Completed reading from db")
    return filtered_docs


def qna_jira(query, similarity_threshold, broad_k, k):
    prompt_template = """Use the following set of documents to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

Documents: {context}

Question: {question}
Helpful Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = LLMChain(llm=llm, prompt=PROMPT)
    docs = read_from_db(
        vectordb,
        query,
        similarity_threshold,
        broad_k,
        k
    )
    input = [{"context": docs, "question": query}]
    try:
        text = chain.apply(input)[0]['text']
    except:
        text = "No match found"

    return text

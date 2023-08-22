# vector db on sentence embedding
##  info:
- Vector DB (https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/faiss)
- Embedding LLM(https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)

## Compute requirement:
- NVIDIA GPU processors with > 4 GB GPU memory recommended

## Steps to deploy local server:
- Pull repo from Git
```commandline
git clone https://github.com/digital-ai/llm-poc.git
cd llm-poc/vectordb/
```
- Install dependencies
```commandline
pip install -r requirements.txt
```
- Start service
```commandline
uvicorn app.main:app --host 0.0.0.0 --port 8800
```
Service URL: http://localhost:8088 \
Swagger API docs URL: http://localhost:8088/docs

## Steps to deploy using docker
### Note: 
Ensure GPU hardware is exposed to docker engine \
Please use [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installation guide for reference
- Pull repo from Git
```commandline
git clone https://github.com/digital-ai/llm-poc.git
cd llm-poc/vectordb/
```
- Build image
```commandline
docker build -t vectordb .
```
- Start service
```commandline
docker run --env-file ./.env -d -p 8800:8800 --runtime=nvidia --gpus all vectordb
```
Service URL: http://localhost:8088 \
Swagger docs URL: http://localhost:8000/docs
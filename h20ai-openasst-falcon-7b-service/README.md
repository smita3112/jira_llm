# H2O AI Falcon 7B model service
## Model info: 
- [Hugging Face Model card](https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2)

## Compute requirement:
- NVIDIA GPU processors with > 16 GB GPU memory
- Internally tested using AWS g5.4xlarge instance with [this](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-2-0-ubuntu-20-04/) Deep Learning AMI

## Steps to deploy local server:
- Pull repo from Git
```commandline
git clone https://github.com/digital-ai/llm-poc.git
cd llm-poc/fa/
```
- Install dependencies
```commandline
pip install -r requirements.txt
```
- Start service
```commandline
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Service URL: http://localhost:8000 \
Swagger API docs URL: http://localhost:8000/docs

## Steps to deploy using docker
### Note: 
Ensure GPU hardware is exposed to docker engine \
Please use [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installation guide for reference
- Pull repo from Git
```commandline
git clone https://github.com/digital-ai/llm-poc.git
cd llm-poc/h20ai-openasst-falcon-7b-service/
```
- Build image
```commandline
docker build -t h20ai-openasst-falcon-7b-service .
```
- Start service
```commandline
docker run -d -p 8000:8000 --runtime=nvidia --gpus all h20ai-openasst-falcon-7b-service
```
Service URL: http://localhost:8000 \
Swagger docs URL: http://localhost:8000/docs

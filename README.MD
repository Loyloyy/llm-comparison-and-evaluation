# LLM Red Teaming and Evaluation Tool

This application provides a user-friendly interface for red teaming, comparing, and evaluating large language models (LLMs). It's designed to help both technical and non-technical users assess model behavior, particularly in response to potentially harmful queries.

## 🌟 Key Features

- **Side-by-side Model Comparison**: Compare responses from two different LLMs in real-time
- **Interactive Web Interface**: Easy-to-use Gradio-based web interface
- **Custom Prompt Templates**: Use or customize prompt templates to evaluate model behavior
- **NVIDIA NeMo Guardrails Integration**: Apply safety guardrails to any model with one click
- **Server Health Monitoring**: Real-time monitoring of available models with health status checks

## 📋 Requirements

- Python 3.8+
- VLLM server(s) running with accessible API endpoints
- Model files for the LLMs you want to evaluate
- NVIDIA NeMo Guardrails library

## 🚀 Getting Started

1. Git clone this repository
2. Build docker image
```
    docker pull vllm/vllm-openai:latest
    docker build -f Dockerfile_frontend -t your-registry/image-name:version .
```
3. Configure your environment variables in a .env file with the help of .env.template:
```
    ENVIRONMENT=DEV # or PROD
    HOME_DEV=/path/to/dev/directory
    HOME_PROD=/path/to/prod/directory
```
4. Make sure your VLLM servers are running and properly configured in the model configuration files. Usage of SGLang servers are also possible, requires minimal changes to this script. :)
```
    docker run --gpus "device=0" --cpus="16" --memory="16g" --shm-size="8gb" \
        --env-file ${pwd}/.env \
        -v ${pwd}:/nfs \
        -p 8001:8001 -it --rm --name app-name \
        vllm/vllm-openai:latest \
        python -m vllm.entrypoints.openai.api_server --host "0.0.0.0" --port "8001" \
        --gpu-memory-utilization 0.5 --model ${pwd}/your-model
```
5. Start the Gradio frontend
```
    docker run --cpus="8" --memory="16g" --gpus "device=1" \
        -v ${pwd}:/nfs -p 7860:7860 -it --rm --name app-name \
        --env-file .env \
        your-registry/image-name:version
```
6. Access the web interface at http://localhost:7860

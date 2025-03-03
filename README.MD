# LLM Comparison

This application was created so that users can better understand the differences between models. 

## 1. Clone this repository

## 2. Create a .env file with the help of the .env.template file

## 3. Build docker image
    
    docker pull vllm/vllm-openai:latest
    docker build -f Dockerfile_frontend -t your-registry/image-name:version .

## 4. Start the vLLM-powered inference API

    docker run --gpus "device=0" --cpus="16" --memory="16g" --shm-size="8gb" \
        --env-file ${pwd}/.env \
        -v ${pwd}:/nfs \
        -p 8001:8001 -it --rm --name app-name \
        vllm/vllm-openai:latest \
        python -m vllm.entrypoints.openai.api_server --host "0.0.0.0" --port "8001" \
        --gpu-memory-utilization 0.5 --model ${pwd}/your-model \

## 5. Start the Gradio frontend
    docker run --cpus="8" --memory="16g" --gpus "device=1" \
        -v ${pwd}:/nfs -p 7860:7860 -it --rm --name app-name \
        --env-file .env \
        your-registry/image-name:version

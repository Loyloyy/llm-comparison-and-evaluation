"""
Server Health Check Module for vLLM Servers - Clean Version
"""

import requests
import threading
import pandas as pd
import time
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vllm_health_check')

def check_vllm_health(api_url: str, port: int, timeout: int = 2) -> bool:
    """
    Check if a vLLM server is healthy by making a request to its health endpoint.
    """
    try:
        url = f"http://{api_url}:{port}/v1/models"
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        # logger.warning(f"Error checking vLLM health for port {port}: {str(e)}")
        return False

def get_healthy_llms(home_path: str, api_url: str) -> Tuple[List[str], Dict[str, int], Dict[str, str]]:
    """
    Check all vLLM servers' health and return lists of healthy ones.
    
    Returns:
        Tuple containing:
            - List of healthy LLM names
            - Dictionary mapping LLM names to their ports
            - Dictionary mapping LLM names to their model IDs
    """
    try:
        vllm_df = pd.read_csv(VLLM_FILE)
        
        healthy_llms = []
        llm_port_map = {}
        llm_model_map = {}
        
        for _, row in vllm_df.iterrows():
            llm_name = row['name']
            port = row['port']
            model_id = row['llm']
            
            if pd.isna(llm_name) or pd.isna(port):
                continue
                
            llm_port_map[llm_name] = port
            llm_model_map[llm_name] = model_id
            
            if check_vllm_health(api_url, port):
                healthy_llms.append(llm_name)
                # logger.info(f"LLM {llm_name} on port {port} is healthy")
            # else:
                # logger.warning(f"LLM {llm_name} on port {port} is not responding")
        
        return healthy_llms, llm_port_map, llm_model_map
    except Exception as e:
        # logger.error(f"Error updating healthy LLM list: {str(e)}")
        return [], {}, {}

# For backward compatibility
update_healthy_llm_list = get_healthy_llms

def start_health_check_process(home_path: str, api_url: str, health_status_elem, dropdown1, dropdown2):
    """
    Start the background health check process.
    
    Args:
        home_path: Path to the home directory
        api_url: Base URL for the vLLM servers
        health_status_elem: Gradio Markdown component for health status
        dropdown1: First Gradio dropdown component
        dropdown2: Second Gradio dropdown component
    """
    health_thread = threading.Thread(
        target=health_check_process,
        args=(api_url, home_path, health_status_elem, dropdown1, dropdown2),
        daemon=True
    )
    health_thread.start()
    return "Health check process started."

def health_check_process(api_url, home_path, health_status_elem, dropdown1, dropdown2):
    """
    Background process to check server health and update UI elements
    """
    global MODEL_ID1, VLLM_API_URL1, MODEL_ID2, VLLM_API_URL2
    
    while True:
        try:
            # Get current time
            current_time = time.strftime('%H:%M:%S')
            
            # Check which models are healthy
            healthy_llms, llm_port_map, llm_model_map = get_healthy_llms(home_path, api_url)
            
            # Update status indicator
            status_emoji = "🟢" if healthy_llms else "🔴"
            health_status_elem.value = f"**Server Status:** {status_emoji} {len(healthy_llms)} models online | Last checked: {current_time}"
            
            # Update dropdown choices to only include healthy models
            dropdown1.choices = healthy_llms
            dropdown2.choices = healthy_llms
            
            # Check if currently selected models are still healthy
            if dropdown1.value not in healthy_llms and healthy_llms:
                dropdown1.value = healthy_llms[0]
                
            if dropdown2.value not in healthy_llms and healthy_llms:
                # Try to select a different model from model1 if possible
                other_models = [m for m in healthy_llms if m != dropdown1.value]
                dropdown2.value = other_models[0] if other_models else healthy_llms[0]
            
        except Exception as e:
            # logger.error(f"Error in health check: {str(e)}")
            pass
        
        # Sleep for 60 seconds before next check
        time.sleep(60)

def refresh_health_status():
    """
    One-time refresh of the health status - can be called from button clicks
    """
    current_time = time.strftime('%H:%M:%S')
    return f"**Server Status:** 🔄 Checking models... | Last checked: {current_time}"
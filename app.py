import os
import io
import re
import sys
import json
import time
import logging
import lancedb
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from typing import Union, Literal, List, Dict, Optional
from lancedb.rerankers import CrossEncoderReranker
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.embeddings.base import Embeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
import numpy as np
from openai import OpenAI


load_dotenv()
HOME_PATH = os.getenv('HOME_PROD')


def obtain_vector_stores():
    folder = f"{HOME_PATH}/location_of_vector_store"
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    sub_folders = [vs.split('.lance')[0] for vs in sub_folders]
    return sub_folders


def change_llm1(new_llm):
    global MODEL_ID1, VLLM_API_URL1, api_url
    vllmFn=f'{HOME_PATH}/location_of_vllm_list.csv'
    vllmDf=pd.read_csv(vllmFn, error_bad_lines=False)
    port = vllmDf[vllmDf['name']==new_llm]['port'].tolist()[0]
    MODEL_ID1 = vllmDf[vllmDf['name']==new_llm]['llm'].tolist()[0]
    VLLM_API_URL1=f'http://{api_url}:{port}/v1/'


def change_llm2(new_llm):
    global MODEL_ID2, VLLM_API_URL2, api_url
    vllmFn=f'{HOME_PATH}/location_of_vllm_list.csv'
    vllmDf=pd.read_csv(vllmFn, error_bad_lines=False)
    port = vllmDf[vllmDf['name']==new_llm]['port'].tolist()[0]
    MODEL_ID2 = vllmDf[vllmDf['name']==new_llm]['llm'].tolist()[0]
    VLLM_API_URL2=f'http://{api_url}:{port}/v1/'


def model_prep():
    parameterFn=f'{HOME_PATH}/location_of_api_parameters.csv'
    parameterDf=pd.read_csv(parameterFn)
    api_url=parameterDf[parameterDf['parameter']=='url']['value'].tolist()[0]
    
    vllmFn=f'{HOME_PATH}/location_of_vllm_list.csv'
    vllmDf=pd.read_csv(vllmFn, error_bad_lines=False)
    llm_list = vllmDf['name'].dropna().tolist()

    port1 = vllmDf[vllmDf['name']=="llama31_8b_instruct"]['port'].tolist()[0]
    MODEL_ID1 = vllmDf[vllmDf['name']=="llama31_8b_instruct"]['llm'].tolist()[0]
    VLLM_API_URL1=f'http://{api_url}:{port1}/v1/'

    port2 = vllmDf[vllmDf['name']=="DeepSeek-R1-Distill-Llama-8B"]['port'].tolist()[0]
    MODEL_ID2 = vllmDf[vllmDf['name']=="DeepSeek-R1-Distill-Llama-8B"]['llm'].tolist()[0]
    VLLM_API_URL2=f'http://{api_url}:{port2}/v1/'

    return api_url, llm_list, MODEL_ID1, VLLM_API_URL1, MODEL_ID2, VLLM_API_URL2


def preparation():
    db = lancedb.connect(f"{HOME_PATH}/location_of_vector_stores")
    tbl = db.open_table("table_name")
    tbl.create_fts_index(['original_content'], replace=True)

    # Create the reranker
    ce_reranker = CrossEncoderReranker(model_name=f"{HOME_PATH}/llm_models/BAAI/bge-reranker-base", column="original_content")

    # Prepare the embedder
    gte_embedding = GTEEmbeddings(model_path=f'{HOME_PATH}/llm_models/Alibaba-NLP/gte-large-en-v1.5', device="cuda")

    # Create the retriever instance
    retriever = CustomLanceDBRetriever.initialize(tbl, ce_reranker, gte_embedding)
    return db, tbl, ce_reranker, gte_embedding, retriever


class GTEEmbeddings(Embeddings):
    def __init__(self, model_path='Alibaba-NLP/gte-large-en-v1.5', device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.device = device

    def embed_documents(self, texts):
        batch_dict = self.tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class CustomLanceDBRetriever(BaseRetriever):
    _tbl = None
    _reranker = None

    @classmethod
    def initialize(cls, db_table, reranker, embedder):
        cls._tbl = db_table
        cls._reranker = reranker
        cls._embedder = embedder
        return cls()

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query to make it more suitable for full-text search
        """
        # Remove or replace special characters that cause issues
        processed = query
        # Replace various dashes and hyphens with spaces
        processed = re.sub(r'[-–—]', ' ', processed)
        # Replace slashes with spaces
        processed = re.sub(r'[/\\]', ' ', processed)
        # Remove other problematic characters
        processed = re.sub(r'[&:,\(\)]', ' ', processed)
        # Normalize whitespace
        processed = ' '.join(processed.split())
        return processed

    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self._tbl is None or self._reranker is None:
            raise ValueError("Retriever not properly initialized.")

        embedded_query = self._embedder.embed_documents([query])[0]
        processed_query = self._preprocess_query(query)

        try:
            # First attempt hybrid search with processed query
            results = (self._tbl.search(
                (embedded_query, processed_query),
                query_type="hybrid",
                vector_column_name="embeddings"
            )
            .limit(30)
            .rerank(reranker=self._reranker)
            .to_pandas())
        except ValueError as e:
            try:
                # If hybrid search fails, try vector-only search
                print(f"Falling back to vector-only search due to error: {str(e)}")
                results = (self._tbl.search(
                    embedded_query,
                    vector_column_name="embeddings"
                )
                .limit(30)
                .rerank(reranker=self._reranker)
                .to_pandas())
            except Exception as e2:
                # If all else fails, use a more basic vector search
                print(f"Falling back to basic vector search due to error: {str(e2)}")
                results = (self._tbl.search(
                    embedded_query,
                    vector_column_name="embeddings"
                )
                .limit(30)
                .to_pandas())

        documents = []
        for _, row in results.head(10).iterrows():
            metadata = {
                "original_content": row["original_content"],
                "summarized_content": row["summarized_content"],
                "embeddings": list(row["embeddings"]),
                "id": row["id"],
                "type": row["type"],
                "doc_name": row["doc_name"],
                "pg_no": row["pg_no"],
                "_relevance_score": row.get("_relevance_score", 0)
            }
            document = Document(
                page_content=row["original_content"],
                metadata=metadata
            )
            documents.append(document)
        return documents


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []

    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc_page_content = doc.page_content
        if looks_like_base64(doc_page_content) and is_image_data(doc_page_content):
            doc = resize_base64_image(doc_page_content, size=(1300, 600))
            b64_images.append(doc)
            # image_base64_list.append(doc)
        else:
            texts.append(doc_page_content)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """
    Join all the contexts into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for b64_image in data_dict["context"]["images"]:

            image_message = {
                "type": "image_url",
                "image_url": b64_image,
            }
            messages.append(image_message)

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            f"""
You are a smart and detail-oriented pre-sales engineer assistant tasked with checking compliance of a company's products with customer requirements.

Your task:
Determine if the company's product is compliant with the customer requirements. Think through this step-by-step and in detail.

Process:
1. Carefully read the customer requirement (question).
2. List all specific criteria mentioned in the requirement.
3. For each criterion, find relevant information in the provided context.
4. Compare each criterion against the company's capabilities.
5. Ensure ALL criteria are met for compliance.
6. Double-check your analysis before concluding.

Output format:
Explanation: [Your thought process and reasoning on how the company's product meets or does not meet each specific requirement.]
Compliance: [Yes/No]

Here are some examples of good responses:

Question: Should protect existing device investments with auto-sensing 8, 16, and 32 Gbit/sec capabilities.
Explanation: Copmpany's product supports auto sensing of 8, 16, and 32 Gbit/sec capabilities, which aligns with the customer's requirement to protect existing device investments.
Compliance: Yes or No (only for specific criteria)

Now, please analyze the following in detail:

Context: {data_dict['question']}

Question: {formatted_texts}
"""
        ),
    }
    messages.append(text_message)  # list of image_json(s) and text_json
    return messages


def MultiModal_LLM(msgs_input, model_id, api_url):
    temp_base64_image_list = []
    for each_input in msgs_input:
        if each_input["type"] == "text":
            txt_prompt = each_input["text"]
        elif each_input["type"] == "image_url":
            temp_base64_image_list.append(each_input["image_url"])
    
    num_images = len(temp_base64_image_list)

    client = OpenAI(
        api_key="EMPTY",
        base_url=api_url,
    )

    messages = [{
        "role": "user",
        "content": []
    }]
    
    # Add text content
    messages[0]["content"].append({
        "type": "text",
        "text": txt_prompt
    })
    
    # Add image content if present
    if num_images > 0:
        for b64_img in temp_base64_image_list:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}"
                }
            })

    # Create completion with explicit timeout and chunk_size settings
    chat_completion_from_url = client.chat.completions.create(
        messages=messages,
        model=model_id,
        temperature=0.1,
        stream=True,
        # max_tokens=512,
        extra_body={
            'repetition_penalty': 1,
        },
        timeout=60  # Add explicit timeout
    )

    return chat_completion_from_url


def multi_modal_rag_chain(model_id, vllm_url):
    """
    Multi-modal RAG chain that accepts pre-retrieved documents and specific model
    """
    chain = (
        {
            "processed_docs": RunnableLambda(lambda x: x["processed_docs"]),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | RunnableLambda(lambda x: {
            "context": x["processed_docs"],
            "question": x["question"]
        })
        | RunnableLambda(img_prompt_func)
        | RunnableLambda(lambda x: MultiModal_LLM(x, model_id, vllm_url))
    )
    return chain


def obtaining_chatbot_msg(user_message: str, history1, history2, answers_dict: Dict[str, str]):
    """
    Updated to handle correct answer lookup
    """
    correct_answer = get_correct_answer(user_message, answers_dict)
    return "", history1 + [[user_message, None]], history2 + [[user_message, None]], correct_answer


def format_message_with_thinking(text):
    """Format text by replacing thinking tags with visual indicators"""
    formatted_text = text

    # Handle opening tag
    if "<think>" in formatted_text:
        formatted_text = formatted_text.replace("<think>", "\n🤔 Thinking Process: \n")
    
    # Handle closing tag
    if "</think>" in formatted_text:
        formatted_text = formatted_text.replace("</think>", "\n\n💭 End of Thinking Process \n--- \n--- \n# Response:\n\n")
    
    return formatted_text

def generating_output_store(history1, history2):
    query = history1[-1][0]
    docs = retriever.get_relevant_documents(query)
    processed_docs = split_image_text_types(docs)
    
    # Prepare document data
    text_results = []
    img_results = []
    for doc in docs:
        if doc.metadata.get("type") == "text":
            text_results.append({
                "id": doc.metadata.get("id"),
                "original_content": doc.page_content,
                "doc_name": doc.metadata.get("doc_name"),
                "pg_no": doc.metadata.get("pg_no")
            })
        elif doc.metadata.get("type") == "image":
            img_results.append(doc.page_content)

    # Create chains
    chain1 = multi_modal_rag_chain(MODEL_ID1, VLLM_API_URL1)
    chain2 = multi_modal_rag_chain(MODEL_ID2, VLLM_API_URL2)

    # Get streams
    stream1 = chain1.invoke({
        "processed_docs": processed_docs,
        "question": query
    })
    
    stream2 = chain2.invoke({
        "processed_docs": processed_docs,
        "question": query
    })
    
    # Initialize histories and thinking state
    history1[-1][1] = ""
    history2[-1][1] = ""
    
    # Convert streams to iterators
    iter1 = iter(stream1)
    iter2 = iter(stream2)
    
    # Keep track of which streams are still active
    active1 = True
    active2 = True
    buffer1 = ""
    buffer2 = ""
    
    while active1 or active2:
        chunk1 = None
        chunk2 = None
        
        if active1:
            try:
                chunk1 = next(iter1)
            except StopIteration:
                active1 = False
                
        if active2:
            try:
                chunk2 = next(iter2)
            except StopIteration:
                active2 = False
        
        # Process chunks and update histories
        if chunk1 and chunk1.choices[0].delta.content:
            content1 = chunk1.choices[0].delta.content
            buffer1 += content1
            formatted_content1 = format_message_with_thinking(buffer1)
            history1[-1][1] = formatted_content1
            
        if chunk2 and chunk2.choices[0].delta.content:
            content2 = chunk2.choices[0].delta.content
            buffer2 += content2
            formatted_content2 = format_message_with_thinking(buffer2)
            history2[-1][1] = formatted_content2
        
        # Yield updates if there's new content
        if (chunk1 and chunk1.choices[0].delta.content) or (chunk2 and chunk2.choices[0].delta.content):
            yield history1, history2, pd.DataFrame(text_results), img_results


def change_vector_store_for_llm(new_vector_store_name):
    global VS_in_use, db, tbl, retriever, ce_reranker, gte_embedding

    db = lancedb.connect(f"{HOME_PATH}/data/presales_RFP/vector_stores2")
    tbl = db.open_table(f"{new_vector_store_name}")
    tbl.create_fts_index(['original_content'], replace=True)

    # Create the retriever instance
    retriever = CustomLanceDBRetriever.initialize(tbl, ce_reranker, gte_embedding)

    VS_in_use = new_vector_store_name



def load_example_questions() -> tuple[list[str], Dict[str, str]]:
    """
    Load example questions and their corresponding answers from CSV.
    Returns a tuple of (list of questions, dict mapping questions to answers)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(f"{HOME_PATH}/rfp_compliance_simple.csv")
        # Convert to list and dict
        questions = df["Specifications"].tolist()
        answers_dict = dict(zip(df["Specifications"], df["Compliance (Yes/No)"]))
        return questions, answers_dict
    except Exception as e:
        print(f"Error loading examples: {e}")
        # Return empty defaults if file can't be loaded
        return [], {}


def get_correct_answer(question: str, answers_dict: Dict[str, str]) -> str:
    """
    Get the correct answer for a question if it exists in our dataset.
    """
    return answers_dict.get(question, "No reference answer available for this question.")


css = """
footer {visibility: hidden}

"""


def llm_comparison_demo():
    global db, tbl, ce_reranker, gte_embedding, retriever, llm
    global MODEL_ID1, VLLM_API_URL1, MODEL_ID2, VLLM_API_URL2, api_url

    example_questions, answers_dict = load_example_questions()

    api_url, llm_list, MODEL_ID1, VLLM_API_URL1, MODEL_ID2, VLLM_API_URL2 = model_prep()
    db, tbl, ce_reranker, gte_embedding, retriever = preparation()
    
    with gr.Blocks(analytics_enabled=False, 
                   css = css) as llm_comparison_interface:
        with gr.Row():
            with gr.Column(scale=15):
                gr.Markdown("""
                            # LLM Comparison
                            """)

            with gr.Column(scale=1, min_width=200):
                toggle_dark = gr.Button(value="💡🌙")
        with gr.Tab("Model Comparison"):
            gr.Markdown("")
            with gr.Row():
                list_of_vector_stores = obtain_vector_stores()
                available_vector_stores = gr.Dropdown(choices=list_of_vector_stores,
                                                      type="value",
                                                      multiselect=False,
                                                      allow_custom_value=False,
                                                      label="Available Vector Stores",
                                                      value="table_name",
                                                      interactive=True,
                                                      )
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        available_llms1 = gr.Dropdown(choices=llm_list,
                            type="value",
                            multiselect=False,
                            allow_custom_value=False,
                            label="Available LLMs",
                            value="llama31_8b_instruct",
                            interactive=True,
                            )
                with gr.Column():
                    with gr.Row():
                        available_llms2 = gr.Dropdown(choices=llm_list,
                            type="value",
                            multiselect=False,
                            allow_custom_value=False,
                            label="Available LLMs",
                            value="DeepSeek-R1-Distill-Llama-8B",
                            interactive=True,
                            )
            with gr.Row():
                with gr.Column():
                    chatbot1 = gr.Chatbot(label="Chatbot 1",
                                height=600, 
                                show_copy_button=True,
                                render_markdown=True,

                    )
                with gr.Column():
                    chatbot2 = gr.Chatbot(label="Chatbot 2",
                                height=600, 
                                show_copy_button=True,
                                render_markdown=True,

                    )
            with gr.Row():
                correct_answer = gr.Textbox(label="Correct Answer", interactive=False, placeholder="Ask a question first :)")
            with gr.Row():
                msg = gr.Textbox(label="Question", interactive=True, placeholder="Ask a question!")

            with gr.Row():
                with gr.Column():
                    clear = gr.ClearButton([msg, chatbot1])
                with gr.Column():
                    btn = gr.Button(value="Submit", variant="primary")

            with gr.Row():
                gr.Examples(label="Example Questions", 
                    examples=example_questions,
                    inputs=msg)

            with gr.Accordion("Documents/Images used for chatbot with vector store inferencing", open=False):
                with gr.Row():
                    doc_table = gr.Dataframe(label="Referenced Documents",
                                                        interactive=False,
                                                        wrap=True,
                                                        max_height=900, 
                                                        headers=["original_content",
                                                        "doc_name",
                                                        "pg_no"])
                with gr.Row():
                    retrieved_images = gr.Gallery(label="Referenced Pictures", interactive=False, columns=2, allow_preview=True)


                    

        ###################################################################################
        ################################ Dark mode toggle #################################
        ###################################################################################
        toggle_dark.click(
            None,
            js="""
            () => {
                document.body.classList.toggle('dark');
                document.body.classList.toggle('vsc-initialized dark');
                document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primar)'
            }
            """, )

        ###################################################################################
        ################################# Chatbot Gradio ##################################
        ###################################################################################
        msg.submit(
            fn=obtaining_chatbot_msg,
            inputs=[msg, chatbot1, chatbot2, gr.State(answers_dict)],
            outputs=[msg, chatbot1, chatbot2, correct_answer],
            queue=False
        ).then(
            fn=generating_output_store,
            inputs=[chatbot1, chatbot2],
            outputs=[chatbot1, chatbot2, doc_table, retrieved_images]
        )
        
        btn.click(
            fn=obtaining_chatbot_msg,
            inputs=[msg, chatbot1, chatbot2, gr.State(answers_dict)],
            outputs=[msg, chatbot1, chatbot2, correct_answer],
            queue=False
        ).then(
            fn=generating_output_store,
            inputs=[chatbot1, chatbot2],
            outputs=[chatbot1, chatbot2, doc_table, retrieved_images]
        )
        available_vector_stores.change(fn=change_vector_store_for_llm, inputs=available_vector_stores)
        available_llms1.change(fn=change_llm1, inputs=available_llms1)
        available_llms2.change(fn=change_llm2, inputs=available_llms2)
        


    return llm_comparison_interface


if __name__ == "__main__":
    demo = llm_comparison_demo()
    demo.queue(max_size=10).launch(server_name='0.0.0.0', share=False)
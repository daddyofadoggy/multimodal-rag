from pathlib import Path
import os
from os import path as osp
import gradio as gr
from dotenv import load_dotenv
from crud.vector_store import MultimodalLanceDB
from preprocess.embedding import BridgeTowerEmbeddings
from preprocess.preprocessing import extract_and_save_frames_and_metadata
#from utils import encode_image
from utils import (
    download_video, 
    get_transcript_vtt,
    download_youtube_subtitle,
    get_video_id_from_url,
    str2time,
    maintain_aspect_ratio_resize,
    getSubs,
    encode_image,
)
from mistralai import Mistral
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from PIL import Image

import lancedb

# -------------------------------
# 1. Setup
# -------------------------------
load_dotenv()
LANCEDB_HOST_FILE = "./shared_data/.lancedb"
TBL_NAME = "vectorstore"

db = lancedb.connect(LANCEDB_HOST_FILE)
embedder = BridgeTowerEmbeddings()

# -------------------------------
# 2. Preprocessing + Storage
# -------------------------------
def preprocess_and_store(youtube_url: str):
    """Download video, extract frames+metadata, embed & store in LanceDB"""

    video_url = youtube_url
    video_dir = "./shared_data/videos/video1"
    # download Youtube video to ./shared_data/videos/video1
    video_filepath = download_video(video_url, video_dir)

    # download Youtube video's subtitle to ./shared_data/videos/video1
    video_transcript_filepath = download_youtube_subtitle(video_url, video_dir)

    extracted_frames_path = osp.join(video_dir, 'extracted_frame')

    # create these output folders if not existing
    Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    # call the function to extract frames and metadatas
    metadatas = extract_and_save_frames_and_metadata(
                video_filepath, 
                video_transcript_filepath,
                extracted_frames_path,
                video_dir,
            )
    # collect transcripts and image paths
    video_trans = [vid['transcript'] for vid in metadatas]
    video_img_path = [vid['extracted_frame_path'] for vid in metadatas]

    n = 7
    updated_video_trans = [
    ' '.join(video_trans[i-int(n/2) : i+int(n/2)]) if i-int(n/2) >= 0 else
    ' '.join(video_trans[0 : i + int(n/2)]) for i in range(len(video_trans))
    ]
    # also need to update the updated transcripts in metadata
    for i in range(len(updated_video_trans)):
        metadatas[i]['transcript'] = updated_video_trans[i]

    _ = MultimodalLanceDB.from_text_image_pairs(
        texts=updated_video_trans,
        image_paths=video_img_path,
        embedding=embedder,
        metadatas=metadatas,
        connection=db,
        table_name=TBL_NAME,
        mode="overwrite",
    )
    return f"✅ Video processed and stored: {youtube_url}"

# -------------------------------
# 3. Retrieval + Prompt Functions
# -------------------------------
vectorstore = MultimodalLanceDB(
    uri=LANCEDB_HOST_FILE,
    embedding=embedder,
    table_name=TBL_NAME
)

retriever_module = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

def prompt_processing(input):
    retrieved_results = input["retrieved_results"]
    user_query = input["user_query"]
    #retrieved_results = retriever_module.invoke(user_query)

    retrieved_results = retrieved_results[0]
    prompt_template = (
        "The transcript associated with the image is '{transcript}'. "
        "{user_query}"
    )

    retrieved_metadata = retrieved_results.metadata
    transcript = retrieved_metadata["transcript"]
    frame_path = retrieved_metadata["extracted_frame_path"]

    return {
        "prompt": prompt_template.format(transcript=transcript, user_query=user_query),
        "frame_path": frame_path,
    }


def lvlm_inference(input):

    # get the retrieved results and user's query
    lvlm_prompt = input['prompt']
    frame_path = input['frame_path']
    
    
    # Retrieve the API key from environment variables
    api_key = os.getenv("MISTRAL_API_KEY")
    
    # Initialize the Mistral client
    client = Mistral(api_key=api_key)

    base64_image = encode_image(frame_path)

    # Define the messages for the chat
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": lvlm_prompt
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}" 
                }
            ]
        }
    ]

    # Get the chat response
    chat_response = client.chat.complete(
        model="pixtral-12b-2409",
        messages=messages
    )

    # Print the content of the response
    return chat_response.choices[0].message.content, frame_path

# LangChain Runnable chain
prompt_processing_module = RunnableLambda(prompt_processing)
lvlm_inference_module = RunnableLambda(lvlm_inference)

mm_rag_chain = (
    RunnableParallel({"retrieved_results": retriever_module, "user_query": RunnablePassthrough()})
    | prompt_processing_module
    | lvlm_inference_module
)

# -------------------------------
# 4. Chat API for Gradio
# -------------------------------
video_loaded = False

def load_video(youtube_url):
    global video_loaded
    status = preprocess_and_store(youtube_url)
    video_loaded = True
    return status

def chat_interface(message, history):
    if not video_loaded:
        return "", history, None

    final_text_response, frame_path = mm_rag_chain.invoke(message)
    history.append((message, final_text_response))
    
    # Load and return the image
    try:
        retrieved_image = Image.open(frame_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        retrieved_image = None
    
    return "", history, retrieved_image

# -------------------------------
# 5. Enhanced Gradio Interface
# -------------------------------
with gr.Blocks(title="Multimodal RAG Video Chat") as demo:
    gr.Markdown("# 🎬 Multimodal RAG Video Chat\nChat with YouTube clips using BridgeTower + LanceDB + Pixtral!")

    with gr.Tab("1. Load Video"):
        youtube_url = gr.Textbox(
            label="YouTube URL", 
            placeholder="Paste a YouTube link here...",
            lines=1
        )
        load_btn = gr.Button("Process Video", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        load_btn.click(load_video, inputs=youtube_url, outputs=status)

    with gr.Tab("2. Chat with Video"):
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat about the video",
                    height=500,
                    show_label=True
                )
                
            with gr.Column(scale=1):
                retrieved_image = gr.Image(
                    label="Retrieved Frame",
                    height=400,
                    show_label=True,
                    interactive=False
                )
                
        with gr.Row():
            msg = gr.Textbox(
                label="Your question",
                placeholder="Ask something about the video...",
                lines=2,
                scale=4
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Clear message after sending
        msg.submit(chat_interface, inputs=[msg, chatbot], outputs=[msg, chatbot, retrieved_image])
        send_btn.click(chat_interface, inputs=[msg, chatbot], outputs=[msg, chatbot, retrieved_image])

    # Add some usage instructions
    with gr.Tab("📖 Instructions"):
        gr.Markdown("""
        ## How to use this Multimodal RAG system:
        
        1. **Load Video**: 
           - Go to the "Load Video" tab
           - Paste a YouTube URL
           - Click "Process Video" and wait for processing to complete
           
        2. **Chat with Video**:
           - Go to the "Chat with Video" tab
           - Ask questions about the video content
           - The system will retrieve the most relevant frame and provide answers
           - The retrieved frame will be displayed on the right side
           
        ## Features:
        - 🎥 Processes YouTube videos automatically
        - 🧠 Uses BridgeTower for multimodal embeddings
        - 💾 Stores data in LanceDB vector database
        - 🤖 Powered by Pixtral vision-language model
        - 🖼️ Shows relevant video frames alongside responses
        """)

if __name__ == "__main__":
    print('App starting...')
    demo.launch(server_name="0.0.0.0", server_port=7860)
# Multimodal RAG System: Code Walkthrough and Project Flow Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Detailed Code Walkthrough](#detailed-code-walkthrough)
6. [Project Flow](#project-flow)
7. [Key Components Deep Dive](#key-components-deep-dive)
8. [Data Flow Diagram](#data-flow-diagram)
9. [API and Endpoints](#api-and-endpoints)
10. [Usage Guide](#usage-guide)

---

## Project Overview

This project implements a **Multimodal Retrieval-Augmented Generation (RAG)** system that enables users to interact with YouTube videos through natural language queries. The system processes video content, stores it in a vector database, and retrieves relevant frames with contextual information to answer user questions.

### Key Features
- Downloads and processes YouTube videos automatically
- Extracts video frames aligned with transcript segments
- Creates multimodal embeddings using BridgeTower model
- Stores embeddings in LanceDB vector database
- Retrieves relevant video frames based on user queries
- Generates contextual responses using Pixtral vision-language model
- Interactive web interface built with Gradio

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface (Gradio)                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
            ┌───────▼───────┐      ┌───────▼────────┐
            │ Video Loading │      │  Query & Chat  │
            │    Module     │      │     Module     │
            └───────┬───────┘      └───────┬────────┘
                    │                      │
        ┌───────────▼──────────┐          │
        │  Preprocessing       │          │
        │  • Video Download    │          │
        │  • Transcript Fetch  │          │
        │  • Frame Extraction  │          │
        └───────────┬──────────┘          │
                    │                      │
        ┌───────────▼──────────┐          │
        │  BridgeTower         │          │
        │  Embedding Creation  │          │
        └───────────┬──────────┘          │
                    │                      │
        ┌───────────▼──────────┐    ┌─────▼──────┐
        │   LanceDB Vector     │◄───┤  Retriever │
        │      Storage         │    └─────┬──────┘
        └──────────────────────┘          │
                                    ┌─────▼──────┐
                                    │  Pixtral   │
                                    │   (LLM)    │
                                    └────────────┘
```

---

## Technology Stack

### Core Technologies
- **Python 3.13**: Main programming language
- **PyTorch**: Deep learning framework
- **Transformers (HuggingFace)**: Model loading and processing

### Models
- **BridgeTower**: Multimodal embedding model for vision-text alignment
- **Pixtral-12B**: Vision-language model by Mistral AI for response generation

### Vector Database
- **LanceDB**: Vector database for storing and retrieving multimodal embeddings

### Frameworks
- **LangChain**: Orchestration framework for RAG pipeline
- **Gradio**: Web interface framework

### Utilities
- **OpenCV (cv2)**: Video processing and frame extraction
- **webvtt-py**: Transcript file parsing
- **yt-dlp / pytubefix**: YouTube video downloading
- **youtube-transcript-api**: Fetching YouTube subtitles

---

## Project Structure

```
Multimodal-RAG-BTM/
├── src/
│   ├── app.py                       # Main Gradio application
│   ├── utils.py                     # Utility functions
│   ├── preprocess/
│   │   ├── embedding.py            # BridgeTower embedding wrapper
│   │   └── preprocessing.py        # Video frame extraction
│   ├── crud/
│   │   └── vector_store.py         # Custom LanceDB implementation
│   ├── data/                       # Data storage directory
│   ├── shared_data/
│   │   ├── .lancedb/              # LanceDB database files
│   │   └── videos/                # Downloaded videos and frames
│   └── *.ipynb                     # Jupyter notebooks (tutorials)
├── requirements.txt                 # Python dependencies
└── README.md                       # Project documentation
```

---

## Detailed Code Walkthrough

### 1. Main Application (`src/app.py`)

The main application orchestrates the entire system using Gradio for the user interface.

#### 1.1 Setup and Initialization (Lines 1-38)

```python
# Import dependencies
from pathlib import Path
import os
from dotenv import load_dotenv
from crud.vector_store import MultimodalLanceDB
from preprocess.embedding import BridgeTowerEmbeddings
from preprocess.preprocessing import extract_and_save_frames_and_metadata
from utils import *
from mistralai import Mistral
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import lancedb

# Configuration
load_dotenv()
LANCEDB_HOST_FILE = "./shared_data/.lancedb"
TBL_NAME = "vectorstore"

# Initialize components
db = lancedb.connect(LANCEDB_HOST_FILE)
embedder = BridgeTowerEmbeddings()
```

**Key Points:**
- Loads environment variables (MISTRAL_API_KEY)
- Establishes LanceDB connection
- Initializes BridgeTower embedder

#### 1.2 Video Preprocessing Function (Lines 43-89)

**Location:** `app.py:43-89`

```python
def preprocess_and_store(youtube_url: str):
    """Download video, extract frames+metadata, embed & store in LanceDB"""
```

**Step-by-step Process:**

1. **Download Video** (Line 49)
   - Uses `download_video()` to fetch YouTube video
   - Saves to `./shared_data/videos/video1`

2. **Download Transcript** (Line 52)
   - Uses `download_youtube_subtitle()` to get VTT subtitle file
   - Falls back to YouTube Transcript API

3. **Extract Frames** (Lines 61-66)
   - Calls `extract_and_save_frames_and_metadata()`
   - Extracts one frame per transcript segment
   - Saves metadata (timestamp, transcript, path)

4. **Transcript Augmentation** (Lines 71-78)
   - Creates contextual transcripts by combining n=7 adjacent segments
   - Provides more context for each frame

5. **Create Embeddings & Store** (Lines 80-88)
   - Uses BridgeTower to create multimodal embeddings
   - Stores in LanceDB with metadata
   - Mode: "overwrite" (replaces existing data)

#### 1.3 Retrieval Setup (Lines 94-103)

**Location:** `app.py:94-103`

```python
vectorstore = MultimodalLanceDB(
    uri=LANCEDB_HOST_FILE,
    embedding=embedder,
    table_name=TBL_NAME
)

retriever_module = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

**Configuration:**
- Retrieves top-3 similar frames
- Uses cosine similarity search

#### 1.4 Prompt Processing (Lines 105-123)

**Location:** `app.py:105-123`

```python
def prompt_processing(input):
    retrieved_results = input["retrieved_results"]
    user_query = input["user_query"]

    # Use top result
    retrieved_results = retrieved_results[0]

    # Build prompt
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
```

**Process:**
- Extracts top retrieved result
- Combines transcript context with user query
- Returns formatted prompt and image path

#### 1.5 Vision-Language Model Inference (Lines 126-165)

**Location:** `app.py:126-165`

```python
def lvlm_inference(input):
    lvlm_prompt = input['prompt']
    frame_path = input['frame_path']

    # Initialize Mistral client
    api_key = os.getenv("MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)

    # Encode image to base64
    base64_image = encode_image(frame_path)

    # Create multimodal message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": lvlm_prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
            ]
        }
    ]

    # Get response from Pixtral
    chat_response = client.chat.complete(
        model="pixtral-12b-2409",
        messages=messages
    )

    return chat_response.choices[0].message.content, frame_path
```

**Key Operations:**
- Encodes retrieved frame as base64
- Sends image + prompt to Pixtral model
- Returns generated response and frame path

#### 1.6 LangChain Pipeline (Lines 167-175)

**Location:** `app.py:167-175`

```python
prompt_processing_module = RunnableLambda(prompt_processing)
lvlm_inference_module = RunnableLambda(lvlm_inference)

mm_rag_chain = (
    RunnableParallel({"retrieved_results": retriever_module, "user_query": RunnablePassthrough()})
    | prompt_processing_module
    | lvlm_inference_module
)
```

**Pipeline Flow:**
1. **RunnableParallel**: Retrieves similar frames while passing through query
2. **prompt_processing_module**: Formats prompt with context
3. **lvlm_inference_module**: Generates response using Pixtral

#### 1.7 Gradio Interface (Lines 180-276)

**Location:** `app.py:180-276`

Three main tabs:
1. **Load Video**: Input YouTube URL and process
2. **Chat with Video**: Interactive Q&A interface
3. **Instructions**: Usage guide

---

### 2. Utility Functions (`src/utils.py`)

Provides essential helper functions for the entire system.

#### 2.1 Image Encoding (`utils.py:21-30`)

```python
def encode_image(image_path_or_PIL_img):
    """Encodes image to base64 format"""
```
- Handles both file paths and PIL Image objects
- Returns base64-encoded string

#### 2.2 BridgeTower Embeddings (`utils.py:52-81`)

```python
def bt_embeddings(prompt, base64_image=None):
    """Generate embeddings using BridgeTower model"""
```

**Process:**
1. Loads BridgeTower processor and model
2. If image provided: creates cross-modal embeddings
3. If no image: creates text-only embeddings
4. Returns embedding vector (512 dimensions)

#### 2.3 Video Download (`utils.py:197-221`)

```python
def download_video(video_url, path='/tmp/'):
    """Download YouTube video to specified path"""
```

**Features:**
- Checks if video already exists
- Prefers 720p resolution
- Shows download progress bar
- Returns filepath

#### 2.4 Subtitle Download (`utils.py:264-283`)

```python
def download_youtube_subtitle(video_url, path):
    """Download YouTube subtitles in VTT format"""
```

**Uses:**
- yt-dlp for reliable subtitle extraction
- Supports English subtitles
- VTT format for timestamp alignment

#### 2.5 Time Conversion (`utils.py:109-117`)

```python
def str2time(strtime):
    """Convert VTT timestamp to milliseconds"""
```

**Example:**
- Input: "00:01:23.456"
- Output: 83456 (milliseconds)

---

### 3. Preprocessing Module (`src/preprocess/`)

#### 3.1 Frame Extraction (`preprocessing.py:9-65`)

**Location:** `preprocess/preprocessing.py:9-65`

```python
def extract_and_save_frames_and_metadata(
    path_to_video,
    path_to_transcript,
    path_to_save_extracted_frames,
    path_to_save_metadatas
):
```

**Process:**

1. **Load Video and Transcript**
   ```python
   video = cv2.VideoCapture(path_to_video)
   trans = webvtt.read(path_to_transcript)
   ```

2. **Iterate Through Transcript Segments**
   - For each subtitle entry:
     - Calculate mid-point timestamp
     - Seek to that position in video
     - Extract frame
     - Resize to height=350px (maintaining aspect ratio)
     - Save as JPEG

3. **Create Metadata**
   ```python
   metadata = {
       'extracted_frame_path': img_fpath,
       'transcript': text,
       'video_segment_id': idx,
       'video_path': path_to_video,
       'mid_time_ms': mid_time_ms,
   }
   ```

4. **Save Metadata JSON**
   - All metadata saved to `metadatas.json`

**Example Output:**
- Frame: `frame_0.jpg`, `frame_1.jpg`, ...
- Metadata: Transcript text, timestamp, paths

#### 3.2 BridgeTower Embeddings (`embedding.py:8-69`)

**Location:** `preprocess/embedding.py:8-69`

```python
class BridgeTowerEmbeddings(BaseModel, Embeddings):
    """BridgeTower embedding model wrapper"""
```

**Key Methods:**

1. **`embed_image_text_pairs()`** (Lines 11-37)
   - Takes parallel lists of texts and image paths
   - Generates multimodal embeddings
   - Uses `bt_embeddings()` from utils

2. **`embed_documents()`** (Lines 39-56)
   - Text-only embedding generation
   - Used for pure text documents

3. **`embed_query()`** (Lines 58-69)
   - Embeds user query
   - Returns single embedding vector

**Integration:**
- Extends LangChain's `Embeddings` base class
- Compatible with LangChain vector stores

---

### 4. Vector Store (`src/crud/vector_store.py`)

**Location:** `crud/vector_store.py:6-140`

#### 4.1 MultimodalLanceDB Class

```python
class MultimodalLanceDB(LanceDB):
    """LanceDB vector store for multimodal data"""
```

**Extends:** LangChain's LanceDB implementation

**Additional Fields:**
- `image_path_key`: Stores path to image for each entry

#### 4.2 Adding Image-Text Pairs (`vector_store.py:51-111`)

```python
def add_text_image_pairs(
    self,
    texts: Iterable[str],
    image_paths: Iterable[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    **kwargs
) -> List[str]:
```

**Process:**

1. **Validation**
   ```python
   assert len(texts)==len(image_paths)
   ```

2. **Generate Embeddings**
   ```python
   embeddings = self._embedding.embed_image_text_pairs(
       texts=list(texts),
       images=list(image_paths)
   )
   ```

3. **Create Documents**
   ```python
   docs.append({
       self._vector_key: embedding,
       self._id_key: ids[idx],
       self._text_key: text,
       self._image_path_key: image_paths[idx],
       "metadata": metadata,
   })
   ```

4. **Store in LanceDB**
   - Creates table if doesn't exist
   - Appends/overwrites based on mode

#### 4.3 Factory Method (`vector_store.py:113-140`)

```python
@classmethod
def from_text_image_pairs(cls, texts, image_paths, embedding, ...):
    """Create vectorstore from image-text pairs"""
```

**Usage:**
```python
vectorstore = MultimodalLanceDB.from_text_image_pairs(
    texts=updated_video_trans,
    image_paths=video_img_path,
    embedding=embedder,
    metadatas=metadatas,
    connection=db,
    table_name=TBL_NAME,
    mode="overwrite",
)
```

---

## Project Flow

### Complete End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 1: VIDEO LOADING PHASE                       │
└─────────────────────────────────────────────────────────────────────┘

1. User enters YouTube URL in Gradio interface
   ↓
2. load_video(youtube_url) triggered
   ↓
3. preprocess_and_store(youtube_url) executed:

   a. download_video(video_url, video_dir)
      • Downloads MP4 file to ./shared_data/videos/video1/

   b. download_youtube_subtitle(video_url, video_dir)
      • Fetches VTT subtitle file

   c. extract_and_save_frames_and_metadata()
      • Reads VTT file
      • For each transcript segment:
        - Calculate mid-point timestamp
        - Extract frame at that time
        - Resize and save as JPEG
        - Store metadata (transcript, timestamp, path)

   d. Transcript augmentation (n=7 context window)
      • video_trans[i-3:i+3] combined
      • Provides richer context

   e. BridgeTower embedding generation
      • embed_image_text_pairs(texts, image_paths)
      • Creates 512-dim multimodal embeddings

   f. Store in LanceDB
      • MultimodalLanceDB.from_text_image_pairs()
      • Mode: overwrite
      • Creates vectorstore table

4. Status returned: "✅ Video processed and stored"

┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 2: QUERY PHASE                              │
└─────────────────────────────────────────────────────────────────────┘

1. User enters question in chat interface
   ↓
2. chat_interface(message, history) triggered
   ↓
3. mm_rag_chain.invoke(message) executes:

   a. RunnableParallel stage:
      • retriever_module.invoke(query)
        - Embeds query using BridgeTower
        - Performs similarity search in LanceDB
        - Returns top-3 most similar frames with metadata
      • user_query passed through

   b. prompt_processing stage:
      • Extracts top result (most similar)
      • Retrieves metadata:
        - transcript
        - extracted_frame_path
      • Creates prompt:
        "The transcript associated with the image is '{transcript}'. {user_query}"

   c. lvlm_inference stage:
      • Loads frame image
      • Encodes to base64
      • Sends to Mistral Pixtral API:
        - Text: augmented prompt
        - Image: base64-encoded frame
      • Receives response from vision-language model

4. Response displayed in chatbot
5. Retrieved frame shown in image panel
6. History updated

┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 3: ITERATION                                 │
└─────────────────────────────────────────────────────────────────────┘

User can ask multiple questions:
• Each query goes through same retrieval pipeline
• Fresh retrieval for each question (no memory between queries)
• Different frames may be retrieved for different questions
```

---

## Key Components Deep Dive

### 1. BridgeTower Model

**Purpose:** Creates aligned multimodal embeddings for vision and text

**Architecture:**
- Pre-trained on large-scale image-text pairs
- Outputs 512-dimensional embeddings
- Supports both unimodal and cross-modal embeddings

**Model Variants Used:**
- `BridgeTower/bridgetower-large-itm-mlm-itc`

**Embedding Types:**
- **Cross-modal** (`cross_embeds`): When image + text provided
- **Text-only** (`text_embeds`): When only text provided

**Usage in Project:**
```python
# Creating embeddings
processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

inputs = processor(images=images, text=texts, padding=True, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.cross_embeds  # Shape: (batch_size, 512)
```

---

### 2. LanceDB Vector Store

**Why LanceDB?**
- Fast similarity search
- Native support for multimodal data
- Lightweight and embedded
- No separate server needed

**Schema:**
```python
{
    "id": "uuid-string",
    "vector": [512-dim embedding],
    "text": "transcript text",
    "image_path": "/path/to/frame.jpg",
    "metadata": {
        "extracted_frame_path": "/path/to/frame.jpg",
        "transcript": "original transcript",
        "video_segment_id": 0,
        "video_path": "/path/to/video.mp4",
        "mid_time_ms": 12345.0
    }
}
```

**Search Process:**
1. Query embedded using BridgeTower
2. Cosine similarity computed against all vectors
3. Top-k results returned with metadata
4. Results include image paths for display

---

### 3. Pixtral Vision-Language Model

**Model:** `pixtral-12b-2409` by Mistral AI

**Capabilities:**
- Understands both images and text
- Generates contextual responses
- 12B parameters
- Optimized for visual question answering

**API Usage:**
```python
client = Mistral(api_key=api_key)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
        ]
    }
]
chat_response = client.chat.complete(model="pixtral-12b-2409", messages=messages)
```

**Input Format:**
- Text: Augmented prompt with transcript context
- Image: Base64-encoded JPEG frame

**Output:**
- Natural language response based on visual and textual context

---

### 4. Transcript Augmentation Strategy

**Problem:** Single subtitle segment may lack context

**Solution:** Sliding window aggregation

```python
n = 7  # Window size
updated_video_trans = [
    ' '.join(video_trans[i-int(n/2) : i+int(n/2)])
    if i-int(n/2) >= 0
    else ' '.join(video_trans[0 : i + int(n/2)])
    for i in range(len(video_trans))
]
```

**Example:**
- Original: `["Hello", "how", "are", "you", "doing", "today"]`
- For index 3 ("you"), augmented becomes:
  - `"hello how are you doing today"` (3 before + current + 3 after)

**Benefits:**
- Richer semantic context
- Better retrieval accuracy
- More informative responses

---

### 5. Frame Extraction Strategy

**Timing:** Extract frame at mid-point of each subtitle segment

```python
start_time_ms = str2time(transcript.start)  # e.g., 1000ms
end_time_ms = str2time(transcript.end)      # e.g., 3000ms
mid_time_ms = (end_time_ms + start_time_ms) / 2  # 2000ms

video.set(cv2.CAP_PROP_POS_MSEC, mid_time_ms)
success, frame = video.read()
```

**Advantages:**
- Aligns with spoken content
- Captures stable frames (not transitions)
- One frame per semantic unit

**Frame Processing:**
```python
image = maintain_aspect_ratio_resize(frame, height=350)
cv2.imwrite(img_fpath, image)
```

---

## Data Flow Diagram

### Storage Phase

```
YouTube URL
    │
    ├─→ download_video() ──────────→ video.mp4
    │
    └─→ download_youtube_subtitle() → captions.vtt
             │
             ▼
    extract_and_save_frames_and_metadata()
             │
             ├─→ frame_0.jpg, frame_1.jpg, ...
             │
             └─→ metadatas.json
                      │
                      ▼
              Transcript Augmentation (n=7)
                      │
                      ▼
          BridgeTower Embeddings
          (text + image pairs)
                      │
                      ▼
              LanceDB Storage
         (vectors + metadata + paths)
```

### Retrieval Phase

```
User Query
    │
    ▼
BridgeTower.embed_query()
    │
    ▼
LanceDB.similarity_search(k=3)
    │
    ├─→ Result 1 (score: 0.95) ─┐
    ├─→ Result 2 (score: 0.87)  │ Top result selected
    └─→ Result 3 (score: 0.82)  │
                                 │
                                 ▼
                    Extract metadata:
                    • transcript
                    • frame_path
                                 │
                                 ▼
                    Format Prompt:
                    "The transcript associated with
                    the image is '{transcript}'.
                    {user_query}"
                                 │
                                 ▼
                    Pixtral VLM Inference
                    (prompt + frame image)
                                 │
                                 ▼
                    Generated Response
                                 │
                                 ▼
                    Display in Gradio
                    (text + retrieved frame)
```

---

## API and Endpoints

### Gradio Interface Endpoints

#### 1. Load Video
**Trigger:** Button click on "Process Video"

**Function:** `load_video(youtube_url)`

**Input:**
- `youtube_url` (str): YouTube URL

**Output:**
- Status message (str)

**Process Time:** 2-10 minutes depending on video length

---

#### 2. Chat Interface
**Trigger:** Message submission or Send button

**Function:** `chat_interface(message, history)`

**Input:**
- `message` (str): User query
- `history` (List[Tuple]): Chat history

**Output:**
- Empty string (clears input)
- Updated history
- Retrieved frame (PIL Image)

**Process Time:** 3-10 seconds per query

---

### External APIs Used

#### 1. Mistral API
**Endpoint:** Pixtral chat completion

**Configuration:**
```python
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
model = "pixtral-12b-2409"
```

**Rate Limits:** Based on Mistral API tier

**Cost:** Pay-per-use based on tokens processed

---

#### 2. YouTube APIs
**Methods:**
- `pytubefix.YouTube`: Video download
- `yt_dlp`: Subtitle download
- `youtube_transcript_api`: Transcript fetching

**Rate Limits:** None for public videos

---

## Usage Guide

### Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   Create `.env` file:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

3. **Run Application**
   ```bash
   python src/app.py
   ```

4. **Access Interface**
   Open browser to `http://localhost:7860`

---

### Using the System

#### Step 1: Load a Video

1. Navigate to **"1. Load Video"** tab
2. Paste YouTube URL (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
3. Click **"Process Video"**
4. Wait for processing (status will show progress)
5. Confirmation: `✅ Video processed and stored`

**Note:** Processing includes:
- Downloading video (~30 seconds)
- Downloading subtitles (~5 seconds)
- Extracting frames (~1-2 minutes)
- Creating embeddings (~2-5 minutes)
- Storing in database (~10 seconds)

---

#### Step 2: Chat with Video

1. Navigate to **"2. Chat with Video"** tab
2. Enter question in text box
3. Click **Send** or press Enter
4. View response in chat area
5. See retrieved frame on the right

**Example Queries:**
- "What is being discussed in this video?"
- "Describe the scene in detail"
- "What equipment is shown?"
- "Summarize the main points"

---

### Advanced Usage

#### Adjusting Retrieval Parameters

In `app.py`, modify:
```python
retriever_module = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieve more results
)
```

#### Changing Context Window

In `app.py`, modify:
```python
n = 11  # Larger context window (was 7)
```

#### Using Different VLM

Replace Pixtral with another vision-language model:
```python
# Example: Use OpenAI GPT-4V
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Modify lvlm_inference() accordingly
```

---

## File References

### Configuration Files
- **`.env`**: API keys and environment variables
- **`requirements.txt`**: Python dependencies
- **`.gitignore`**: Git ignore patterns

### Data Files
- **`shared_data/.lancedb/`**: Vector database storage
- **`shared_data/videos/video1/*.mp4`**: Downloaded videos
- **`shared_data/videos/video1/*.vtt`**: Subtitle files
- **`shared_data/videos/video1/extracted_frame/`**: Extracted frames
- **`shared_data/videos/video1/metadatas.json`**: Frame metadata

### Code Files
- **`src/app.py`** (276 lines): Main application
- **`src/utils.py`** (284 lines): Utility functions
- **`src/preprocess/embedding.py`** (69 lines): Embedding wrapper
- **`src/preprocess/preprocessing.py`** (65 lines): Frame extraction
- **`src/crud/vector_store.py`** (140 lines): Vector store implementation

---

## Performance Considerations

### Processing Time
- **Video Download:** ~30 seconds (for 5-min video)
- **Subtitle Download:** ~5 seconds
- **Frame Extraction:** ~1-2 minutes (depends on subtitle count)
- **Embedding Generation:** ~2-5 minutes (depends on frame count)
- **Total:** 4-8 minutes for initial processing

### Query Time
- **Embedding Query:** ~100ms
- **Vector Search:** ~50ms
- **Pixtral API Call:** 2-8 seconds
- **Total:** 3-10 seconds per query

### Storage
- **LanceDB:** ~10MB per 100 frames
- **Extracted Frames:** ~50KB per frame
- **Video File:** Original size (100-500MB typical)

---

## Limitations and Future Improvements

### Current Limitations
1. **No Chat History:** Each query is independent
2. **Single Video:** Only one video stored at a time (overwrite mode)
3. **English Only:** Subtitles must be in English
4. **No Fine-tuning:** Uses pre-trained models as-is
5. **API Dependency:** Requires Mistral API key and internet

### Potential Improvements
1. **Chat History Support:**
   ```python
   # Add conversation memory
   from langchain.memory import ConversationBufferMemory
   memory = ConversationBufferMemory()
   ```

2. **Multi-Video Support:**
   ```python
   # Use video_id in table_name
   table_name = f"vectorstore_{video_id}"
   mode = "append"
   ```

3. **Multilingual Support:**
   ```python
   # Auto-detect and translate
   from langdetect import detect
   from deep_translator import GoogleTranslator
   ```

4. **Fine-tuning BridgeTower:**
   - Domain-specific training
   - Improved embedding quality

5. **Local VLM:**
   - Use LLaVA or similar open-source models
   - Eliminate API dependency

6. **Re-ranking:**
   ```python
   # Add cross-encoder re-ranking
   from sentence_transformers import CrossEncoder
   reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
   ```

---

## Troubleshooting

### Common Issues

#### 1. "No module named 'lancedb'"
**Solution:**
```bash
pip install lancedb==0.18.0
```

#### 2. "MISTRAL_API_KEY not found"
**Solution:**
- Create `.env` file in project root
- Add: `MISTRAL_API_KEY=your_key_here`

#### 3. "Video download failed"
**Solution:**
- Check internet connection
- Verify YouTube URL is valid
- Try different video (some may be restricted)

#### 4. "No subtitles available"
**Solution:**
- Use videos with English subtitles/captions
- Alternatively, implement Whisper transcription:
  ```python
  import whisper
  model = whisper.load_model("base")
  result = model.transcribe(video_path)
  ```

#### 5. "CUDA out of memory"
**Solution:**
```python
# Reduce batch size in embedding.py
batch_size = 1  # Instead of 2
```

---

## Conclusion

This Multimodal RAG system demonstrates the power of combining:
- **Vision-Language Models** for understanding multimodal content
- **Vector Databases** for efficient similarity search
- **RAG Architecture** for grounded, context-aware responses

The implementation is modular, extensible, and production-ready with minor enhancements. The code is well-structured with clear separation of concerns across preprocessing, storage, retrieval, and inference modules.

---

## References

### Models
- BridgeTower: https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-itc
- Pixtral: https://docs.mistral.ai/capabilities/vision/

### Libraries
- LanceDB: https://lancedb.github.io/lancedb/
- LangChain: https://python.langchain.com/
- Gradio: https://www.gradio.app/

### Papers
- BridgeTower: "BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning"
- RAG: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

---

*Document generated: 2025-11-16*
*Project: Multimodal RAG with BridgeTower and Mistral*
*Version: 1.0*

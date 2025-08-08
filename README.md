# Multimodal RAG with BridgeTower Model

## Description
This repository contains the complete code and tutorials for implementing a multimodal retrieval-augmented generation (RAG) system capable of processing, storing, and retrieving video content. The system uses BridgeTower for multimodal embeddings, LanceDB as the vector store, and Pixtral as the conversation LLM.

## Installation
To install the necessary dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Tutorials
1. `mm_rag.ipynb`: Complete end-to-end implementation of the multimodal RAG system
2. `embedding_creation.ipynb`: Deep dive into generating multimodal embeddings using BridgeTower
3. `vector_store.ipynb`: Detailed guide on setting up and populating LanceDB for vector storage
4. `preprocessing_video.ipynb`: Comprehensive coverage of video preprocessing techniques, including:

    * Frame extraction
    * Transcript processing
    * Handling videos without transcripts
    * Transcript optimization strategies

## Required API Keys
You'll need to set up the following API keys:

`MISTRAL_API_KEY` for PixTral model access

## Data
The tutorial uses a sample video about a space expedition. You can replace it with any video of your choice, but make sure to:

* Include a transcript file (.vtt format)
* Or generate transcripts using Whisper
* Or use vision language models for caption generation

## Contributing
Contributions are welcome! Some areas for improvement include:

* Adding chat history support
* Prompt engineering refinements
* Alternative retrieval strategies e.g., Graph-RAG for reasoning 
* Testing with GPT 5, Gemini 1.5, Claude 3 Opus / Sonnet, Qwen-VL / Qwen-VL-Max, LLaVA-1.6 etc  

To contribute:(for use different Vision Language Model and compare performance)
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

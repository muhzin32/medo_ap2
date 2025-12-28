# Meddollina Voice Assistant

Meddollina is an advanced voice assistant web application that leverages Azure Cognitive Services and Hugging Face models to provide a natural, conversational interface. It supports wake word detection, barge-in interruption, emotion recognition, and contextual memory.

## Features

- **Voice Interaction**: 
  - **Wake Word**: Activates on "Hey Meddollina" (and other configurable triggers).
  - **Speech-to-Text (STT)**: High-accuracy transcription using Azure Speech SDK.
  - **Text-to-Speech (TTS)**: Natural sounding voice response.
- **Smart Conversation**:
  - **Contextual Memory**: Remembers past interactions using vector embeddings and MongoDB.
  - **Fact Extraction**: Automatically learns user details (name, preferences).
  - **Emotion Detection**: analyzes speech to detect user emotion and adapt responses.
- **Advanced Control**:
  - **Barge-in**: Interrupt the assistant mid-sentence naturally using "Stop", "Wait", etc.
  - **Active Listening**: Stays listening for 4 seconds after speaking to allow fluid conversation.
  - **Noise Filtering**: Intelligently ignores background noise and logs it separately.

## Tech Stack

- **Frontend**: React, TypeScript, Vite, SCSS
- **Backend**: Node.js (Express), Python (Flask, NLTK)
- **Database**: MongoDB (Atlas)
- **AI Services**: 
  - Azure OpenAI (Embeddings)
  - Azure Speech Services (STT/TTS)
  - Hugging Face Inference (LLM)

## Prerequisites

- Node.js (v18+)
- Python (v3.9+)
- MongoDB Atlas URI
- Azure Subscription (Speech Services & OpenAI)
- Hugging Face API Token

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Backend
PORT=3001
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/meddollina?retryWrites=true&w=majority

# Azure OpenAI (For Memory Embeddings)
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_API_VERSION=2024-05-01-preview

# Frontend (Vite)
VITE_AZURE_SPEECH_KEY=<your-speech-key>
VITE_AZURE_SPEECH_REGION=<your-region>
VITE_HUGGING_FACE_TOKEN=<your-hf-token>
```

## Installation

1.  **Install Frontend/Backend Dependencies**:
    ```bash
    npm install
    ```

2.  **Install Python Dependencies**:
    ```bash
    pip install -r backend/requirements.txt
    ```
    *(Note: Ensure you have a virtual environment active if preferred)*

## Running the Application

To start both the Frontend, Node.js Backend, and Python NLP Service:

```bash
npm start
```

This command uses `concurrently` to run:
- Frontend: `http://localhost:5173`
- Backend: `http://localhost:3001`
- Python Service: Internal port (usually 5001)

## Development

- **Frontend Only**: `npm run frontend`
- **Backend Only**: `npm run backend`

## License

Private Project.

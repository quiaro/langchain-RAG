---
title: RAG Application with FastAPI, LangChain and React
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
python_version: 3.12
app_port: 7860
app_file: main.py
pinned: false
short_description: Ask questions about an uploaded text or PDF file
---

# RAG Application with FastAPI, LangChain and React

This project implements a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about uploaded text and PDF files. The app consists of a FastAPI backend using LangChain and a React frontend.

## Features

- Upload text (.txt) and PDF (.pdf) files up to 2MB in size
- Process documents using text splitting and embeddings
- Chat with your document through a conversational interface
- Streams responses in real-time as they're generated
- Responsive, modern UI built with Material UI

## Project Structure

```
.
├── backend/                # FastAPI backend
│   ├── main.py             # Main API implementation
│   ├── requirements.txt    # Python dependencies
│   └── aimakerspace/       # Utility modules
├── frontend/               # React frontend
│   ├── public/             # Public assets
│   └── src/                # React source code
│       ├── components/     # React components
│       └── App.js          # Main React application
├── Dockerfile              # Docker configuration
└── docker-build-verify.sh  # Script for building and verifying Docker image
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- Node.js 20+
- npm
- Docker (optional, for containerized deployment)

### Environment Setup

Create a `.env` file in the root directory with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Development Setup

#### Backend Setup

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 7860
   ```

#### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm run start
   ```

4. The app will open in your browser at `http://localhost:3000`

### Production Deployment

#### Option 1: Docker Deployment

1. Build and verify the Docker image:

   ```bash
   chmod +x docker-build-verify.sh
   ./docker-build-verify.sh
   ```

2. This script will:

   - Build the Docker image
   - Verify that the frontend build files are correctly located

3. Run the Docker container:

   ```bash
   docker run -p 7860:7860 -e OPENAI_API_KEY=your_openai_api_key_here langchain-rag:latest
   ```

4. Access the application at `http://localhost:7860`

#### Option 2: Separate Frontend and Backend Deployment

For more advanced production deployments, you can deploy the frontend and backend separately:

1. Backend:

   ```bash
   cd backend
   pip install -r requirements.txt
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app -b 0.0.0.0:7860
   ```

2. Frontend:

   ```bash
   cd frontend
   npm install
   npm run build
   ```

3. Serve the frontend using a static file server like Nginx or a service like Vercel, Netlify, etc.

## Usage

1. Upload a text or PDF file using the file upload interface
2. Wait for the file to be processed
3. Ask questions about the content of the document
4. View the AI's responses in the chat interface
5. Use the Reset button to upload a different file

## API Endpoints

The backend provides the following API endpoints:

- `POST /upload` - Upload a file and create a new session
- `POST /query` - Send a query to be answered based on the uploaded document

## Technologies Used

- **Backend:**

  - FastAPI
  - OpenAI API
  - PyPDF2 for PDF processing
  - Qdrant for vector database
  - LangChain

- **Frontend:**
  - React
  - Material UI
  - Fetch API for streaming responses
  - Modern ES6+ JavaScript

## Production Considerations

When deploying to production, consider the following:

1. **Environment Variables**: Ensure all sensitive information (like API keys) are set as environment variables, not hardcoded.

2. **CORS Settings**: Update the CORS middleware in `main.py` to only allow requests from your frontend domain.

3. **Error Handling**: Implement proper error handling and logging for production.

4. **Scaling**: For high-traffic applications, consider deploying behind a load balancer and implementing caching.

5. **Monitoring**: Add application monitoring and logging for production use.

## Troubleshooting

- If you encounter issues with Docker, verify the file paths using `docker-build-verify.sh`
- For OpenAI API issues, check your API key and ensure it has sufficient quota
- For streaming issues, ensure your server and proxy configurations support streaming responses

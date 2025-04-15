from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import tempfile
import shutil
from typing import Dict, List, Optional
import asyncio
import uvicorn
from pydantic import BaseModel

# Import utility modules from aimakerspace
import sys
from aimakerspace.text_utils import TextFileLoader, PDFLoader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

sys.path.append('..')
# Determine the frontend build directory 
# If in Docker, the frontend build is at /app/frontend/build
# If running locally, use relative path ../frontend/build
FRONTEND_BUILD_DIR = "/app/frontend/build" if os.path.exists("/app/frontend/build") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "build")
FRONTEND_STATIC_DIR = os.path.join(FRONTEND_BUILD_DIR, "static")
FRONTEND_INDEX_HTML = os.path.join(FRONTEND_BUILD_DIR, "index.html")


app = FastAPI(title="RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDING_DIM = 1536
session = { "status": "idle" }
text_splitter = RecursiveCharacterTextSplitter(keep_separator='end', chunk_size=400, chunk_overlap=100)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
openai_chat_model = ChatOpenAI(model="gpt-4o-mini")
retriever = None

HUMAN_TEMPLATE = """
CONTEXT:
{context}

QUERY:
{query}

Use the provided context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context response with "I don't know"
"""

def chunk_file(file_path: str, file_name: str):
    # Create appropriate loader based on file extension
    if file_name.lower().endswith('.pdf'):
        loader = PDFLoader(file_path)
    else:
        loader = TextFileLoader(file_path)
        
    # Load and process the documents
    documents = loader.load_documents()
    return text_splitter.split_text(documents[0])


class QueryRequest(BaseModel):
    query: str


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Validate file type
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.txt', '.pdf']:
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")
    
    # Create a temporary file with the correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        # Copy the uploaded file content to the temporary file
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    result = await process_file_task(temp_file_path, file.filename)
    return result


async def process_file_task(file_path: str, file_name: str):
    global retriever
    try:
        # Chunk the file and embed the chunks into a vector database
        chunks = chunk_file(file_path, file_name)

        client = QdrantClient(":memory:")
        collection_name = "upload_collection"
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embedding_model,
        )

        vector_store.add_texts(chunks)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        session["status"] = "ready"
        return {"status": "ready"} 

    except Exception as e:
        session["status"] = "error"
        return {"status": "error", "error": str(e)}
    finally:
        # Clean up the temporary file
        try:
            os.unlink(file_path)
        except Exception:
            pass


@app.post("/query")
async def query(request: QueryRequest):
    query_text = request.query
    
    if session["status"] != "ready":
        raise HTTPException(status_code=400, detail=f"Not ready!")
    
    try:
        chat_prompt = ChatPromptTemplate.from_messages([
            ("human", HUMAN_TEMPLATE)
        ])

        chain  = (
            {"context": retriever, "query": RunnablePassthrough()}
            | chat_prompt
            | openai_chat_model
            | StrOutputParser()
        )
        
        # Create a streaming response
        async def stream_response():
            try:
                async for chunk in chain.astream(query_text):
                    print(chunk, end="", flush=True)
                    yield chunk                    
            except Exception as e:
                # Log the error but don't raise it to avoid breaking the stream
                print(f"Error in streaming response: {str(e)}")
                yield f"\n\nError during response generation: {str(e)}"
        
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",  # Disable buffering for Nginx
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    except Exception as e:
        # Handle exceptions during pipeline execution
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Mount the frontend build folder
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(FRONTEND_INDEX_HTML)

# Catch-all route to serve React Router paths
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_react_app(full_path: str):
    # If the path is an API endpoint, skip this handler
    if full_path.startswith("upload") or full_path.startswith("query"):
        raise HTTPException(status_code=404, detail="Not found")
    
    # Check if a static file exists in the build folder
    static_file_path = os.path.join(FRONTEND_BUILD_DIR, full_path)
    if os.path.isfile(static_file_path):
        return FileResponse(static_file_path)
    
    # Otherwise, serve the index.html for client-side routing
    return FileResponse(FRONTEND_INDEX_HTML)

# Mount static files (JavaScript, CSS, images)
app.mount("/static", StaticFiles(directory=FRONTEND_STATIC_DIR), name="static")

# Main entry point
if __name__ == "__main__":
    # Make sure the frontend build folder exists
    if not os.path.exists("../frontend/build"):
        print("Frontend build directory not found. Building frontend...")
        # Build the frontend
        os.chdir("../frontend")
        os.system("npm install")
        os.system("npm run build")
        os.chdir("../backend")

    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True) 
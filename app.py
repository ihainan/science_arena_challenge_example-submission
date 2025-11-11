import os
import json
import base64
from typing import Optional, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import numpy as np

# Load environment variables
load_dotenv()

app = FastAPI(title="Science Arena Challenge Example Submission")

# Initialize AsyncOpenAI client for LLM models
client = AsyncOpenAI(
    base_url=os.getenv("SCI_MODEL_BASE_URL"),
    api_key=os.getenv("SCI_MODEL_API_KEY")
)

# Initialize AsyncOpenAI client for embedding model
embedding_client = AsyncOpenAI(
    base_url=os.getenv("SCI_EMBEDDING_BASE_URL"),
    api_key=os.getenv("SCI_EMBEDDING_API_KEY")
)


def extract_pdf_text_from_base64(pdf_b64: str) -> str:
    """
    Extract text from base64-encoded PDF using PyPDF2
    """
    try:
        pdf_bytes = base64.b64decode(pdf_b64)
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        return "\n".join(pages)

    except Exception as e:
        print(f"PDF parsing error: {str(e)}")
        return ""


async def get_embedding(text: str) -> List[float]:
    """
    Get embedding vector for text using embedding model
    """
    try:
        response = await embedding_client.embeddings.create(
            model=os.getenv("SCI_EMBEDDING_MODEL"),
            input=text
        )
        embedding = response.data[0].embedding
        # Log embedding results (truncated)
        print(f"[get_embedding] Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"[get_embedding] Embedding dimension: {len(embedding)}")
        print(f"[get_embedding] Embedding (first 5 values): {embedding[:5]}")
        return embedding
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return []


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    """
    if not vec1 or not vec2:
        return 0.0

    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for non-streaming endpoints
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc)
        }
    )


@app.post("/literature_review")
async def literature_review(request: Request):
    """
    Literature review endpoint - uses standard LLM model

    Request body:
    {
        "query": "What are the latest advances in transformer models?"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        print(f"[literature_review] Received query: {query}")
        print(f"[literature_review] Using model: {os.getenv('SCI_LLM_MODEL')}")

        async def generate():
            # Prepare prompt for literature review
            prompt = f"""Conduct a literature review on the following topic:

{query}"""

            # Call LLM model with streaming
            stream = await client.chat.completions.create(
                model=os.getenv("SCI_LLM_MODEL"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.2,
                stream=True
            )

            # Stream back results
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        response_data = {
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {
                                    "content": delta_content
                                }
                            }]
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/paper_qa")
async def paper_qa(request: Request):
    """
    Paper Q&A endpoint - uses reasoning model with PDF content

    Request body:
    {
        "query": "Please carefully analyze and explain the reinforcement learning training methods used in this article.",
        "pdf_content": "base64_encoded_pdf_content"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        pdf_content = body.get("pdf_content", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "pdf_content is required"}
            )

        print(f"[paper_qa] Received query: {query}")
        print(f"[paper_qa] Using reasoning model: {os.getenv('SCI_LLM_REASONING_MODEL')}")

        async def generate():
            # Extract text from PDF
            text = extract_pdf_text_from_base64(pdf_content)

            # Build prompt with PDF content
            prompt = f"""Answer the question based on the paper content.

Paper:
{text}

Question: {query}"""

            # Call reasoning model with streaming
            stream = await client.chat.completions.create(
                model=os.getenv("SCI_LLM_REASONING_MODEL"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.2,
                stream=True
            )

            # Stream back results
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # Extract and log reasoning content
                    reasoning_content = getattr(delta, 'reasoning_content', None)
                    if reasoning_content:
                        print(f"[paper_qa] Reasoning: {reasoning_content}", flush=True)

                    # Stream regular content to client
                    delta_content = delta.content
                    if delta_content:
                        response_data = {
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {
                                    "content": delta_content
                                }
                            }]
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/ideation")
async def ideation(request: Request):
    """
    Ideation endpoint - uses embedding model for similarity and LLM for generation

    Request body:
    {
        "query": "Generate research ideas about climate change"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        # Hardcoded reference ideas for testing embedding model
        reference_ideas = [
            "Using deep learning to predict protein folding structures",
            "Applying transformer models to drug discovery and molecular design",
            "Leveraging reinforcement learning for automated experiment design",
            "Developing AI-powered literature review and knowledge synthesis tools",
            "Creating neural networks for climate modeling and weather prediction",
            "Using machine learning to analyze large-scale genomic datasets"
        ]

        print(f"[ideation] Received query: {query}")
        print(f"[ideation] Using {len(reference_ideas)} hardcoded reference ideas for embedding similarity")
        print(f"[ideation] Using LLM model: {os.getenv('SCI_LLM_MODEL')}")
        print(f"[ideation] Using embedding model: {os.getenv('SCI_EMBEDDING_MODEL')}")

        async def generate():
            prompt = f"""Generate innovative research ideas for:

{query}"""

            # Use embedding model to find similarities with hardcoded reference ideas
            print("[ideation] Computing embeddings for similarity analysis...")

            # Get embedding for query
            query_embedding = await get_embedding(query)

            # Get embeddings for reference ideas and compute similarities
            similarities = []
            for idx, idea in enumerate(reference_ideas):
                idea_embedding = await get_embedding(idea)
                similarity = cosine_similarity(query_embedding, idea_embedding)
                similarities.append((idx, idea, similarity))

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[2], reverse=True)

            # Add similarity analysis to prompt
            prompt += f"\n\nReference ideas (ranked by similarity):\n"
            for idx, idea, sim in similarities:
                prompt += f"\n{idx+1}. (similarity: {sim:.3f}) {idea}"

            prompt += "\n\nGenerate novel research ideas based on the above."

            # Call LLM model with streaming
            stream = await client.chat.completions.create(
                model=os.getenv("SCI_LLM_MODEL"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.2,
                stream=True
            )

            # Stream back results
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        response_data = {
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {
                                    "content": delta_content
                                }
                            }]
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.post("/paper_review")
async def paper_review(request: Request):
    """
    Paper review endpoint - uses LLM model with PDF content

    Request body:
    {
        "query": "Please review this paper",  # optional, default review prompt will be used
        "pdf_content": "base64_encoded_pdf_content"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "Please provide a comprehensive review of this paper")
        pdf_content = body.get("pdf_content", "")

        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "pdf_content is required"}
            )

        print(f"[paper_review] Received query: {query}")
        print(f"[paper_review] Using model: {os.getenv('SCI_LLM_MODEL')}")

        async def generate():
            # Extract text from PDF
            text = extract_pdf_text_from_base64(pdf_content)

            # Build prompt with PDF content
            prompt = f"""Review the following paper:

Paper:
{text}

Instruction: {query}"""

            # Call LLM model with streaming
            stream = await client.chat.completions.create(
                model=os.getenv("SCI_LLM_MODEL"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.2,
                stream=True
            )

            # Stream back results
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        response_data = {
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {
                                    "content": delta_content
                                }
                            }]
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)

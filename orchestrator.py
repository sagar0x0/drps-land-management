import os
import json
import re
import asyncio
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from asyncio import Semaphore
from fastapi.responses import JSONResponse
import yaml 
from pathlib import Path
from io import BytesIO


class Orchestrator:
    def __init__(self):
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self):
        AGENT_URLS = {
            "fraud_agent":       "http://localhost:8001/comprehensive-fraud-analysis",
            "verification_agent":"http://localhost:8002/comprehensive-verification",
            "valuation_agent":   "http://localhost:8003/comprehensive-valuation",
        }

        @self.app.post("/send-doc")
        async def send_doc():
            """
            Send a PDF document path to all agents for comprehensive analysis.
            The agents are expected to access the file using this path.
            """
            pdf_path_str = "/Users/sagar/Desktop/Hackathon/Perplexity/ai-swarm-agent/data/SaleDeed.pdf" # This path must be accessible by all agents

            # 1. Check file existence from the orchestrator's perspective (optional, but good practice)
            if not os.path.isfile(pdf_path_str):
                raise HTTPException(status_code=404, 
                                    detail=f"File not found by orchestrator: {pdf_path_str}. "
                                           "Ensure it's also accessible by agents at this exact path.")

            # 2. Call agents concurrently, sending the path string
            async with httpx.AsyncClient() as client:
                async def call_agent(name: str, url: str):
                    # Send the path as form data with the key "pdf_path"
                    form_data = {"pdf_path": pdf_path_str}
                    try:
                        # Increased timeout for potentially longer agent processing
                        resp = await client.post(url, data=form_data, timeout=180)
                        resp.raise_for_status()
                        return name, {"status": resp.status_code, "data": resp.json()}
                    except httpx.TimeoutException:
                        return name, {"status": 504, "error": f"Request to {name} timed out."}
                    except httpx.HTTPStatusError as e:
                        # Try to get error detail from agent's response if available
                        error_detail = str(e)
                        try:
                            agent_error = e.response.json()
                            if "detail" in agent_error:
                                error_detail = f"{str(e)} - Agent detail: {agent_error['detail']}"
                        except ValueError: # Not JSON
                            pass
                        return name, {"status": e.response.status_code, "error": error_detail}
                    except httpx.RequestError as e:
                        return name, {"status": 502, "error": f"Could not connect to {name}: {str(e)}"}

                tasks = [call_agent(name, url) for name, url in AGENT_URLS.items()]
                results = await asyncio.gather(*tasks)

            # 3. Aggregate and return
            aggregated = {name: result for name, result in results}
            return JSONResponse(content=aggregated)

    def run(self, host: str = "0.0.0.0", port: int = 8004, reload: bool = False):
        """Run the FastAPI service with uvicorn programmatically."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    Orchestrator().run()
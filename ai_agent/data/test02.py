import asyncio
import httpx

async def call_extract():
    url = "http://localhost:8002/extract-document-info"
    params = {
        "pdf_path": "/Users/sagar/Desktop/Hackathon/Perplexity/ai-swarm-agent/data/SaleDeed.pdf"
    }

    # Set a 60-second total timeout (you can customize connect/read/write separately if needed)
    timeout = httpx.Timeout(60.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"Server returned error {e.response.status_code}: {e.response.text}")
            return None
        except httpx.RequestError as e:
            print(f"Request failed: {e}")
            return None

        return resp.json()

if __name__ == "__main__":
    result = asyncio.run(call_extract())
    if result is not None:
        print(result)


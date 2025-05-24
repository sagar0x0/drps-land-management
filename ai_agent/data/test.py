# client.py
import httpx

def call_send_doc():
    url = "http://localhost:8004/send-doc"  # adjust port if needed
    try:
        response = httpx.post(url, timeout=180)
        response.raise_for_status()
    except httpx.HTTPError as e:
        print(f"Request failed: {e}")
        return

    print("Status:", response.status_code)
    print("Response JSON:")
    print(response.json())

if __name__ == "__main__":
    call_send_doc()

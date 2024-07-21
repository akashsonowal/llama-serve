import httpx
import asyncio
import random

# Define the endpoint and payload
url = "http://localhost:7000/endpoint"
payload = {"prompt": "I am Akash Sonowal", "new_tokens": 500}

async def send_request(client):
    retries = 3  # Number of retries for each request
    for attempt in range(retries):
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()
        except (httpx.ReadTimeout, httpx.HTTPStatusError) as e:
            if attempt < retries - 1:
                wait_time = random.uniform(1, 3)  # Random wait time between retries
                await asyncio.sleep(wait_time)
                continue  # Retry the request
            else:
                print(f"Request failed after {retries} attempts: {e}")
                return None

async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        tasks = [send_request(client) for _ in range(64)]  # Simulate 10 concurrent users
        responses = await asyncio.gather(*tasks)
        print(len(responses))

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
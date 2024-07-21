from locust import HttpUser, task

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

class LLMServiceUser(HttpUser):
    @task
    def endpoint(self):
        data = {"prompt": "Generate a short story about a mysterious adventure.", "new_tokens": 100}
        response = self.client.post("/", headers=headers, json=data)
        response.raise_for_status()
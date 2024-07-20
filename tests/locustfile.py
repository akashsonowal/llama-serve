from locust import HttpUser, task

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

class LLMServiceUser(HttpUser):
    @task
    def endpoint(self):
        # data = {"prompt": "Generate a short story about a mysterious adventure."}
        response = self.client.post("/", headers=headers)#, json=data)
        response.raise_for_status()
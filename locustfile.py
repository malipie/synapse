from locust import HttpUser, task, between
import json
import random

class MedicalUser(HttpUser):
    # User "think time" between actions (1-5 seconds)
    wait_time = between(1, 5)

    @task(1)
    def health_check(self):
        """Checks if the API is alive (lightweight test)"""
        self.client.get("/api/v1/chat/health") # Ensure you have this endpoint or another lightweight one

    @task(3)
    def chat_request(self):
        """Simulates a standard conversation (Intent: CHAT)"""
        payload = {
            "messages": [{"role": "user", "content": "Cześć, kim jesteś?"}],
            "model": "gpt-3.5-turbo"
        }
        self.client.post("/api/v1/chat/", json=payload, name="/chat (Simple)")

    @task(1)
    def rag_request(self):
        """Simulates a heavy document query (Intent: RAG)"""
        # In MOCK mode this will always go as RAG
        payload = {
            "messages": [{"role": "user", "content": "Analizuj ten przypadek kliniczny"}],
            "model": "gpt-3.5-turbo"
        }
        
        # 1. Send request
        with self.client.post("/api/v1/chat/", json=payload, catch_response=True, name="/chat (RAG Trigger)") as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("intent") == "RAG":
                    job_id = data.get("job_id")
                    # 2. Polling (asking for result) - frontend simulation
                    self.poll_result(job_id)
                else:
                    response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    def poll_result(self, job_id):
        """Simulates polling for task status"""
        for _ in range(10): # Try 10 times
            with self.client.get(f"/api/v1/chat/tasks/{job_id}", catch_response=True, name="/tasks/{id}") as res:
                if res.status_code == 200:
                    job_data = res.json()
                    if job_data["status"] == "complete":
                        res.success() # Mark success manually
                        return
                elif res.status_code == 404:
                    # If task doesn't exist yet (e.g. worker starts slowly), don't panic
                    res.failure("Task not found yet")
            
            # Wait a moment before next poll
            import time
            time.sleep(1)
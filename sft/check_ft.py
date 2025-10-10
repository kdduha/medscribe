from together import Together
import os

client = Together(
    api_key=os.environ.get("TOGETHER_API_KEY"),
)

response = client.fine_tuning.list()
for fine_tune in response.data:
    print(f"ID: {fine_tune.id}, Status: {fine_tune.status}")
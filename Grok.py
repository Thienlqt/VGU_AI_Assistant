import os
import json
import logging
import requests
from typing import Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
#first commit
load_dotenv()

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GrokHelper:
    def __init__(self):
        self.api_key = os.getenv("GROK_KEY")
        if not self.api_key:
            raise ValueError("GROK_KEY environment variable is not set")

        self.base_url = "https://api.x.ai/v1/chat/completions"

        try:
            with open("Data/qa_data_fewshot_updated.json", encoding="utf-8") as f:
                self.few_shot_examples = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("qa_data_fewshot_updated.json not found")


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def call_model(self, user_input: str) -> Optional[str]:

        system_prompt = self.few_shot_examples.get("system_prompt", {})
        examples = self.few_shot_examples.get("examples", {})

        messages = [{"role": "system", "content": system_prompt}] + examples
        messages.append({"role": "user", "content": user_input})

        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://your-site.com",
                    "X-Title": "VGU Chatbot"
                },
                json={
                    "model": "grok-3-mini-beta",
                    "messages": messages,
                    "stream": False,
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            )
            logger.info("Sending request to Grok API...")
            logger.debug("Request payload: %s", json.dumps(json, ensure_ascii=False, default=str))

            logger.info(f"Grok API response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()

            logger.debug(f"Grok response: {json.dumps(data, ensure_ascii=False)[:500]}")

            usage = data.get("usage", {})
            logger.info(f"Grok usage: {usage.get('total_tokens', 0)} tokens "
                        f"(prompt: {usage.get('prompt_tokens', 0)}, "
                        f"completion: {usage.get('completion_tokens', 0)})")

            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                logger.error(f"Unexpected response format: {data}")
                return None

        except requests.exceptions.HTTPError as e:
            self.handle_http_error(e, response)
            return None
        except Exception as e:
            logger.error(f"xAI API error: {e}")
            return None

    def handle_http_error(self, error, response):
        try:
            status = response.status_code
            error_data = response.json()

            if status == 429:
                logger.error("Rate limit exceeded.")
            elif status == 401:
                logger.error("401 Unauthorized. Invalid API key?")
            elif status == 503:
                logger.error("503 Service Unavailable. Too many users. Try again later.")
            elif status == 404:
                logger.error(f"404 Not Found. Check your endpoint: {self.base_url}")
            else:
                logger.error(f"HTTP {status} Error: {error_data}")
        except Exception:
            logger.error(f"Unhandled HTTP error: {error}")

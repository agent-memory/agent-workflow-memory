import os
import base64
import openai
import numpy as np
from PIL import Image
from typing import Union, Optional
from openai import OpenAI, ChatCompletion
from huggingface_hub import InferenceClient

class LM_Client:
    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        self.model_name = model_name
        if "gpt" in model_name:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")
            self.client = OpenAI()
        else:
            # Assuming HF Inf models by default
            print(f"HF Inf model load: {self.model_name}")
            self.client = InferenceClient(self.model_name, token = os.environ.get('HF_TOKEN'))

    def chat(self, messages, json_mode: bool = False) -> tuple[str, ChatCompletion]:
        """
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ])
        """
        # HFInfClient has an alias to match OpenAI API format
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"} if json_mode else None,
            temperature=0,
            max_tokens=8_000,
        )
        response = chat_completion.choices[0].message.content
        return response, chat_completion

    def one_step_chat(
        self, text, system_msg: str = None, json_mode=False
    ) -> tuple[str, ChatCompletion]:
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, json_mode=json_mode)


class GPT4V_Client:
    def __init__(self, model_name: str = "gpt-4o", max_tokens: int = 512):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = OpenAI()

    def encode_image(self, path: str):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
                         
    def one_step_chat(
        self, text, image: Union[Image.Image, np.ndarray], 
        system_msg: Optional[str] = None,
    ) -> tuple[str, ChatCompletion]:
        jpg_base64_str = self.encode_image(image)
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages += [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}"},},
                ],
        }]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content, response


CLIENT_DICT = {
    "gpt-3.5-turbo": LM_Client,
    "gpt-4": LM_Client,
    "gpt-4o": GPT4V_Client,
    "meta-llama/Llama-3.1-70B-Instruct": LM_Client,
    "meta-llama/Llama-3.1-8B-Instruct": LM_Client
}
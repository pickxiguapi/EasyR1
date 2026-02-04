# Copyright 2025 POLAR Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from time import sleep

import requests
from transformers import AutoTokenizer


# Config for Skywork reward model
MODEL_PATH = "Skywork/Skywork-Reward-V2-Qwen3-4B"


class RewardModelClient:
    """Reward client for Skywork reward model using sglang server."""

    def __init__(self, model_path=MODEL_PATH, port=12321, server_num=1):
        """
        Args:
            model_path: Path to the Skywork reward model.
            base_urls: List of server URLs for the sglang /classify endpoint.
        """
        self.model_path = model_path
        self.base_urls = [f"http://127.0.0.1:{port + i}/classify" for i in range(server_num)]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.current_url_idx = 0

    def _get_next_url(self):
        """Get next URL for load balancing."""
        url = self.base_urls[self.current_url_idx]
        self.current_url_idx = (self.current_url_idx + 1) % len(self.base_urls)
        return url

    def __call__(self, convs, base_url=None, retry_delay=0.2, max_retries=5, timeout=20):
        """Process conversations and return reward scores.

        Args:
            convs: List of conversations, where each conversation is a list of messages
                   with 'role' and 'content' keys.
            base_url: Optional specific server URL. If None, uses load balancing.
            retry_delay: Delay in seconds before retrying the request.
            max_retries: Maximum number of retries for the request.
            timeout: Request timeout in seconds.

        Returns:
            List of reward scores, or list of None values if error occurs.
        """
        if base_url is None:
            base_url = self._get_next_url()

        payload = {"model": self.model_path}
        convs_formatted = []
        for conv in convs:
            conv_str = self.tokenizer.apply_chat_template(conv, tokenize=False)
            if self.tokenizer.bos_token is not None and conv_str.startswith(self.tokenizer.bos_token):
                conv_str = conv_str[len(self.tokenizer.bos_token):]
            convs_formatted.append(conv_str)

        payload.update({"text": convs_formatted})

        # Retry logic
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    base_url,
                    json=payload,
                    proxies={"http": None, "https": None},  # Disable proxy
                    timeout=timeout
                )
                response.raise_for_status()
                rewards = [item["embedding"][0] for item in response.json()]
                assert len(rewards) == len(convs), f"Expected {len(convs)} rewards, got {len(rewards)}"
                return rewards
            except Exception as e:
                print(f"Error requesting reward (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    sleep(retry_delay)
                else:
                    print(f"Failed to request reward after {max_retries} retries")
                    return [None] * len(convs)


if __name__ == "__main__":
    client = RewardClient()

    print("\n" + "=" * 50)
    print("Batch Test: 10 conversations with reference answers")
    print("=" * 50)

    # Test cases with prompt, reference answer, and candidate responses
    test_cases = [
        {
            "prompt": "What is 2+2?",
            "reference": "4",
            "responses": {
                "correct": "2+2 equals 4.",
                "incorrect": "2+2 equals 5."
            }
        },
        {
            "prompt": "What is the capital of France?",
            "reference": "Paris",
            "responses": {
                "correct": "The capital of France is Paris.",
                "incorrect": "The capital of France is London."
            }
        },
        {
            "prompt": "How many days are in a week?",
            "reference": "7 days",
            "responses": {
                "correct": "There are 7 days in a week.",
                "incorrect": "There are 5 days in a week."
            }
        },
        {
            "prompt": "What color is the sky?",
            "reference": "Blue",
            "responses": {
                "correct": "The sky is blue.",
                "incorrect": "The sky is green."
            }
        },
        {
            "prompt": "How many legs does a dog have?",
            "reference": "4 legs",
            "responses": {
                "correct": "A dog has 4 legs.",
                "incorrect": "A dog has 6 legs."
            }
        },
    ]

    # Build conversations for batch processing
    batch_convs = []
    test_info = []  # Track which test case each conversation belongs to

    for test_case in test_cases:
        prompt = test_case["prompt"]
        reference = test_case["reference"]

        # Include reference in the prompt to help model evaluate
        prompt_with_ref = f"{prompt}\n\nReference answer: {reference}"

        for response_type, response in test_case["responses"].items():
            conv = [
                {"role": "user", "content": prompt_with_ref},
                {"role": "assistant", "content": response}
            ]
            batch_convs.append(conv)
            test_info.append({
                "prompt": prompt,
                "reference": reference,
                "response_type": response_type,
                "response": response
            })

    print(f"Processing {len(batch_convs)} conversations ({len(test_cases)} test cases)...")

    # Measure inference time
    start_time = time.time()
    batch_rewards = client(batch_convs)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"Average time per conversation: {inference_time / len(batch_convs):.3f} seconds")
    print(f"Throughput: {len(batch_convs) / inference_time:.2f} conversations/second")

    if batch_rewards:
        print("\nResults:")
        print("-" * 80)
        for i, (info, reward) in enumerate(zip(test_info, batch_rewards)):
            status = "✓" if info["response_type"] == "correct" else "✗"
            print(f"{status} [{info['response_type'].upper()}] Score: {reward:.3f}")
            print(f"  Q: {info['prompt']}")
            print(f"  A: {info['response']}")
            print(f"  Ref: {info['reference']}")
            print()
    else:
        print("Failed to get rewards.")

    # ========== Batch=100 Test ==========
    print("\n" + "=" * 50)
    print("Batch Test: 100 conversations")
    print("=" * 50)

    # Replicate the conversations 10 times to get 100 conversations
    batch_100_convs = batch_convs * 10
    batch_100_info = test_info * 10

    print(f"Processing {len(batch_100_convs)} conversations...")

    # Measure inference time
    start_time = time.time()
    batch_100_rewards = client(batch_100_convs)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"\nInference time: {inference_time:.3f} seconds")
    print(f"Average time per conversation: {inference_time / len(batch_100_convs):.3f} seconds")
    print(f"Throughput: {len(batch_100_convs) / inference_time:.2f} conversations/second")

    if batch_100_rewards:
        # Calculate statistics
        valid_rewards = [r for r in batch_100_rewards if r is not None]
        correct_rewards = [r for i, r in enumerate(batch_100_rewards) if r is not None and batch_100_info[i]["response_type"] == "correct"]
        incorrect_rewards = [r for i, r in enumerate(batch_100_rewards) if r is not None and batch_100_info[i]["response_type"] == "incorrect"]

        print("\nStatistics:")
        print("-" * 80)
        print(f"Total conversations: {len(batch_100_convs)}")
        print(f"Valid rewards: {len(valid_rewards)}")
        print(f"Failed: {len(batch_100_convs) - len(valid_rewards)}")

        if correct_rewards:
            print(f"\nCorrect answers:")
            print(f"  Count: {len(correct_rewards)}")
            print(f"  Average score: {sum(correct_rewards) / len(correct_rewards):.3f}")
            print(f"  Min: {min(correct_rewards):.3f}, Max: {max(correct_rewards):.3f}")

        if incorrect_rewards:
            print(f"\nIncorrect answers:")
            print(f"  Count: {len(incorrect_rewards)}")
            print(f"  Average score: {sum(incorrect_rewards) / len(incorrect_rewards):.3f}")
            print(f"  Min: {min(incorrect_rewards):.3f}, Max: {max(incorrect_rewards):.3f}")

        if correct_rewards and incorrect_rewards:
            avg_correct = sum(correct_rewards) / len(correct_rewards)
            avg_incorrect = sum(incorrect_rewards) / len(incorrect_rewards)
            print(f"\nScore difference (correct - incorrect): {avg_correct - avg_incorrect:.3f}")
    else:
        print("Failed to get rewards.")

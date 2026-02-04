import requests
from transformers import AutoTokenizer


model_name_or_path = "Skywork/Skywork-Reward-V2-Qwen3-4B"
base_urls = "http://127.0.0.1:12321/classify"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def process_convs(convs, base_url, tokenizer, model_name_or_path):
    payload = {"model": model_name_or_path}
    convs_formatted = []
    for conv in convs:
        conv = tokenizer.apply_chat_template(conv, tokenize=False)
        if tokenizer.bos_token is not None and conv.startswith(tokenizer.bos_token):
            conv = conv[len(tokenizer.bos_token):]
        convs_formatted.append(conv)

    payload.update({"text": convs_formatted})
    print(payload)
    rewards = []
    try:
        responses = requests.post(base_url, json=payload).json()
        for response in responses:
            rewards.append(response["embedding"][0])
        assert len(rewards) == len(
            convs
        ), f"Expected {len(convs)} rewards, got {len(rewards)}"
        return rewards
    except Exception as e:
        print(f"Error: {e}")
        return [None] * len(convs)
    

prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
response1 = """1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.
2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.
3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."""
response2 = """1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.
2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.
3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."""


conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}]

rewards = process_convs([conv1, conv2], base_urls, tokenizer, model_name_or_path)
print(f"Score for response 1: {rewards[0]}")
print(f"Score for response 2: {rewards[1]}")

# Expected output:
# Score for response 1: 23.125
# Score for response 2: 3.578125

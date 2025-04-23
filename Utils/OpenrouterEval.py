##evaluates an llms response using Deepseek V3 (requires an account on llm-chutes, see api url below). The prompt send to the api can be adjusted in line 21 for task specific evaluation.

import requests

OPENROUTER_API_KEY = "APIKEYHERE"

# OpenRouter API URL
API_URL = "https://llm.chutes.ai/v1/chat/completions"

def evaluateAnswer(text):
    """Generate a 10-point summary of a research paper using OpenRouter."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-ai/DeepSeek-V3-0324",  # Use OpenAI's GPT-4 via OpenRouter  featherless/qwerky-72b:free  deepseek/deepseek-chat-v3-0324:free
        "messages": [
            {"role": "system", "content": "Given is a question, an answer to said question and the solution. Please verify if the answer is correct and reply with a clear and definite YES if it does and NO if it doesnt. Do not answer the question yourself, but only judge if the answer is correct or not."},
            {"role": "user", "content": text}
        ],
        "temperature": 0.2,  # Adjust randomness (0 = strict, 1 = creative)
        "max_tokens": 500  # Limit output length
    }

    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"

##Set path to text file containing the llms responses. IMPORTANT: the log file needs to be structured like this: SLURM messages followed by QUESTION: (question to answer here) RESPONSE: (the LLMS response here) SOLUTION: (the solution to the problem here). 
path = "/path/to/file/.out"

with open(path, 'r') as file:
    txt = file.read()
splitTXT = txt.split("QUESTION:")
correct = 0
false = 0
for x in splitTXT[1:]:
    resp = summarize_research_paper(x)
    found = re.search(r"yes", resp,  re.IGNORECASE)
    if found:
        logging.info("correct: " + resp)
        correct += 1
    else:
        logging.info("wrong: " + resp)
        false+=1
    logging.info("correct: "+ str(correct) + " False: " + str(false))

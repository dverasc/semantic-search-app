import streamlit as st
from typing import List
# Define a function to calculate embeddings
def calculate_embeddings(texts, model):
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings

#define a function to clean up data
def clean_textfiled(df, TEXT_FIELD_NAME):
    # Handle missing or non-string values in the TEXT_FIELD_NAME column
    df[TEXT_FIELD_NAME] = df[TEXT_FIELD_NAME].fillna('')  # Replace NaN with empty string
    df[TEXT_FIELD_NAME] = df[TEXT_FIELD_NAME].astype(str)  # Ensure all values are strings

    df[TEXT_FIELD_NAME] =  df[TEXT_FIELD_NAME].map(lambda x: x.lstrip('Make sure this fits by entering your model number. |').rstrip('aAbBcC'))
    return df

# def conversational_chat(query, chain, sessionstate):
#         result = chain({"question": query, "chat_history": sessionstate})
#         sessionstate.append((query, result["answer"]))
        
#         return result["answer"]

def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)

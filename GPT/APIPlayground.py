import os

from revChatGPT.V3 import Chatbot

# os.environ['API_URL'] = "http://api2.geekerwan.net/"
chatbot = Chatbot(api_key="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", proxy="http://127.0.0.1:7890")
print("Chatbot Start: ")
prev_text = ""
complete_text = ""
for data in chatbot.ask(
        "你现在要回复我一段中文的文字，这段文字需要超过两句话。回复中必须用中文标点。",
):
    message = data
    print(message, end="", flush=True)
    if "。" in message or "！" in message or "？" in message:
        print('')
        print(complete_text)
        complete_text = ""
    else:
        complete_text += message
    prev_text = data
print()

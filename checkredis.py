from langchain_community.chat_message_histories import RedisChatMessageHistory

REDIS_URL = "redis://:ssacgang@35.224.151.92:6379/0"

history = RedisChatMessageHistory("ssac0830", REDIS_URL)

print(history.messages)
# AI Document Chat Bot - RAG Implementation Using OpenAI and Redis VL.

![Redis](https://redis.io/wp-content/uploads/2024/04/Logotype.svg?auto=webp&quality=85,75&width=120)

A Document Chat Bot, with data governance, hybrid search and semantic caching<br>
Set REDIS_HOST, REDIS_PORT, REDIS_PASSWORD as env variables.<br>
This uses gpt-3.5-turbo as the LLM, set OPENAI_API_KEY env variable with your API Key<br>
Run redisVL-app.py and follow the prompts.<br>
You can define user profiles and in the user-profiles.json file.<br>
You can add routes for the semantic router in the llm-router.json file.<br>
rag-q.txt has some sample questions to ask the chat bot.



## Common Tools

- https://redis.io/try-free/ Free Redis Cloud account
- https://redis.io/insight/ Redis Insight
- https://redis.io/download/ Redis Stack

## Documentation

- https://www.redisvl.com/ RedisVL Library
- https://redis-py.readthedocs.io/en/stable/index.html Redis Python client docs

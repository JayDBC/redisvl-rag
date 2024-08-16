import os
import warnings
import redis





def get_connection(REDIS_URL):
    redis_connection = redis.Redis.from_url(REDIS_URL)
    return redis_connection


def main():

    warnings.filterwarnings("ignore")

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_URL = f"redis://default:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"

    #CHECK Connection
    r = get_connection(REDIS_URL)
    print(f"Test Connection: {r.ping()}")



if __name__ == "__main__":
    main()
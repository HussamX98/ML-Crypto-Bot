# src/utils/notifications.py

import requests
import yaml

def load_config():
    """
    Load configuration settings from config/config.yaml.
    """
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def send_telegram_message(message):
    """
    Send a message via Telegram.
    """
    config = load_config()
    telegram_cfg = config.get('telegram')

    if not telegram_cfg:
        raise ValueError("Telegram configuration not found in config/config.yaml under 'telegram'.")

    bot_token = telegram_cfg.get('bot_token')
    chat_id = telegram_cfg.get('chat_id')

    if not bot_token or not chat_id:
        raise ValueError("Telegram bot token or chat ID not provided in the configuration.")

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    params = {
        'chat_id': chat_id,
        'text': message
    }

    try:
        response = requests.post(url, params=params)
        response.raise_for_status()
    except Exception as e:
        print(f"An error occurred while sending Telegram message: {e}")

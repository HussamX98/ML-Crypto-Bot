# scripts/send_alert.py

from scripts.predict import predict_new_tokens
from src.utils.notifications import send_telegram_message

def send_token_alerts():
    # Get predictions
    promising_tokens_df = predict_new_tokens()

    if promising_tokens_df is not None:
        message = "🚀 Promising New Tokens Detected:\n\n"
        for index, row in promising_tokens_df.iterrows():
            token_name = row['token_name']
            token_address = row['token_address']
            message += f"Token: {token_name}\nAddress: {token_address}\n\n"

        send_telegram_message(message)
        print('Alert sent via Telegram.')
    else:
        print('No promising tokens found at this time.')

if __name__ == '__main__':
    send_token_alerts()

#import logging
import predict
from telegram.ext import MessageHandler, CommandHandler, Filters, Updater


#logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                    level=logging.INFO)    
#
#logger = logging.getLogger(__name__)

def chat(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=predict.prediction(update.message.text))

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Merhaba")

#def error(bot, update, error):
#    """Log Errors caused by Updates."""
#    logger.warning('Update "%s" caused error "%s"', update, error)   
#    

def main():

    updater = Updater(token='YOUR TOKEN', use_context=True)
    dispatcher = updater.dispatcher

    chat_handler = MessageHandler(Filters.text, chat)
    dispatcher.add_handler(chat_handler)

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    updater.start_polling() 
    updater.idle()
    


if __name__ == "__main__":
    main()



import re
import torch
import logging
import requests
import joblib
import torch.nn.functional as F
from aiogram import Bot, Dispatcher
from aiogram.types import Message, BotCommand
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from transformers import BertTokenizer, BertForSequenceClassification


TOKEN = "7820899741:AAHFSmAJQDXHaStcR9VXBgwotvNJyrgzxCE"
DEEPL_API_KEY = "45be946e-5438-437f-be2a-7bb839a54eeb:fx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_PATH_1 = "./bert_training/final_training/bert_disease_model"
MODEL_PATH_2 = "./bert_training/kaggle_training/bert_disease_model"
MODEL_PATH_3 = "./bert_training/reddit_training/bert_disease_model"


model_1 = BertForSequenceClassification.from_pretrained(MODEL_PATH_1).to(device)
model_2 = BertForSequenceClassification.from_pretrained(MODEL_PATH_2).to(device)
model_3 = BertForSequenceClassification.from_pretrained(MODEL_PATH_3).to(device)

tokenizer_1 = BertTokenizer.from_pretrained(MODEL_PATH_1)
tokenizer_2 = BertTokenizer.from_pretrained(MODEL_PATH_2)
tokenizer_3 = BertTokenizer.from_pretrained(MODEL_PATH_3)

le_1 = joblib.load(f"{MODEL_PATH_1}/label_encoder.joblib")
le_2 = joblib.load(f"{MODEL_PATH_2}/label_encoder.joblib")
le_3 = joblib.load(f"{MODEL_PATH_3}/label_encoder.joblib")


logging.basicConfig(level=logging.INFO)
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()


def translate_to_english(text: str) -> str:
    url = "https://api-free.deepl.com/v2/translate"
    params = {
        "auth_key": DEEPL_API_KEY,
        "text": text,
        "source_lang": "RU",
        "target_lang": "EN"
    }
    try:
        response = requests.post(url, data=params)
        response.raise_for_status()
        return response.json()["translations"][0]["text"]
    except Exception as e:
        logging.warning(f"Ошибка перевода (RU->EN): {e}")
        return text

def translate_to_russian(text: str) -> str:
    url = "https://api-free.deepl.com/v2/translate"
    params = {
        "auth_key": DEEPL_API_KEY,
        "text": text,
        "target_lang": "RU"
    }
    try:
        response = requests.post(url, data=params)
        response.raise_for_status()
        return response.json()["translations"][0]["text"]
    except Exception as e:
        logging.warning(f"Ошибка перевода (EN->RU): {e}")
        return text


def preprocess_text(text: str) -> str:
    return re.sub(r"[^\w\s,]", "", text).strip()


def predict_disease(symptoms: str, model, tokenizer, label_encoder, top_n: int = 3) -> str:
    model.eval()
    inputs = tokenizer(symptoms, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    top_probs, top_indices = torch.topk(probs, top_n, dim=1)

    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        disease = label_encoder.inverse_transform([idx.cpu().item()])[0]
        predictions.append(f"{disease}: {round(prob.item() * 100, 2)}%")

    return "\n".join(predictions)


async def set_commands(bot: Bot):
    commands = [
        BotCommand(command="/start", description="Запустить бота"),
        BotCommand(command="/help", description="Как использовать бота"),
        BotCommand(command="/diseases", description="Показать список всех болезней")
    ]
    await bot.set_my_commands(commands)

@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer(
        "Привет! Я бот для определения вероятных болезней.\n"
        "Отправь список симптомов на русском, и я покажу возможные диагнозы."
    )

@dp.message(Command("help"))
async def help_handler(message: Message):
    await message.answer(
        "Просто отправь список симптомов, например:\n"
        "жажда, головная боль, утомляемость\n"
        "Я переведу и определю наиболее вероятные болезни."
    )

@dp.message(Command("diseases"))
async def list_diseases(message: Message):
    diseases_text = """
📋 Болезни, которые умеет определять модель:
- (вертиго) пароксизмальное позиционное головокружение
- акне
- СДВГ
- СПИД
- алкогольный гепатит
- аллергия
- тревога
- аритмия
- артрит
- расстройство аутистического спектра
- биполярное расстройство
- бронхиальная астма
- целиакия
- шейный спондилез
- ветряная оспа
- хронический холестаз
- синдром хронической усталости
- простуда
- ковид-19
- лихорадка денге
- депрессия
- диабет
- диморфный гемморрой (сваи)
- лекарственная реакция
- экзема
- эндометриоз
- фибромиалгия
- грибковая инфекция
- гастроэнтерит
- гард
- инфаркт
- гепатит a
- гепатит b
- гепатит c
- гепатит d
- гепатит e
- гипертония
- гипертиреоз
- гипогликемия
- гипотиреоз
- гиповолемический шок
- импетиго
- бессонница
- желтуха
- малярия
- климактерический синдром
- мигрень
- рассеянный склероз xml-ph-0024 ожирение
- ОКР
- остеоартрит
- паралич (кровоизлияние в мозг)
- язвенная болезнь
- пневмония
- псориаз
- псориатический артрит
- посттравматическое стрессовое расстройство
- инсульт
- наркомания
- туберкулез
- диабет 2 типа
- брюшной тиф
- инфекция мочевыводящих путей
- варикозное расширение вен
""".strip()
    await message.answer(diseases_text)


@dp.message()
async def symptoms_handler(message: Message):
    user_text = message.text.strip()

    if not user_text:
        await message.answer("Пожалуйста, отправь симптомы в виде текста.")
        return

    symptoms_clean = preprocess_text(user_text)
    symptoms_en = translate_to_english(symptoms_clean)

    pred1 = predict_disease(symptoms_en, model_1, tokenizer_1, le_1)
    pred2 = predict_disease(symptoms_en, model_2, tokenizer_2, le_2)
    pred3 = predict_disease(symptoms_en, model_3, tokenizer_3, le_3)

    pred1_ru = translate_to_russian(pred1)
    pred2_ru = translate_to_russian(pred2)
    pred3_ru = translate_to_russian(pred3)

    final_response = f"""📊 <b>Ответы от всех моделей:</b>

💡 <b>Финальная модель:</b>
{pred1_ru}

📁 <b>Kaggle модель:</b>
{pred2_ru}

🧠 <b>Reddit модель:</b>
{pred3_ru}
"""
    await message.answer(final_response)


async def main():
    await set_commands(bot)
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

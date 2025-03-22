import praw
import json
import asyncio
import math
import re
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError, InternalServerError
from collections import defaultdict

# ✅ Константы
MIN_SYMPTOMS_PER_ENTRY = 2  # изменено с 3 на 2
TARGET_EXAMPLES_PER_DISEASE = 300
MAX_THREADS_PER_DISEASE = 15
COMMENTS_PER_BATCH = 5
RESULTS_FILE = "dataset.json"

reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent=""
)

my_open_ai_key = ''
client = AsyncOpenAI(api_key=my_open_ai_key)

# diseases = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']




diseases = [
    '(vertigo) paroymsal positional vertigo',
    'acne',
    'adhd',
    'aids',
    'alcoholic hepatitis',
    'allergy',
    'anxiety',
    'arrhythmia',
    'arthritis',
    'autism spectrum disorder',
    'bipolar disorder',
    'bronchial asthma',
    'celiac disease',
    'cervical spondylosis',
    'chicken pox',
    'chronic cholestasis',
    'chronic fatigue syndrome',
    'common cold',
    'covid-19',
    'dengue',
    'depression',
    'diabetes',
    'dimorphic hemmorhoids(piles)',
    'drug reaction',
    'eczema',
    'endometriosis',
    'fibromyalgia',
    'fungal infection',
    'gastroenteritis',
    'gerd',
    'heart attack',
    'hepatitis a',
    'hepatitis b',
    'hepatitis c',
    'hepatitis d',
    'hepatitis e',
    'hypertension',
    'hyperthyroidism',
    'hypoglycemia',
    'hypothyroidism',
    'hypovolemic shock',
    'impetigo',
    'insomnia',
    'jaundice',
    'malaria',
    'menopausal syndrome',
    'migraine',
    'multiple sclerosis',
    'obesity',
    'ocd',
    'osteoarthritis',
    'paralysis (brain hemorrhage)',
    'peptic ulcer disease',
    'pneumonia',
    'psoriasis',
    'psoriatic arthritis',
    'ptsd',
    'stroke',
    'substance abuse',
    'tuberculosis',
    'type 2 diabetes',
    'typhoid',
    'urinary tract infection',
    'varicose veins'
]


system_prompt = {
    "role": "system",
    "content": (
        "You are a medical assistant extracting symptom data from Reddit posts.\n"
        "You will receive the title of a Reddit thread and 5 top-level user comments describing their experiences.\n\n"
        "For each comment, extract one structured entry ONLY IF the user clearly mentions specific symptoms.\n"
        "If no symptoms are described, SKIP the entry.\n"
        "If symptoms are extremely vague (e.g., 'feeling unwell', 'tired'), attempt to make them more specific based on the text — but DO NOT invent symptoms that are not directly or reasonably implied.\n\n"
        "Important: Only extract data for the following diseases (using the exact names):\n"
        + ", ".join(diseases) + "\n\n"
        "Respond with 1 entry per user, in the following format:\n"
        "Disease: <Disease name>, Symptoms: <comma-separated list of symptoms>\n\n"
        "Return only the valid entries. Do not include numbering or any commentary."
    )
}

def load_existing_data():
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def count_valid_examples(data, disease):
    count = 0
    for item in data:
        if item["disease"] != disease:
            continue
        matches = re.findall(r"Symptoms:\s*(.+)", item["raw_response"], flags=re.IGNORECASE)
        for match in matches:
            symptoms = [s.strip() for s in match.split(",")]
            if len(symptoms) >= MIN_SYMPTOMS_PER_ENTRY and not any(s.lower() == "none" for s in symptoms):
                count += 1
    return count

async def request_get_entries(prompt_text, retries=3):
    messages = [
        system_prompt,
        {"role": "user", "content": prompt_text}
    ]
    for attempt in range(1, retries + 1):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            return response.choices[0].message.content

        except (RateLimitError, TimeoutError, APIConnectionError, InternalServerError) as e:
            wait_time = 5 * attempt
            await asyncio.sleep(wait_time)
        except APIError as e:
            print(f" API error")
            raise e

    raise Exception("")

async def process_one_thread(thread, disease, all_data):
    await asyncio.sleep(0.5)
    thread.comments.replace_more(limit=0)
    top_level_comments = [c for c in thread.comments if c.is_root and hasattr(c, "body")]

    groups = []
    for comment in top_level_comments:
        text = comment.body.strip()
        if len(text) > 30:
            groups.append(text)

    for i in range(0, len(groups), COMMENTS_PER_BATCH):
        batch = groups[i:i + COMMENTS_PER_BATCH]
        if len(batch) < COMMENTS_PER_BATCH:
            continue

        numbered = "\n\n".join([f"{j+1}. {text}" for j, text in enumerate(batch)])
        prompt = (
            f"Thread Title: {thread.title}\n\n"
            f"The following are 5 Reddit users describing their experience with {disease}.\n"
            f"For each user, extract one structured entry if symptoms are clearly described (only for the disease exactly as provided):\n\n"
            f"{numbered}"
        )


        try:
            gpt_response = await request_get_entries(prompt)
            all_data.append({
                "disease": disease,
                "thread_title": thread.title,
                "reddit_url": thread.url,
                "comment_batch_index": i // COMMENTS_PER_BATCH + 1,
                "raw_response": gpt_response
            })

            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            continue

async def main():
    all_data = load_existing_data()
    total_batches = len(all_data)

    total_valid = sum(count_valid_examples(all_data, disease) for disease in diseases)
    avg_per_batch = total_valid / total_batches if total_batches > 0 else 0


    total_diseases = len(diseases)
    completed_diseases = 0

    for disease in diseases:

        valid_count = count_valid_examples(all_data, disease)

        # Останавливаем, когда собрано хотя бы 120 данных с 2 симптомами или более
        if valid_count >= TARGET_EXAMPLES_PER_DISEASE:
            completed_diseases += 1
            continue

        threads = reddit.subreddit("all").search(f"{disease} symptoms", sort="relevance", limit=MAX_THREADS_PER_DISEASE)

        for thread_index, thread in enumerate(threads, 1):
            await process_one_thread(thread, disease, all_data)

            valid_count = count_valid_examples(all_data, disease)

            if valid_count >= TARGET_EXAMPLES_PER_DISEASE:
                break

        completed_diseases += 1


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f" Непредвиденная ошибка: {e}")

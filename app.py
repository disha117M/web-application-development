# app.py
from sanic import Sanic, response
from sanic.exceptions import InvalidUsage
import logging
from transformers import pipeline
import os
from motor.motor_asyncio import AsyncIOMotorClient
from statistics import mean, median, stdev

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Sanic application
app = Sanic("SurveyProcessor")

# Load MongoDB URI from environment variables for security
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(mongo_uri)

db = client['survey_database']
collection = db['survey_results']

# Load Hugging Face pipeline for text generation using distilgpt2
generator = pipeline('text-generation', model='distilgpt2', framework='pt')

# MongoDB Connection Test
@app.listener('before_server_start')
async def test_mongodb_connection(app, loop):
    try:
        await client.server_info()
        logging.info("Connected to MongoDB successfully.")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        raise

@app.post("/process-survey")
async def process_survey(request):
    try:
        payload = request.json
        validate_payload(payload)
        insights = generate_insights(payload)
        
        # Store insights in the database asynchronously
        await store_in_database(payload["user_id"], insights)

        return response.json(insights, status=200)
    
    except InvalidUsage as e:
        logging.error(f"Invalid usage: {e}")
        return response.json({"error": str(e)}, status=400)
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return response.json({"description": "Internal Server Error", "status": 500, "message": str(e)}, status=500)

def validate_payload(payload):
    if not isinstance(payload.get("user_id"), str) or len(payload["user_id"]) < 5:
        logging.warning(f"Invalid user_id: {payload.get('user_id')}")
        raise InvalidUsage("Invalid user_id: must be a string with at least 5 characters.")
    
    survey_results = payload.get("survey_results")
    if not isinstance(survey_results, list) or len(survey_results) != 10:
        logging.warning(f"Invalid survey_results: {survey_results}")
        raise InvalidUsage("Invalid survey_results: must contain exactly 10 objects.")
    
    question_numbers = set()
    for result in survey_results:
        if not (1 <= result["question_number"] <= 10):
            logging.warning(f"Invalid question_number: {result['question_number']}")
            raise InvalidUsage("question_number must be between 1 and 10.")
        if result["question_number"] in question_numbers:
            logging.warning(f"Duplicate question_number found: {result['question_number']}.")
            raise InvalidUsage(f"Duplicate question_number found: {result['question_number']}.")
        question_numbers.add(result["question_number"])
        
        if not (1 <= result["question_value"] <= 7):
            logging.warning(f"Invalid question_value: {result['question_value']}")
            raise InvalidUsage("question_value must be between 1 and 7.")

def generate_insights(payload):
    survey_results = payload["survey_results"]
    
    # Calculate insights from survey results
    question_values = [result["question_value"] for result in survey_results]
    
    # Calculate summary statistics
    summary_stats = {
        "mean": round(mean(question_values), 2),
        "median": round(median(question_values), 2),
        "std_dev": round(stdev(question_values), 2) if len(question_values) > 1 else 0
    }

    # Define analysis values based on conditions
    overall_analysis = "certain" if question_values[0] != 7 or question_values[3] >= 3 else "unsure"
    cat_dog = "cats" if question_values[9] > 5 and question_values[8] <= 5 else "dogs"
    fur_value = "long" if summary_stats["mean"] > 5 else "short"
    tail_value = "long" if question_values[6] > 4 else "short"
    description = generate_description(summary_stats["mean"])
    
    return {
        "overall_analysis": overall_analysis,
        "cat_dog": cat_dog,
        "fur_value": fur_value,
        "tail_value": tail_value,
        "description": description,
        "statistics": summary_stats
    }

def generate_description(average):
    try:
        file_name = 'the_value_of_short_hair.txt' if average > 4 else 'the_value_of_long_hair.txt'
        
        with open(file_name, 'r') as file:
            main_content = file.read()

        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read()

        input_text = system_prompt + "\n" + main_content

        # Ensure 'generator' uses the right backend (PyTorch in this case)
        result = generator(input_text, max_length=150, num_return_sequences=1, truncation=True)

        return result[0]['generated_text']

    except Exception as e:
        logging.error(f"Error generating description: {e}")
        raise

async def store_in_database(user_id, insights):
    try:
        record = {
            "user_id": user_id,
            **insights
        }
        await collection.insert_one(record)
        logging.info(f"Stored survey results for user_id: {user_id}")
    except Exception as e:
        logging.error(f"Failed to store data in MongoDB: {e}")
        raise

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

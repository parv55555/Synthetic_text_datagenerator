import json
import random
import asyncio
import litellm
from litellm import completion_cost
from tenacity import retry, stop_after_attempt, wait_exponential

TONES = ["urgent", "casual", "formal", "angry", "confused", "polite", "demanding", "panicked", "friendly", "sarcastic"]
PERSONAS = ["college student", "elderly person", "busy professional", "customer support agent", "scammer", "concerned parent", "manager", "delivery driver", "tech enthusiast"]

def get_api_key_env_var(provider: str) -> str:
    mapping = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google": "GEMINI_API_KEY",
        "Groq": "GROQ_API_KEY",
        "Mistral": "MISTRAL_API_KEY",
        "Together": "TOGETHER_API_KEY"
    }
    return mapping.get(provider, "")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def async_generate_single_example(provider: str, model: str, task: str, category: str, context: str, schema_def: str = "", few_shot: str = "", temperature: float = 0.85, use_diversity: bool = True) -> dict:
    
    tone = random.choice(TONES) if use_diversity else "neutral"
    persona = random.choice(PERSONAS) if use_diversity else "neutral"
    
    prompt = f"""
    You are an expert data generator.
    Task: {task}
    Category/Label to generate: {category}
    """
    if use_diversity:
        prompt += f"\nInject the following traits into the generation:\n- Tone: {tone}\n- Persona: {persona}\nMake it sound natural for this persona and tone."
        
    if schema_def:
        prompt += f"\nTarget Schema:\n{schema_def}\n"
    if few_shot:
        prompt += f"\nFew-Shot Examples:\n{few_shot}\n"
        
    prompt += f"""
    Additional Context:
    {context}
    
    Output ONLY valid JSON matching the exact schema requirements. Do not include markdown formatting or any other text.
    """
    
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    
    raw_text = response.choices[0].message.content.strip()
    
    try:
        cost = completion_cost(completion_response=response)
        if cost is None:
            cost = 0.0
    except:
        cost = 0.0
    
    # Clean up possible markdown code blocks
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
        
    parsed_json = json.loads(raw_text.strip())
    return parsed_json, cost

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
async def async_evaluate_with_llm(provider: str, model: str, data: dict, label: str, task: str) -> dict:
    prompt = f"""
    Evaluate the following generated JSON data intended to fit the category '{label}'.
    Task Context: {task}
    Generated Data: {json.dumps(data)}
    
    Score the data from 1 to 5 based on how well it fulfills the task and represents the category.
    1 = Terrible, completely misses the mark or hallucinates.
    5 = Perfect, highly realistic and exactly what is needed.
    
    Return ONLY a JSON object with this exact structure:
    {{"score": <int>, "reasoning": "<string>"}}
    """
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    raw_text = response.choices[0].message.content.strip()
    
    try:
        cost = completion_cost(completion_response=response)
        if cost is None:
            cost = 0.0
    except:
        cost = 0.0
    
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
        
    result = json.loads(raw_text.strip())
    result["cost"] = cost
    return result

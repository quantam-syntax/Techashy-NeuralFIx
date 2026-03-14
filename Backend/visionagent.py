import ollama
import base64, json
from pathlib import Path

PROMPT = """You are a network equipment technician.
Analyse this image carefully and return ONLY valid JSON, nothing else:
{
  "device_type": "router | switch | modem | access_point | unknown",
  "brand_model": "brand and model string, or null if unclear",
  "led_states": [
    {"label": "WAN", "color": "red | green | amber | off | blinking", "blinking": false}
  ],
  "unplugged_ports": ["port 1", "port 3"],
  "visible_damage": "description of damage, or null",
  "overall_assessment": "one sentence summary of the problem",
  "confidence": 0.85
}
Return ONLY the JSON object. No explanation, no markdown fences."""

def analyse_image(image_bytes: bytes) -> dict:
    # encode image to base64
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # call local LLaVA via Ollama
    response = ollama.generate(
        model="llava-llama3",        # or "llava-llama3"
        prompt=PROMPT,
        images=[img_b64],   # list of base64 strings
        stream=False
    )
    
    raw = response["response"].strip()
    
    # strip markdown fences if model adds them anyway
    raw = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # fallback: model didn't return clean JSON
        return {
            "device_type": "unknown",
            "overall_assessment": raw,  # use raw text as fallback
            "confidence": 0.0,
            "error": "JSON parse failed"
        }

if __name__ == "__main__":
    img_bytes = Path("test_router.jpeg").read_bytes()
    result = analyse_image(img_bytes)
    print(json.dumps(result, indent=2))
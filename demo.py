import aisisax.object_detection.lsa_interface as aisax_object_detection
import aisisax.llm.openai_connector as aisax_openai
import aisisax.llm.ollama_connector as aisax_ollama

# Mein Bild
pImage = "assets/car.jpeg"

# Object detection
res = aisax_object_detection.call_lsa(pImage, "car")

# Interaktion mit LLMs
## OpenAI
result = aisax_openai.generate_answer("Warum ist der Himmel blau?")
print(result)
print("-----")
result = aisax_openai.generate_multimodal_answer("Beschreibe das Bild", image_path=pImage)
print(result)
print("-----")
result = aisax_ollama.generate_answer("Warum ist der Himmel blau?")
print(result)
print("-----")
## LLAMA3.1 via OLLAMA
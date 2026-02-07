import os
import sys
from dotenv import load_dotenv

# Load env vars BEFORE importing src modules to ensure pydantic settings pick them up
load_dotenv()

# Ensure the project root is in python path
sys.path.append(os.getcwd())

from src.models.lm_studio import get_orchestrator_client, get_phi4_client
from src.models.gemini import get_gemini_client
from src.config import get_settings


def test_local_models():
    settings = get_settings()
    print(f"DEBUG: ORCHESTRATOR_MODEL={settings.models.orchestrator_model}")
    print(f"DEBUG: PHI4_MODEL={settings.models.phi4_model}")
    print(
        f"DEBUG: GEMINI_API_KEY={'*' * 8 if settings.models.gemini_api_key else 'NOT_SET'}"
    )

    print("\n--- Testing Local Models (LM Studio) ---")
    try:
        # Test Orchestrator
        orchestrator = get_orchestrator_client()
        print(f"Connecting to Orchestrator: {orchestrator.model_name}...")
        response = orchestrator.chat(
            [{"role": "user", "content": "Hello, are you online?"}]
        )
        print(f"✅ Orchestrator Response: {response.content[:50]}...")
        print(f"   Latency: {response.latency_ms:.2f}ms")

        # Test Phi-4
        phi4 = get_phi4_client()
        print(f"Connecting to Phi-4: {phi4.model_name}...")
        response = phi4.chat([{"role": "user", "content": "Hello, are you online?"}])
        print(f"✅ Phi-4 Response: {response.content[:50]}...")
        print(f"   Latency: {response.latency_ms:.2f}ms")

    except Exception as e:
        print(f"❌ Local Model Error: {e}")
        print("Ensure LM Studio is running on localhost:1234 and models are loaded.")


def test_gemini():
    print("\n--- Testing Gemini API ---")
    try:
        gemini = get_gemini_client()
        print(f"Connecting to Gemini: {gemini.model_name}...")
        response = gemini.chat(
            [{"role": "user", "content": "Hello, confirm you are online."}]
        )
        print(f"✅ Gemini Response: {response.content[:50]}...")
        print(f"   Cost: ${response.cost:.6f}")
        print(f"   Latency: {response.latency_ms:.2f}ms")

    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        print("Check GEMINI_API_KEY in .env file.")


if __name__ == "__main__":
    test_local_models()
    test_gemini()

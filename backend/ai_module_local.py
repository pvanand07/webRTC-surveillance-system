import json
import base64
import requests
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


# Ollama native API endpoint
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"
MODEL_NAME = "ministral-3:3b"


class Wildlife(BaseModel):
    """Wildlife information model."""
    id: Optional[str] = Field(default=None, description="Unique identifier")
    detected_class: str = Field(description="YOLO detection class name (e.g., buffalo, elephant)")
    commonName: Optional[str] = None
    scientificName: Optional[str] = None
    description: Optional[str] = None
    habitat: Optional[str] = None
    behavior: Optional[str] = None
    safetyInfo: Optional[str] = None
    conservationStatus: Optional[str] = Field(default=None, description="LC, NT, VU, EN, or CR")
    isDangerous: bool = Field(default=False)


def test_ollama_connection() -> Dict[str, Any]:
    """
    Test if Ollama is running and the model is available.

    Returns:
        Dictionary with connection status
    """
    try:
        # Try a simple text-only request first
        test_payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ],
            "stream": False
        }
        test_response = requests.post(OLLAMA_CHAT_ENDPOINT, json=test_payload, timeout=10)
        test_response.raise_for_status()
        return {
            "success": True,
            "message": "Ollama connection successful",
            "endpoint": OLLAMA_CHAT_ENDPOINT
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "endpoint": OLLAMA_CHAT_ENDPOINT,
            "suggestion": "Make sure Ollama is running and the model is installed: ollama pull qwen3-vl:2b"
        }


def identify_wildlife(detected_class: str, base64_image: Optional[str] = None, mime_type: str = "image/jpeg", recent_context: Optional[list] = None) -> Dict[str, Any]:
    """
    Identify wildlife from YOLO detection class name and return detailed information.
    Compatible with ai_module.py interface.

    Args:
        detected_class: YOLO detection class name (e.g., "buffalo", "elephant", "bear")
        base64_image: Optional base64-encoded image string
        mime_type: MIME type of the image (default: "image/jpeg")
        recent_context: Optional list of recently identified Wildlife objects for context

    Returns:
        Dictionary containing wildlife information matching Wildlife model structure
    """
    # Build context string from recent identifications with full information
    context_str = ""
    if recent_context and len(recent_context) > 0:
        context_str = "\n\nRecently detected animals in the area (use this context to help identify similar or related species):\n"
        for idx, wildlife in enumerate(recent_context[-2:], 1):  # Last 2 only
            context_str += f"\n--- Animal {idx} ---\n"
            if wildlife.commonName:
                context_str += f"Common Name: {wildlife.commonName}\n"
            if wildlife.scientificName:
                context_str += f"Scientific Name: {wildlife.scientificName}\n"
            if wildlife.description:
                context_str += f"Description: {wildlife.description}\n"
            if wildlife.habitat:
                context_str += f"Habitat: {wildlife.habitat}\n"
            if wildlife.behavior:
                context_str += f"Behavior: {wildlife.behavior}\n"
            if wildlife.conservationStatus:
                context_str += f"Conservation Status: {wildlife.conservationStatus}\n"
            if wildlife.isDangerous:
                context_str += f"Dangerous: Yes\n"
            if wildlife.safetyInfo:
                context_str += f"Safety Info: {wildlife.safetyInfo}\n"

    # Build user message content (animal-only; no non-animal handling)
    base_prompt = "Provide detailed information about the detected animal including: common name, scientific name, description, habitat, behavior, safety information, conservation status (LC, NT, VU, EN, or CR), and whether it is dangerous to humans."

    if context_str:
        base_prompt += context_str

    # Add detected class context
    if detected_class:
        base_prompt = f"The YOLO detection system identified this object as: {detected_class}. {base_prompt}"

    # Add JSON format requirement (no is_animal)
    prompt = f"""{base_prompt}

You must respond with ONLY a valid JSON object, no other text. The JSON must have this exact structure:
{{
    "commonName": "string or null",
    "scientificName": "string or null",
    "description": "string or null",
    "habitat": "string or null",
    "behavior": "string or null",
    "safetyInfo": "string or null",
    "conservationStatus": "LC, NT, VU, EN, CR, or null",
    "isDangerous": true or false
}}

Return ONLY the JSON object, no markdown, no code blocks, no explanations."""

    # For Ollama native API, images are passed as base64 strings in the images array
    # Prepare messages for Ollama native API format
    messages = [
        {
            "role": "system",
            "content": "You are a wildlife expert that provides accurate, detailed information about animals. Output only valid JSON."
        }
    ]

    # Add user message with image if provided
    user_message = {
        "role": "user",
        "content": prompt
    }

    if base64_image:
        user_message["images"] = [base64_image]  # Ollama native API uses images array with base64 strings

    messages.append(user_message)

    try:
        # Call Ollama native API with format: json
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "format": "json"  # Request JSON format response
        }

        api_response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=120)
        api_response.raise_for_status()

        # Extract response content
        response_data = api_response.json()
        content = response_data.get("message", {}).get("content", "")

        # Parse JSON from response
        # With format: "json", Ollama should return valid JSON, but we'll handle edge cases
        json_data = None
        try:
            # Clean the content
            content_clean = content.strip()

            # Remove markdown code blocks if present (shouldn't happen with format: json, but just in case)
            if content_clean.startswith("```json"):
                content_clean = content_clean[7:]
            elif content_clean.startswith("```"):
                content_clean = content_clean[3:]
            if content_clean.endswith("```"):
                content_clean = content_clean[:-3]
            content_clean = content_clean.strip()

            # Try parsing directly (format: json should return valid JSON)
            json_data = json.loads(content_clean)
        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, try to extract JSON object from the response
            try:
                # Find JSON object in the response
                start_idx = content_clean.find("{")
                end_idx = content_clean.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content_clean[start_idx:end_idx]
                    json_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found in response") from e
            except (json.JSONDecodeError, ValueError) as e2:
                # If JSON parsing fails, raise exception with debugging info
                error_msg = f"Failed to parse JSON response: {str(e2)}"
                error_msg += f"\nRaw response: {content}"
                error_msg += f"\nCleaned response: {content_clean}"
                error_msg += f"\nOriginal error: {str(e)}"
                raise ValueError(error_msg) from e2

        # Validate and structure the response (no is_animal)
        wildlife_data = {
            "commonName": json_data.get("commonName"),
            "scientificName": json_data.get("scientificName"),
            "description": json_data.get("description"),
            "habitat": json_data.get("habitat"),
            "behavior": json_data.get("behavior"),
            "safetyInfo": json_data.get("safetyInfo"),
            "conservationStatus": json_data.get("conservationStatus"),
            "isDangerous": json_data.get("isDangerous", False)
        }

        # Add detected_class back to the data (it's an input parameter, not from VLM)
        wildlife_data["detected_class"] = detected_class

        return wildlife_data

    except Exception as e:
        # Re-raise as a more specific error for compatibility
        raise RuntimeError(f"Failed to identify wildlife: {str(e)}") from e


def get_wildlife_info(detected_class: str, base64_image: Optional[str] = None, mime_type: str = "image/jpeg", recent_context: Optional[list] = None) -> Wildlife:
    """
    Get wildlife information and return as Wildlife model instance.
    Compatible with ai_module.py interface.

    Args:
        detected_class: YOLO detection class name
        base64_image: Optional base64-encoded image string
        mime_type: MIME type of the image (default: "image/jpeg")
        recent_context: Optional list of recently identified Wildlife objects for context

    Returns:
        Wildlife model instance with all information populated
    """
    data = identify_wildlife(detected_class, base64_image, mime_type, recent_context)
    return Wildlife(**data)


if __name__ == "__main__":

    print("=" * 70)
    print("Wildlife Identification System - Test (Local Ollama)")
    print("=" * 70)

    # You may change image_source to a local file path or URL for testing
    image_source = "http://cdn.britannica.com/16/234216-050-C66F8665/beagle-hound-dog.jpg"
    # image_source = "local_test_image.jpg"  # Uncomment and set your image file path

    import os

    def load_image_as_base64(source: str):
        """
        Loads an image from a URL or local file and returns the base64-encoded string and mime type.
        """
        if source.startswith("http://") or source.startswith("https://"):
            print(f"Downloading image from URL: {source}")
            response = requests.get(source, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
            mime_type = response.headers.get("Content-Type", "image/jpeg")
        else:
            print(f"Reading image from local file: {source}")
            with open(source, "rb") as f:
                image_bytes = f.read()
            ext = os.path.splitext(source)[1].lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".bmp": "image/bmp",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }.get(ext, "image/jpeg")
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return base64_image, mime_type

    print(f"Image Source: {image_source}")
    print(f"Model: {MODEL_NAME}")
    print(f"Ollama Endpoint: {OLLAMA_CHAT_ENDPOINT}\n")

    print("Testing Ollama connection...")
    connection_test = test_ollama_connection()
    if not connection_test["success"]:
        print(f"✗ Connection test failed: {connection_test.get('error', 'Unknown error')}")
        if 'suggestion' in connection_test:
            print(f"  Suggestion: {connection_test['suggestion']}")
        print(f"  Endpoint: {connection_test['endpoint']}")
        exit(1)
    print(f"✓ {connection_test['message']}\n")

    try:
        # Load and encode image to base64 from either URL or local file
        base64_image, mime_type = load_image_as_base64(image_source)
        print(f"✓ Image encoded ({len(base64_image)} characters)\n")

        # Get wildlife information
        print("Querying LLM for wildlife information...")
        wildlife = get_wildlife_info(
            detected_class="dog",
            base64_image=base64_image,
            mime_type=mime_type
        )
        print("✓ Information retrieved\n")

        # Display results
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Detected Class: {wildlife.detected_class}")
        print(f"Common Name: {wildlife.commonName}")
        print(f"Scientific Name: {wildlife.scientificName}")
        print(f"\nDescription:\n  {wildlife.description}")
        print(f"\nHabitat:\n  {wildlife.habitat}")
        print(f"\nBehavior:\n  {wildlife.behavior}")
        print(f"\nSafety Information:\n  {wildlife.safetyInfo}")
        print(f"\nConservation Status: {wildlife.conservationStatus}")
        print(f"Is Dangerous: {wildlife.isDangerous}")
        print("=" * 70)

        # JSON output
        print("\nJSON Output:")
        print(json.dumps(wildlife.model_dump(), indent=2))

    except requests.RequestException as e:
        print(f"✗ Network error: {e}")
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

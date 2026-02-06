import json
from typing import Optional, Dict, Any
from openai import OpenAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv(".env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize client with OpenRouter's base URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


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


def identify_wildlife(detected_class: str, base64_image: Optional[str] = None, mime_type: str = "image/jpeg", recent_context: Optional[list] = None) -> Dict[str, Any]:
    """
    Identify wildlife from YOLO detection class name and return detailed information.
    
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
    
    # Build user message content
    base_prompt = "Provide detailed information about the detected animal including: common name, scientific name, description, habitat, behavior, safety information, conservation status (LC, NT, VU, EN, or CR), and whether it is dangerous to humans."
    
    if context_str:
        base_prompt += context_str
    
    user_content = [
        {
            "type": "text",
            "text": base_prompt
        }
    ]
    
    # Add image if provided
    if base64_image:
        data_url = f"data:{mime_type};base64,{base64_image}"
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        })
    
    response = client.chat.completions.create(
        model="google/gemini-3-flash-preview",  # Use a model that supports JSON schema
        messages=[
            {
                "role": "system",
                "content": "You are a wildlife expert that provides accurate, detailed information about animals. Output only valid JSON."
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        temperature=0.1,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "wildlife_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "commonName": {
                            "type": ["string", "null"],
                            "description": "Common name of the animal"
                        },
                        "scientificName": {
                            "type": ["string", "null"],
                            "description": "Scientific name (binomial nomenclature)"
                        },
                        "description": {
                            "type": ["string", "null"],
                            "description": "Detailed description of the animal"
                        },
                        "habitat": {
                            "type": ["string", "null"],
                            "description": "Natural habitat and distribution"
                        },
                        "behavior": {
                            "type": ["string", "null"],
                            "description": "Typical behavior patterns"
                        },
                        "safetyInfo": {
                            "type": ["string", "null"],
                            "description": "Safety information for human encounters"
                        },
                        "conservationStatus": {
                            "type": ["string", "null"],
                            "description": "IUCN conservation status (LC, NT, VU, EN, or CR)"
                        },
                        "isDangerous": {
                            "type": "boolean",
                            "description": "Whether the animal is dangerous to humans"
                        }
                    },
                    "required": [],
                    "additionalProperties": False
                }
            }
        }
    )

    # Extract and parse the JSON string from the response
    content = response.choices[0].message.content
    wildlife_data = json.loads(content)
    
    # Add detected_class back to the data (it's an input parameter, not from VLM)
    wildlife_data["detected_class"] = detected_class

    return wildlife_data


def get_wildlife_info(detected_class: str, base64_image: Optional[str] = None, mime_type: str = "image/jpeg", recent_context: Optional[list] = None) -> Wildlife:
    """
    Get wildlife information and return as Wildlife model instance.
    
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
    import base64
    import requests
    import os

    # Test configuration: set image_source to either a URL or a local file path
    # You may change these for testing
    image_source = "http://cdn.britannica.com/16/234216-050-C66F8665/beagle-hound-dog.jpg"
    # image_source = "local_test_image.jpg"  # Uncomment and set to your local file path for testing
    detected_class = "dog"

    def load_image_as_base64(source: str) -> (str, str):
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
            # Guess mime type from extension (rudimentary)
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

    print("=" * 70)
    print("Wildlife Identification System - Test")
    print("=" * 70)
    print(f"Detected Class: {detected_class}")
    print(f"Image Source: {image_source}\n")

    try:
        # Load and encode image to base64 from either URL or local file
        base64_image, mime_type = load_image_as_base64(image_source)
        print(f"✓ Image encoded ({len(base64_image)} characters)\n")

        # Get wildlife information
        print("Querying LLM for wildlife information...")
        wildlife = get_wildlife_info(
            detected_class=detected_class,
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

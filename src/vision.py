import base64
from config import VISION_MODEL


def analyze_image(image_bytes: bytes, question: str, groq_client, context: str = "") -> str:
    """Analyze an image using Groq Vision model."""
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    system_msg = "You are a helpful assistant analyzing images for users."
    if context:
        system_msg += f"\n\nAdditional context from uploaded documents:\n{context}"

    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    },
                    {"type": "text", "text": question or "Describe this image in detail."}
                ]
            }
        ],
        max_tokens=1024
    )
    return response.choices[0].message.content
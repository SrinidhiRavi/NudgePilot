import json
import os
import time
import urllib.request
import urllib.error

SYSTEM = """You help small-business owners follow up with customers without being pushy.
The conversation is from the business's point of view: "You" is the business, "Customer" is the customer.
If an image is provided, first read the chat in it, then analyze it.

Reply with ONLY a JSON object in exactly this shape:
{
  "engagement_score": <integer 0-100>,
  "engagement_label": "Cold" | "Warm" | "Hot" | "Ready",
  "detected_need": "<what the customer needs, one sentence>",
  "confidence": <integer 0-100>,
  "confidence_label": "Low" | "Medium" | "High",
  "best_move": "<one or two sentences: what to do and when>",
  "signals": [{"text": "<customer's exact words or a short observation>", "type": "positive" | "concern" | "neutral"}],
  "nudges": {"warm": "<message>", "direct": "<message>", "gentle": "<message>"}
}

Rules:
- 2 to 5 signals. Quote the customer's own words where possible.
- Each nudge is a WhatsApp message under 300 characters, uses the customer's name if known, and sounds human.
- Respect any timing the customer asked for (for example "get back tomorrow").
- Never invent prices, dates, or facts that are not in the conversation.
- No pressure, no guilt, no fake urgency.
"""


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
        text = (body.get("text") or "").strip()
        image = body.get("image")  # data URL, e.g. data:image/jpeg;base64,...
        if not text and not image:
            return reply(400, {"error": "Send either text or image."})

        parts = [{"text": text or "Analyze the conversation in the screenshot."}]
        if image:
            header, b64 = image.split(",", 1)
            mime = header.split(":")[1].split(";")[0]
            parts.append({"inlineData": {"mimeType": mime, "data": b64}})

        model = os.environ.get("MODEL", "gemini-2.5-flash")
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": os.environ["GEMINI_API_KEY"],
            },
        )
        # Retry a few times: Gemini sometimes returns 503 (busy) or 429 (too fast) briefly
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=8) as resp:
                    result = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < 2:
                    time.sleep(2)
                    continue
                raise
        out = result["candidates"][0]["content"]["parts"][0]["text"]
        return reply(200, json.loads(out))
    except urllib.error.HTTPError as e:
        return reply(e.code, {"error": e.read().decode("utf-8", "ignore")})
    except Exception as e:
        return reply(500, {"error": str(e)})


# Note: CORS headers are set on the Lambda Function URL itself, so we do not add them here.
def reply(status, data):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(data),
    }

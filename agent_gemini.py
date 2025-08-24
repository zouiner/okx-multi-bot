# agent_gemini.py
import json
import google.generativeai as genai

class GeminiAgent:
    def __init__(self, api_key: str, model="gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def get_signal(self, symbol: str, price: float):
        prompt = f"""
        You are a crypto spot trading AI. Be concise.
        Analyze {symbol} at price {price}.
        Return ONLY compact JSON with fields:
        trend, recommendation, entry_price, stop_loss, take_profits (array), confidence (0..1),
        crisis (bool), crisis_reason.
        """
        resp = self.model.generate_content(prompt)
        txt = resp.text.strip()
        try:
            return json.loads(txt)
        except Exception:
            # lax fallback: try to extract JSON substring
            l = txt.find("{"); r = txt.rfind("}")
            if l != -1 and r != -1 and r > l:
                return json.loads(txt[l:r+1])
            # ultimate fallback
            return {
                "trend": "neutral",
                "recommendation": "hold",
                "entry_price": None,
                "stop_loss": None,
                "take_profits": [],
                "confidence": 0.0,
                "crisis": False,
                "crisis_reason": "parse_error"
            }

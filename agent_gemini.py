import google.generativeai as genai

class GeminiAgent:
    def __init__(self, api_key: str, model="gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def get_signal(self, symbol: str, price: float):
        prompt = f"""
        You are a crypto spot trading AI.
        Analyze {symbol} at price {price}.
        Return JSON with fields:
        trend, recommendation, entry_price, stop_loss, take_profits, confidence,
        crisis (bool), crisis_reason (str).
        """
        resp = self.model.generate_content(prompt)
        return resp.text

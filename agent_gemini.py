import json
import google.generativeai as genai

class GeminiAgent:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def _safe_json(self, text: str) -> dict:
        text = (text or "").strip()
        try:
            return json.loads(text)
        except Exception:
            l = text.find("{"); r = text.rfind("}")
            if l != -1 and r != -1 and r > l:
                return json.loads(text[l:r+1])
            return {}

    def get_signal(self, symbol: str, price: float) -> dict:
        prompt = f"""
        You are a crypto trading AI.
        Analyze {symbol} at price {price}.
        Return JSON only:
        {{
          "trend": "...",
          "recommendation": "long|short|hold",
          "entry_price": ...,
          "stop_loss": ...,
          "take_profits": [...],
          "confidence": 0.0,
          "crisis": false,
          "crisis_reason": "..."
        }}
        """
        resp = self.model.generate_content(prompt)
        return self._safe_json(getattr(resp, "text", "") or "")

    def suggest_params(self, symbol: str, snapshot: dict) -> dict:
        prompt = f"""
        Given snapshot for {symbol}: {snapshot},
        suggest tuned params in JSON:
        {{
          "grid": {{ "grids": 12, "per_order_quote": 40, "atr_mult": 2.0, "ema_fast_n": 20,
                     "ema_slow_n": 60, "recenter_threshold": 0.35, "recenter_every": 240,
                     "dca_tp_pct": 0.10, "dca_add_drawdown_pct": 0.15, "dca_max_adds": 2,
                     "dca_equity_fraction": 0.10 }},
          "perps": {{ "risk_per_trade_pct": 0.01, "max_leverage": 3,
                      "sl_atr_mult": 1.5, "tp_r_multiple": 1.5,
                      "enter_threshold": 0.25, "exit_threshold": 0.05, "pyramids": 1 }}
        }}
        """
        resp = self.model.generate_content(prompt)
        return self._safe_json(getattr(resp, "text", "") or "")

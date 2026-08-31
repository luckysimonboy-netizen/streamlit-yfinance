import os,json,requests
def secret(n,d=""):
    try:
        import streamlit as st
        v=st.secrets.get(n)
        if v is not None:return str(v)
    except Exception:pass
    return os.getenv(n,d)
def available():return {"Gemini":bool(secret("GEMINI_API_KEY")),"OpenRouter":bool(secret("OPENROUTER_API_KEY")),"Groq":bool(secret("GROQ_API_KEY"))}
def call(prompt):
    p=available()
    if p["Gemini"]:
        try:
            from google import genai
            r=genai.Client(api_key=secret("GEMINI_API_KEY")).models.generate_content(model=secret("GEMINI_MODEL","gemini-2.5-flash"),contents=prompt)
            return getattr(r,"text",str(r)),"Gemini"
        except Exception:pass
    if p["OpenRouter"]:
        try:
            r=requests.post("https://openrouter.ai/api/v1/chat/completions",headers={"Authorization":"Bearer "+secret("OPENROUTER_API_KEY"),"Content-Type":"application/json"},json={"model":secret("OPENROUTER_MODEL","openrouter/free"),"messages":[{"role":"system","content":"You are Simon Stock, rigorous equity research. Never invent missing data. Do not reveal chain-of-thought."},{"role":"user","content":prompt}]},timeout=60)
            r.raise_for_status();return r.json()["choices"][0]["message"]["content"],"OpenRouter"
        except Exception:pass
    if p["Groq"]:
        try:
            r=requests.post("https://api.groq.com/openai/v1/chat/completions",headers={"Authorization":"Bearer "+secret("GROQ_API_KEY"),"Content-Type":"application/json"},json={"model":secret("GROQ_MODEL","openai/gpt-oss-120b"),"messages":[{"role":"system","content":"You are Simon Stock, rigorous equity research. Never invent missing data. Do not reveal chain-of-thought."},{"role":"user","content":prompt}]},timeout=60)
            r.raise_for_status();return r.json()["choices"][0]["message"]["content"],"Groq"
        except Exception:pass
    return "No AI provider configured. Add an API key in Streamlit Secrets.","None"
class AIOrchestrator:
    def run(self,pack):
        base=json.dumps(pack,default=str,ensure_ascii=False)
        tasks={"Value Analyst":"Evaluate moat, long-term economics, margin of safety and capital allocation.","Business Analyst":"Analyze business model, pricing power, switching costs, customer economics and management quality.","First Principles Analyst":"Decompose the business into technical and economic drivers and identify growth ceilings.","Event Analyst":"Assess news, macro and policy catalysts and whether expectations may already be priced in.","Quant Analyst":"Interpret trend, momentum, volatility, drawdown and conflicting signals.","Risk Officer":"Try to disprove the investment case and identify valuation, execution, macro and concentration risks."}
        specialists={};provider="None"
        for name,task in tasks.items():specialists[name],provider=call(f"Simon Stock specialist assignment. Task: {task}\nUse only supplied evidence.\nPACK:\n{base}")
        debate,provider=call(f"Act as an adversarial Bull/Bear committee. Build strongest bull and bear cases, counter-evidence and thesis-breaking conditions. Do not invent numbers.\nPACK:\n{base}\nREPORTS:\n{json.dumps(specialists,ensure_ascii=False)}")
        judge,provider=call(f"""You are Simon Stock Chief Analyst / AI Judge. Adjudicate evidence rather than averaging opinions. Return FINAL VERDICT, SCORE, CONFIDENCE, TOP REASONS, TOP RISKS, WHAT WOULD CHANGE MY MIND, DATA QUALITY. No guaranteed returns, no invented targets, no chain-of-thought.
PACK: {base}
REPORTS: {json.dumps(specialists,ensure_ascii=False)}
DEBATE: {debate}""")
        verdict=next((v for v in ["STRONG BUY","BUY ON DIPS","BUY","REDUCE","AVOID","WAIT","HOLD"] if v in judge.upper()),"HOLD")
        return {"verdict":verdict,"score":round(pack["score"]),"confidence":75,"judge":judge,"debate":debate,"specialists":specialists,"provider":provider}

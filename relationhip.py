import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import psycopg2
from psycopg2 import Error
from datetime import date
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 환경 설정 ---
os.environ['OLLAMA_MODELS'] = 'D:/ollama_models'
os.environ['HF_HOME'] = 'D:/huggingface_models'

# --- 데이터베이스 설정 ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'km923009!!',
    'dbname': 'gguro',
    'port': 5432
}

# --- LLM 준비 ---
llm = ChatOllama(model="timhan/llama3korean8b4qkm")

app = FastAPI(
    title="관계 조언 생성 API",
    description="아이의 대화 내용을 바탕으로 보호자에게 관계 개선을 위한 조언을 생성합니다."
)

class AdviceRequest(BaseModel):
    profile_id: int

def fetch_today_conversations(profile_id: int) -> List[Tuple[str, str]]:
    """
    오늘 날짜에 해당하는 특정 프로필의 대화 내용과 역할을 가져옵니다.
    """
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        today = date.today()
        
        # [수정] role 컬럼을 함께 조회하고, 오늘 날짜의 대화만 필터링합니다.
        cursor.execute("""
            SELECT role, content 
            FROM talk 
            WHERE profile_id = %s AND DATE(created_at) = %s
            AND category = 'LIFESTYLEHABIT' -- 역할놀이나 퀴즈가 아닌 일상대화만 필터링
            ORDER BY created_at ASC
        """, (profile_id, today))
        
        rows = cursor.fetchall()
        return rows
    except Error as e:
        print(f"[DB 오류] 대화 내용 조회 실패: {e}")
        return []
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.post("/relationship-advice")
async def generate_relationship_advice(req: AdviceRequest):
    """
    오늘의 대화 내용을 기반으로 심층적인 관계 조언을 생성합니다.
    """
    conversations = fetch_today_conversations(req.profile_id)
    
    if not conversations:
        raise HTTPException(status_code=404, detail="오늘의 대화 내용이 없습니다.")

    # [수정] 아이(user)의 대화만 필터링하여 분석용 텍스트를 생성합니다.
    child_talks = [f"- {content}" for role, content in conversations if role == 'user']
    if not child_talks:
        raise HTTPException(status_code=404, detail="오늘 아이의 대화 내용이 없습니다.")
        
    conversation_log = "\n".join(child_talks)

    # [수정] 전문가 역할을 부여하고, 분석 가이드를 명확히 하는 프롬프트로 변경
    prompt = ChatPromptTemplate.from_template("""
당신은 아동 심리 및 부모-자녀 관계 전문가입니다. 당신의 임무는 주어진 아이의 '오늘의 대화' 내용을 심층적으로 분석하여, 보호자를 위한 전문적이고 따뜻한 조언을 제공하는 것입니다.

다음 세 가지 단계에 따라 분석하고, 각 항목을 명확하게 구분하여 작성해주세요.

1.  **아이의 심리 및 성격 분석**:
    * 아이의 발언을 통해 드러나는 감정, 관심사, 사고방식, 성격적 특성을 구체적으로 분석합니다.
    * 긍정적인 면과 함께, 어려움을 겪고 있을 수 있는 부분도 함께 짚어주세요.

2.  **보호자와의 관계 진단**:
    * 아이의 대화 내용을 바탕으로, 현재 보호자와의 관계가 어떻게 형성되어 있을지 유추합니다.
    * 예를 들어, 아이가 자신의 감정을 솔직하게 표현하는지, 혹은 특정 주제에 대해 방어적인 태도를 보이는지 등을 근거로 관계의 질을 진단합니다.

3.  **구체적인 관계 개선 조언**:
    * 위 분석과 진단을 바탕으로, 보호자가 오늘 바로 실천할 수 있는 구체적이고 실용적인 조언을 2~3가지 제안합니다.
    * 추상적인 조언이 아닌, 실제 대화에서 활용할 수 있는 말이나 행동을 예시로 들어주세요.

---
**[분석할 아이의 오늘 대화 내용]**
{conversation_log}
---
""")
    
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({"conversation_log": conversation_log})
    
    return {"profile_id": req.profile_id, "advice": result}

if __name__ == "__main__":
    import uvicorn
    print("🚀 FastAPI 서버를 시작합니다. http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

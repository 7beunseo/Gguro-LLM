import os
import sys
import random
import re
import mysql.connector
from mysql.connector import Error
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Ollama 및 허깅페이스 모델의 로컬 경로를 지정합니다.
os.environ['OLLAMA_MODELS'] = 'D:/ollama_models'
os.environ['HF_HOME'] = 'D:/huggingface_models'

# LangChain 관련 라이브러리들을 가져옵니다.
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import InMemoryChatMessageHistory

# --- MySQL 데이터베이스 설정 (사용자 환경에 맞게 수정) ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'km923009!!',
    'database': 'gguro'
}

# 챗봇의 모든 핵심 로직을 담고 있는 클래스입니다.
class ChatbotLogic:
    def __init__(self, model_name='timhan/llama3korean8b4qkm'):
        print("키즈케어 로봇 '꾸로' 로딩 시작...")
        self.model = ChatOllama(model=model_name)
        self.store = {}
        self.quiz_mode = {}

        self.instruct = self._get_instruct('instruct')
        self.quiz_data = self._load_quiz_data('rag_data/quiz_data.txt')
        self.rag_chain = self._setup_rag_and_history()
        self.quiz_eval_chain = self._create_quiz_eval_chain()

        self._ensure_table_exists()

        if self.rag_chain and self.quiz_eval_chain:
            print("챗봇 로직이 정상적으로 로드되었습니다.")
        else:
            print("[중요] 챗봇 로직 초기화에 실패했습니다.")

    def _get_instruct(self, path):
        file_list = ['base', 'few_shot']
        instruction_template = ''
        try:
            for file_name in file_list:
                with open(f'{path}/{file_name}.txt', 'r', encoding='utf-8-sig') as f:
                    instruction_template += f.read() + "\n"
            return instruction_template
        except FileNotFoundError: return "너는 친절한 친구야."

    def _load_quiz_data(self, file_path):
        quizzes = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
            quiz_blocks = [block for block in content.strip().split('#---') if block.strip()]
            for block in quiz_blocks:
                lines = block.strip().split('\n'); quiz_item = {}
                for line in lines:
                    if line.startswith('질문:'): quiz_item['question'] = line.replace('질문:', '').strip()
                    elif line.startswith('정답:'): quiz_item['answer'] = line.replace('정답:', '').strip()
                    elif line.startswith('힌트:'): quiz_item['hint'] = line.replace('힌트:', '').strip()
                if 'question' in quiz_item and 'answer' in quiz_item and 'hint' in quiz_item: quizzes.append(quiz_item)
            if quizzes: print(f"대화형 퀴즈 데이터 로드 성공: 총 {len(quizzes)}개"); return quizzes
            else: print(f"[경고] 퀴즈 파일({file_path})에서 유효한 퀴즈를 찾지 못했습니다."); return []
        except Exception as e: print(f"[오류] 퀴즈 데이터 처리 중 문제 발생: {e}"); return []

    def _setup_rag_and_history(self):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"); docs = TextLoader("rag_data/info.txt", encoding='utf-8').load()
            retriever = Chroma.from_documents(RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs), embeddings).as_retriever()
            prompt = ChatPromptTemplate.from_messages([("system", f"{self.instruct}\n\n[참고 정보]\n{{context}}"), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])
            rag_chain_main = (RunnablePassthrough.assign(context=lambda x: retriever.get_relevant_documents(x["input"]))| prompt| self.model| StrOutputParser())
            return RunnableWithMessageHistory(rag_chain_main, self._get_session_history, input_messages_key="input", history_messages_key="chat_history")
        except Exception as e: print(f"[오류] RAG 체인 설정 중 문제 발생: {e}"); return None

    def _create_quiz_eval_chain(self):
        try:
            system_prompt = """당신은 아이의 답변을 채점하는, 감정이 배제된 극도로 정교하고 논리적인 AI 시스템입니다.
당신의 임무는 아이의 '답변'을 다음의 규칙에 따라 분석하고, 정해진 형식에 맞춰 결과를 출력하는 것입니다.
**규칙 1 (최우선 규칙):** 먼저 아이의 '답변'을 확인한다. 만약 답변이 '응', '네', '아니', '아니요', '몰라', '글쎄' 와 같이 구체적인 내용이 없는 한두 단어의 표현일 경우, **다른 모든 분석을 중단하고 즉시 오답으로 판단**해야 한다.
**규칙 2 (심층 분석 규칙):** '답변'에 구체적인 내용이 포함된 경우에만, 아래 3단계 '생각의 사슬' 과정을 따라서 분석 결과를 출력한다.
1. **[분석 기준 설정]**: 이 질문의 안전 핵심 개념은 무엇인가?
2. **[답변 분석 및 결론]**: 설정된 기준에 따라 답변을 평가한다. **이 부분은 반드시 아이에게 말하듯 친근한 반말체로 서술해야 한다.**
3. **[최종 판단]**: 위 결론에 따라, 최종 판단을 `[판단: 참]` 또는 `[판단: 거짓]` 형식으로 출력한다.
아래 예시를 완벽하게 이해하고, 주어진 형식과 규칙을 기계적으로 따라야 합니다."""
            few_shot_examples = [
                { "input": "핵심 개념: 안된다고 말하고, 엄마 아빠와 정한 비밀 암호를 물어봐야 한다.\n답변: 엄마한테 가야 한다고 말할 거예요.", "output": """[분석 기준 설정]이 질문의 핵심은 낯선 사람을 따라가지 않는 '거절'과 '안전 확보' 행동이야. 아이의 답변이 이 두 가지 중 하나라도 충족시키는지 확인해야 해.\n[답변 분석 및 결론]음, "엄마한테 가야 한다"고 말하는 건, 낯선 사람을 안 따라가고 제일 안전한 엄마한테 가려는 거니까, 아주 똑똑한 행동이야. 위험한 상황을 잘 피했어.\n[최종 판단][판단: 참]"""},
                { "input": "핵심 개념: 알록달록 예쁜 약을 사탕인 줄 알고 먹어도 될까?\n답변: 아니", "output": """[분석 기준 설정]이 질문은 '하면 안 되는 행동'에 대한 판단력을 묻고 있어.\n[답변 분석 및 결론]아이의 답변 '아니'는 올바른 판단을 내린 것이지만, 너무 짧아서 왜 안 되는지 이해했는지는 알 수 없어. 더 구체적인 설명이 필요해.\n[최종 판단][판단: 거짓]"""},
                { "input": "핵심 개념: 젖은 수건으로 입과 코를 막고, 몸을 낮춰서 기어서 대피해야 한다.\n답변: 119에 전화할 거예요.", "output": """[분석 기준 설정]이 질문의 핵심은 '어떻게 대피하는가'라는 '대피 방법'에 대한 거야. 답변이 연기를 피하고 안전하게 이동하는 행동을 묘사하는지 확인해야 해.\n[답변 분석 및 결론]아이의 답변 '119 신고'는 화재 시 매우 중요하고 올바른 행동이지만, 질문이 요구하는 '대피 방법' 그 자체는 아니야. 질문의 핵심을 벗어난 답변이야.\n[최종 판단][판단: 거짓]"""}
            ]
            prompt = ChatPromptTemplate.from_messages([("system", system_prompt), *sum([[("human", ex["input"]), ("ai", ex["output"])] for ex in few_shot_examples], []), ("human", "핵심 개념: {answer}\n답변: {user_input}")])
            return prompt | self.model | StrOutputParser()
        except Exception as e: print(f"[오류] 퀴즈 채점 체인 생성 중 문제 발생: {e}"); return None

    def _get_session_history(self, session_id: str):
        if session_id not in self.store: self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    # --- [DB 추가] 데이터베이스 관련 함수들 ---
    def _create_db_connection(self):
        try: return mysql.connector.connect(**DB_CONFIG)
        except Error as e: print(f"[DB 오류] 데이터베이스 연결 실패: {e}"); return None

    def _ensure_table_exists(self):
        conn = self._create_db_connection()
        if conn is None: return
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS talk (
                    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    category ENUM('OBJECTPLAY', 'LIFESTYLEHABIT', 'SAFETYSTUDY', 'ANIMALKNOWLEDGE', 'ROLEPLAY') NOT NULL,
                    content TEXT NOT NULL,
                    session_id VARCHAR(255),
                    role VARCHAR(255),
                    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
                    profile_id BIGINT NOT NULL
                );
            """)
            print("[DB 정보] 'talk' 테이블이 준비되었습니다.")
        except Error as e: print(f"[DB 오류] 테이블 생성 실패: {e}")
        finally:
            if conn.is_connected(): cursor.close(); conn.close()
    
    # [핵심] 백그라운드에서 실행될 대화 저장 함수
    def save_conversation_to_db(self, session_id: str, user_input: str, bot_response: str, category: str, profile_id: int = 1):
        """사용자 입력과 봇 응답을 순차적으로 DB에 저장합니다."""
        print(f"📝 백그라운드 저장 시작: 세션 [{session_id}]")
        conn = self._create_db_connection()
        if conn is None: return
        try:
            cursor = conn.cursor()
            query = "INSERT INTO talk (session_id, role, content, category, profile_id) VALUES (%s, %s, %s, %s, %s)"
            # 사용자 메시지 저장
            cursor.execute(query, (session_id, 'user', user_input, category, profile_id))
            # 봇 메시지 저장
            cursor.execute(query, (session_id, 'bot', bot_response, category, profile_id))
            conn.commit()
            print(f"✅ 백그라운드 저장 완료: 세션 [{session_id}]")
        except Error as e:
            print(f"[DB 오류] 메시지 저장 실패: {e}")
        finally:
            if conn.is_connected(): cursor.close(); conn.close()

    # [핵심] 비동기 방식으로 변경
    async def handle_quiz(self, user_input, session_id):
        quiz_state = self.quiz_mode.get(session_id)
        MAX_ATTEMPTS = 2

        if quiz_state:
            current_quiz = quiz_state['quiz_item']
            eval_result_text = await self.quiz_eval_chain.ainvoke({"answer": current_quiz['answer'], "user_input": user_input})
            print(f"\n--- LLM 채점 시작 ---\n{eval_result_text}\n--- LLM 채점 종료 ---\n")
            is_correct = "[판단: 참]" in eval_result_text
            try:
                thought_process = re.search(r"\[답변 분석 및 결론\](.*?)(?=\[최종 판단\])", eval_result_text, re.DOTALL).group(1).strip()
            except AttributeError:
                thought_process = None

            if is_correct:
                del self.quiz_mode[session_id]
                response = "딩동댕! 정답이야!\n"
                if thought_process: response += f"\n[꾸로의 생각]\n{thought_process}\n"
                response += f"\n그래서 모범 답안은 바로 이거야!\n**{current_quiz['answer']}**\n\n정말 똑똑한걸? 또 퀴즈 풀고 싶으면 '퀴즈'라고 말해줘!"
                return response
            else:
                quiz_state['attempts'] += 1
                if quiz_state['attempts'] < MAX_ATTEMPTS:
                    if thought_process: hint_message = f"음... 네 생각도 일리가 있어! 하지만 꾸로가 조금 더 깊이 생각해봤는데,\n\n[꾸로의 생각]\n{thought_process}\n\n그래서 완벽한 정답은 아닌 것 같아. 내가 진짜 힌트를 줄게!\n\n"
                    else: hint_message = "음... 그럴듯한 답변이지만, 더 중요한 점이 있는 것 같아! 자, 진짜 힌트를 줄게!\n\n"
                    hint_message += f"힌트: {current_quiz['hint']}\n\n이 힌트를 보고 다시 한번 생각해볼래?"
                    return hint_message
                else:
                    del self.quiz_mode[session_id]
                    return f"아쉽다! 정답은 '{current_quiz['answer']}'이었어. 괜찮아, 이렇게 하나씩 배우는 거지! 다음엔 꼭 맞힐 수 있을 거야."
        else:
            if not self.quiz_data: return "미안, 지금은 퀴즈를 낼 수 없어."
            quiz = random.choice(self.quiz_data)
            self.quiz_mode[session_id] = {'quiz_item': quiz, 'attempts': 0}
            return f"좋아, 재미있는 안전 퀴즈 시간! \n\n{quiz['question']}"
    
    # [핵심] 비동기 방식으로 변경
    async def invoke(self, user_input, session_id):
        is_quiz_session = session_id in self.quiz_mode or any(k in user_input for k in ["퀴즈", "문제", "게임"])
        
        if is_quiz_session:
            response_text = await self.handle_quiz(user_input, session_id)
        else:
            if not self.rag_chain: 
                response_text = "챗봇 로직이 초기화되지 않았습니다."
            else:
                try:
                    response_text = await self.rag_chain.ainvoke({"input": user_input}, config={'configurable': {'session_id': session_id}})
                except Exception as e: 
                    print(f"[오류] 대화 생성 중 문제 발생: {e}")
                    response_text = "미안, 지금은 대답하기가 좀 힘들어."
        
        return response_text

# FastAPI를 이용해 챗봇 API 서버를 설정하고 실행합니다.
app = FastAPI(
    title="키즈케어 챗봇 '꾸로' API (백그라운드 DB 저장)",
    description="응답을 먼저 반환하고, 대화 내용은 백그라운드에서 MySQL에 기록하는 챗봇입니다.",
    version="8.1-BackgroundSave"
)
class ChatRequest(BaseModel): user_input: str; session_id: str
chatbot_logic = ChatbotLogic()

# [핵심] BackgroundTasks를 사용하여 응답 후 DB 저장
@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    is_quiz_session = request.session_id in chatbot_logic.quiz_mode or any(k in request.user_input for k in ["퀴즈", "문제", "게임"])
    category = 'SAFETYSTUDY'
    
    response_text = await chatbot_logic.invoke(request.user_input, request.session_id)
    
    # 응답을 반환한 후에 실행될 작업을 추가
    background_tasks.add_task(
        chatbot_logic.save_conversation_to_db,
        session_id=request.session_id,
        user_input=request.user_input,
        bot_response=response_text,
        category=category
    )
    
    return {"response": response_text}

if __name__ == "__main__":
    print("FastAPI 서버를 시작합니다. 접속 주소: http://127.0.0.1:8000")
    print("API 문서는 http://127.0.0.1:8000/docs 에서 확인하세요.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

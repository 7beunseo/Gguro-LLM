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
from langchain_core.messages import HumanMessage, AIMessage

# --- MySQL 데이터베이스 설정 (사용자 환경에 맞게 수정) ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # 데이터베이스 사용자 이름
    'password': 'km923009!!',# 데이터베이스 비밀번호
    'database': 'gguro'       # 사용할 스키마 이름
}

# --- 챗봇 로직 클래스 (최종 수정 버전) ---
class ChatbotLogic:
    def __init__(self, model_name='timhan/llama3korean8b4qkm'):
        print("키즈케어 로봇 '꾸로' 로딩 시작...")
        self.model = ChatOllama(model=model_name)
        self.store = {}
        self.instruct = self._get_instruct()
        self.rag_chain = None
        self._setup_rag_and_history()

        # [DB 추가] 데이터베이스 테이블 준비
        self._ensure_table_exists()

        if self.rag_chain:
            print("챗봇 로직이 정상적으로 로드되었습니다.")
        else:
            print("[중요] 챗봇 로직 초기화 실패!")

    def _get_instruct(self):
        """인스트럭션 텍스트를 파일에서 불러옵니다."""
        file_list = ['base', 'few_shot']
        path = 'instruct'
        instruction_template = ''
        try:
            for file in file_list:
                with open(f'{path}/{file}.txt', 'r', encoding='utf-8-sig') as f:
                    full_txt = f.read()
                instruction_template = f'{instruction_template}\n{full_txt}'
            print("인스트럭션 로드 성공.")
            return instruction_template
        except FileNotFoundError as e:
            print(f"[오류] 인스트럭션 파일({e.filename})을 찾을 수 없습니다.")
            return "너는 친절한 친구야."

    def _setup_rag_and_history(self):
        """대화 기록(History)과 정보 검색(RAG)을 결합한 체인을 설정합니다."""
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            documents = TextLoader("rag_data/info.txt", encoding='utf-8').load()
            if not documents:
                raise ValueError("RAG 정보 파일(info.txt)의 내용이 비어있습니다.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            retriever = Chroma.from_documents(docs, embeddings).as_retriever()
            print("RAG 벡터 DB 및 검색기(Retriever) 생성 완료.")

            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""{self.instruct}

[참고할 만한 정보]
{{context}}"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            output_parser = StrOutputParser()

            rag_chain_main = (
                RunnablePassthrough.assign(
                    context=lambda x: retriever.get_relevant_documents(x["input"])
                )
                | prompt
                | self.model
                | output_parser
            )

            self.rag_chain = RunnableWithMessageHistory(
                rag_chain_main,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )
            print("LangChain 대화 기억 RAG 체인 설정 완료.")

        except Exception as e:
            print(f"[오류] RAG 또는 체인 설정 중 심각한 문제 발생: {e}")
            self.rag_chain = None

    def _get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    
    # --- [DB 추가] 데이터베이스 관련 함수들 ---
    def _create_db_connection(self):
        try:
            return mysql.connector.connect(**DB_CONFIG)
        except Error as e:
            print(f"[DB 오류] 데이터베이스 연결 실패: {e}"); return None

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
        except Error as e:
            print(f"[DB 오류] 테이블 생성 실패: {e}")
        finally:
            if conn.is_connected(): cursor.close(); conn.close()

    # [핵심] 백그라운드에서 실행될 대화 저장 함수
    def save_conversation_to_db(self, session_id: str, user_input: str, bot_response: str, profile_id: int = 1):
        """사용자 입력과 봇 응답을 순차적으로 DB에 저장합니다."""
        print(f"📝 백그라운드 저장 시작: 세션 [{session_id}]")
        conn = self._create_db_connection()
        if conn is None: return
        try:
            cursor = conn.cursor()
            query = "INSERT INTO talk (session_id, role, content, category, profile_id) VALUES (%s, %s, %s, %s, %s)"
            category = 'LIFESTYLEHABIT' # 일상 대화이므로 카테고리 고정
            
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

    # [핵심] 비동기 방식으로 변경하여 응답 우선 처리
    async def invoke(self, user_input, session_id):
        if not self.rag_chain:
            return "챗봇 로직 초기화에 실패했습니다."
            
        try:
            # 비동기 invoke 메서드 사용
            response = await self.rag_chain.ainvoke(
                {"input": user_input},
                config={'configurable': {'session_id': session_id}}
            )
            return response
        except Exception as e:
            print(f"[오류] 대화 생성 중 문제 발생: {e}")
            return "미안, 지금은 대답하기가 좀 힘들어."


# --- FastAPI 서버 설정 ---
app = FastAPI(title="은서의 친구 '꾸로' API (백그라운드 DB 저장)")

class ChatRequest(BaseModel):
    user_input: str
    session_id: str

chatbot_logic = ChatbotLogic()

# [핵심] BackgroundTasks를 사용하여 응답 후 DB 저장
@app.post("/chat", summary="챗봇과 대화하기")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    사용자 입력을 받아 챗봇의 응답을 즉시 반환하고,
    대화 내용은 백그라운드에서 데이터베이스에 저장합니다.
    """
    response_text = await chatbot_logic.invoke(request.user_input, request.session_id)
    
    # 응답을 반환한 후에 실행될 작업을 추가
    background_tasks.add_task(
        chatbot_logic.save_conversation_to_db,
        session_id=request.session_id,
        user_input=request.user_input,
        bot_response=response_text
    )
    
    return {"response": response_text}

# uvicorn으로 이 파일을 실행하기 위한 메인 블록
if __name__ == "__main__":
    print("FastAPI 서버 시작... 브라우저에서 http://127.0.0.1:8000/docs 를 열어보세요.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
os.environ['OLLAMA_MODELS'] = 'D:/ollama_models'
os.environ['HF_HOME'] = 'D:/huggingface_models'

import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

# --- LangChain 및 LLM 관련 모듈 ---
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

# --- 챗봇 로직 클래스 (최종 수정 버전) ---
class ChatbotLogic:
    def __init__(self, model_name='timhan/llama3korean8b4qkm'):
        print("키즈케어 로봇 '꾸로' 로딩 시작...")
        self.model = ChatOllama(model=model_name)
        self.store = {}
        self.instruct = self._get_instruct()
        self.rag_chain = None
        self._setup_rag_and_history()
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

            # 시스템, 대화기록, 사용자 질문의 역할을 명확히 하는 프롬프트
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""{self.instruct}

[참고할 만한 정보]
{{context}}"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            output_parser = StrOutputParser()

            # RAG 체인의 핵심 로직
            rag_chain_main = (
                RunnablePassthrough.assign(
                    context=lambda x: retriever.get_relevant_documents(x["input"])
                )
                | prompt
                | self.model
                | output_parser
            )

            # 대화 기억 기능 래핑
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
    
    def invoke(self, user_input, session_id):
        if not self.rag_chain:
            return "챗봇 로직 초기화에 실패했습니다."
            
        try:
            response = self.rag_chain.invoke(
                {"input": user_input},
                config={'configurable': {'session_id': session_id}}
            )
            return response
        except Exception as e:
            print(f"[오류] 대화 생성 중 문제 발생: {e}")
            return "미안, 지금은 대답하기가 좀 힘들어."


# --- FastAPI 서버 설정 ---
app = FastAPI(title="은서의 친구 '꾸로' API")

# API 요청 본문의 형식을 정의합니다.
class ChatRequest(BaseModel):
    user_input: str
    session_id: str

# 서버가 시작될 때 챗봇 로직을 한번만 로드합니다.
chatbot_logic = ChatbotLogic()

@app.post("/chat", summary="챗봇과 대화하기")
def chat(request: ChatRequest):
    """
    사용자 입력과 세션 ID를 받아 챗봇의 응답을 반환합니다.
    """
    response_text = chatbot_logic.invoke(request.user_input, request.session_id)
    return {"response": response_text}

# uvicorn으로 이 파일을 실행하기 위한 메인 블록
if __name__ == "__main__":
    import uvicorn
    print("FastAPI 서버 시작... 브라우저에서 http://127.0.0.1:8000/docs 를 열어보세요.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
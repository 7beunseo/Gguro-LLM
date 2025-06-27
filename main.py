import os
import sys
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTextEdit, QPushButton
from PyQt5.QtCore import pyqtSignal

os.environ['OLLAMA_MODELS'] = 'D:/ollama_models'
os.environ['HF_HOME'] = 'D:/huggingface_models'

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


# --- 챗봇 로직 클래스 ---
class ChatbotLogic:
    def __init__(self, model_name='timhan/llama3korean8b4qkm'):
        print("키즈케어 로봇 초기화 시작... (NVIDIA GPU 사용)")
        self.model = ChatOllama(model=model_name)
        self.store = {}
        self.instruct = self._get_instruct()
        self.rag_chain = None
        self._setup_rag_and_history()
        if self.rag_chain:
            print("RAG 및 인스트럭션 준비 완료. 챗봇이 정상적으로 초기화되었습니다.")
        else:
            print("[중요] 챗봇 로직 초기화 실패! 터미널의 오류 메시지를 확인하세요.")

    def _get_instruct(self):
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
        """
        LangChain의 표준적인 대화형 RAG 체인 방식으로 재구성 
        MessagesPlaceholder를 사용하여 대화 내역을 관리
        """
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            documents = TextLoader("rag_data/info.txt", encoding='utf-8').load()
            if not documents:
                raise ValueError("RAG 정보 파일(info.txt)의 내용이 비어있습니다.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            retriever = Chroma.from_documents(docs, embeddings).as_retriever()
            print("RAG 벡터 DB 및 검색기(Retriever) 생성 완료.")

            # 프롬프트 구조를 System - History - Human 형태로 명확히 분리
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""{self.instruct}

[참고 정보]
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


# --- PyQt5 GUI 클래스  ---
class ChatBotWindow(QWidget):
    message_received = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.llm_logic = ChatbotLogic()
        self.user_no = 0
        self.session_id = f'eunseo_session_{self.user_no}'
        self._init_ui()
        self._init_chat_session()

    def _init_chat_session(self):
        self.chat_history.append("꾸로: 안녕, 은서! 나는 너의 친구 로봇 꾸로야. 오늘 뭐하고 놀까?")

    def _init_ui(self):
        self.setWindowTitle("은서의 친구 '꾸로'")
        self.setGeometry(100, 100, 450, 500)
        self.layout = QVBoxLayout()
        self.chat_history = QTextEdit(self)
        self.chat_history.setReadOnly(True)
        self.layout.addWidget(self.chat_history)
        self.input_text = QLineEdit(self)
        self.input_text.returnPressed.connect(self._send_message)
        self.layout.addWidget(self.input_text)
        self.button_layout = QHBoxLayout()
        self.send_button = QPushButton('보내기', self)
        self.send_button.clicked.connect(self._send_message)
        self.button_layout.addWidget(self.send_button)
        self.clear_button = QPushButton('대화 새로 시작하기', self)
        self.clear_button.clicked.connect(self._clear_chat)
        self.button_layout.addWidget(self.clear_button)
        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)
        self.message_received.connect(self.append_bot_message)

    def _send_message(self):
        user_input = self.input_text.text()
        if user_input:
            self.chat_history.append(f"은서: {user_input}")
            self.input_text.clear()
            threading.Thread(target=self._thread_llm, args=(user_input,), daemon=True).start()

    def _thread_llm(self, user_input):
        response = self.llm_logic.invoke(user_input, self.session_id)
        self.message_received.emit(response)

    def append_bot_message(self, text):
        self.chat_history.append(f"꾸로: {text}")

    def _clear_chat(self):
        if self.session_id in self.llm_logic.store:
            self.llm_logic.store[self.session_id].clear()
        self.user_no += 1
        self.session_id = f'eunseo_session_{self.user_no}'
        self.chat_history.clear()
        self.chat_history.append("꾸로: 좋아, 은서. 대화를 새로 시작하자! 오늘은 뭐하고 놀까?")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    chatbot_window = ChatBotWindow()
    chatbot_window.show()
    sys.exit(app.exec_())
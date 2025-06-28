import os
import sys
import random
import re
from fastapi import FastAPI
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


# 챗봇의 모든 핵심 로직을 담고 있는 클래스입니다.
class ChatbotLogic:
    def __init__(self, model_name='timhan/llama3korean8b4qkm'):
        """챗봇 시스템의 모든 구성 요소를 초기화합니다."""
        print("키즈케어 로봇 '꾸로' 로딩 시작...")
        self.model = ChatOllama(model=model_name)
        self.store = {}
        self.quiz_mode = {}

        self.instruct = self._get_instruct('instruct')
        self.quiz_data = self._load_quiz_data('rag_data/quiz_data.txt')
        self.rag_chain = self._setup_rag_and_history()
        self.quiz_eval_chain = self._create_quiz_eval_chain()

        if self.rag_chain and self.quiz_eval_chain:
            print("챗봇 로직이 정상적으로 로드되었습니다.")
        else:
            print("[중요] 챗봇 로직 초기화에 실패했습니다. 터미널의 오류 메시지를 확인하세요.")

    def _get_instruct(self, path):
        # instruct 폴더의 base.txt, few_shot.txt 등을 읽어 챗봇의 기본 페르소나를 설정합니다.
        file_list = ['base', 'few_shot']
        instruction_template = ''
        try:
            for file_name in file_list:
                with open(f'{path}/{file_name}.txt', 'r', encoding='utf-8-sig') as f:
                    instruction_template += f.read() + "\n"
            return instruction_template
        except FileNotFoundError: return "너는 친절한 친구야."

    def _load_quiz_data(self, file_path):
        # rag_data/quiz_data.txt 파일에서 퀴즈 목록을 불러옵니다.
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
        # 일반 대화를 처리하기 위한 RAG(검색 증강 생성) 체인을 설정합니다.
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"); docs = TextLoader("rag_data/info.txt", encoding='utf-8').load()
            retriever = Chroma.from_documents(RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs), embeddings).as_retriever()
            prompt = ChatPromptTemplate.from_messages([("system", f"{self.instruct}\n\n[참고 정보]\n{{context}}"), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")])
            rag_chain_main = (RunnablePassthrough.assign(context=lambda x: retriever.get_relevant_documents(x["input"]))| prompt| self.model| StrOutputParser())
            return RunnableWithMessageHistory(rag_chain_main, self._get_session_history, input_messages_key="input", history_messages_key="chat_history")
        except Exception as e: print(f"[오류] RAG 체인 설정 중 문제 발생: {e}"); return None

    # <<< 핵심 수정: 시스템 프롬프트에서 불필요한 변수 {답변}을 제거하고 규칙을 명료화 >>>
    def _create_quiz_eval_chain(self):
        # AI가 스스로 추론하여 아이의 답변을 채점하도록 '생각의 사슬' 체인을 생성합니다.
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
                { # 예시 1: 내용이 있는 정답
                    "input": "핵심 개념: 안된다고 말하고, 엄마 아빠와 정한 비밀 암호를 물어봐야 한다.\n답변: 엄마한테 가야 한다고 말할 거예요.",
                    "output": """[분석 기준 설정]이 질문의 핵심은 낯선 사람을 따라가지 않는 '거절'과 '안전 확보' 행동이야. 아이의 답변이 이 두 가지 중 하나라도 충족시키는지 확인해야 해.
[답변 분석 및 결론]음, "엄마한테 가야 한다"고 말하는 건, 낯선 사람을 안 따라가고 제일 안전한 엄마한테 가려는 거니까, 아주 똑똑한 행동이야. 위험한 상황을 잘 피했어.
[최종 판단][판단: 참]"""
                },
                { # 예시 2: 내용이 없는 오답 (최우선 규칙 적용)
                    "input": "핵심 개념: 알록달록 예쁜 약을 사탕인 줄 알고 먹어도 될까?\n답변: 아니",
                    "output": """[분석 기준 설정]이 질문은 '하면 안 되는 행동'에 대한 판단력을 묻고 있어.
[답변 분석 및 결론]아이의 답변 '아니'는 올바른 판단을 내린 것이지만, 너무 짧아서 왜 안 되는지 이해했는지는 알 수 없어. 더 구체적인 설명이 필요해.
[최종 판단][판단: 거짓]"""
                },
                { # 예시 3: 내용이 있는 오답
                    "input": "핵심 개념: 젖은 수건으로 입과 코를 막고, 몸을 낮춰서 기어서 대피해야 한다.\n답변: 119에 전화할 거예요.",
                    "output": """[분석 기준 설정]이 질문의 핵심은 '어떻게 대피하는가'라는 '대피 방법'에 대한 거야. 답변이 연기를 피하고 안전하게 이동하는 행동을 묘사하는지 확인해야 해.
[답변 분석 및 결론]아이의 답변 '119 신고'는 화재 시 매우 중요하고 올바른 행동이지만, 질문이 요구하는 '대피 방법' 그 자체는 아니야. 질문의 핵심을 벗어난 답변이야.
[최종 판단][판단: 거짓]"""
                }
            ]
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                *sum([[("human", ex["input"]), ("ai", ex["output"])] for ex in few_shot_examples], []),
                ("human", "핵심 개념: {answer}\n답변: {user_input}")
            ])
            
            print("[KeyError 수정 버전] 퀴즈 채점 체인 생성 완료.")
            return prompt | self.model | StrOutputParser()
        except Exception as e:
            print(f"[오류] 퀴즈 채점 체인 생성 중 문제 발생: {e}")
            return None

    def _get_session_history(self, session_id: str):
        # 세션 ID별로 대화 기록을 관리합니다.
        if session_id not in self.store: self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def handle_quiz(self, user_input, session_id):
        # 퀴즈 모드일 때의 대화 로직을 처리합니다.
        quiz_state = self.quiz_mode.get(session_id)
        MAX_ATTEMPTS = 2

        if quiz_state: # 이미 퀴즈가 진행 중인 경우
            current_quiz = quiz_state['quiz_item']
            
            eval_result_text = self.quiz_eval_chain.invoke({"answer": current_quiz['answer'], "user_input": user_input})
            
            print(f"\n--- LLM 채점 시작 ---\n{eval_result_text}\n--- LLM 채점 종료 ---\n")

            is_correct = "[판단: 참]" in eval_result_text
            
            try:
                thought_process = re.search(r"\[답변 분석 및 결론\](.*?)(?=\[최종 판단\])", eval_result_text, re.DOTALL).group(1).strip()
            except AttributeError:
                thought_process = None

            if is_correct:
                del self.quiz_mode[session_id]
                response = "딩동댕! 정답이야!\n"
                if thought_process:
                    response += f"\n[꾸로의 생각]\n{thought_process}\n"
                response += f"\n그래서 모범 답안은 바로 이거야!\n**{current_quiz['answer']}**\n\n정말 똑똑한걸? 또 퀴즈 풀고 싶으면 '퀴즈'라고 말해줘!"
                return response

            else: # 오답일 경우
                quiz_state['attempts'] += 1
                if quiz_state['attempts'] < MAX_ATTEMPTS:
                    if thought_process:
                        hint_message = f"음... 네 생각도 일리가 있어! 하지만 꾸로가 조금 더 깊이 생각해봤는데,\n\n[꾸로의 생각]\n{thought_process}\n\n그래서 완벽한 정답은 아닌 것 같아. 내가 진짜 힌트를 줄게!\n\n"
                    else:
                        hint_message = "음... 그럴듯한 답변이지만, 더 중요한 점이 있는 것 같아! 자, 진짜 힌트를 줄게!\n\n"
                    
                    hint_message += f"힌트: {current_quiz['hint']}\n\n이 힌트를 보고 다시 한번 생각해볼래?"
                    return hint_message
                else:
                    del self.quiz_mode[session_id]
                    return f"아쉽다! 정답은 '{current_quiz['answer']}'이었어. 괜찮아, 이렇게 하나씩 배우는 거지! 다음엔 꼭 맞힐 수 있을 거야."

        else: # 새로운 퀴즈를 시작하는 경우
            if not self.quiz_data: return "미안, 지금은 퀴즈를 낼 수 없어."
            quiz = random.choice(self.quiz_data)
            self.quiz_mode[session_id] = {'quiz_item': quiz, 'attempts': 0}
            return f"좋아, 재미있는 안전 퀴즈 시간! \n\n{quiz['question']}"
    
    def invoke(self, user_input, session_id):
        # 사용자의 입력에 따라 퀴즈 모드 또는 일반 대화 모드로 분기합니다.
        if session_id in self.quiz_mode or any(k in user_input for k in ["퀴즈", "문제", "게임"]):
            return self.handle_quiz(user_input, session_id)
        
        if not self.rag_chain: return "챗봇 로직이 초기화되지 않았습니다."
        try:
            return self.rag_chain.invoke({"input": user_input}, config={'configurable': {'session_id': session_id}})
        except Exception as e: print(f"[오류] 대화 생성 중 문제 발생: {e}"); return "미안, 지금은 대답하기가 좀 힘들어."

# FastAPI를 이용해 챗봇 API 서버를 설정하고 실행합니다.
app = FastAPI(
    title="키즈케어 챗봇 '꾸로' API",
    description="KeyError가 수정된 최종 버전의 챗봇입니다.",
    version="7.2.0"
)
class ChatRequest(BaseModel): user_input: str; session_id: str
chatbot_logic = ChatbotLogic()
@app.post("/chat")
def chat(request: ChatRequest): return {"response": chatbot_logic.invoke(request.user_input, request.session_id)}

if __name__ == "__main__":
    print("FastAPI 서버를 시작합니다. 접속 주소: http://127.0.0.1:8000")
    print("API 문서는 http://127.0.0.1:8000/docs 에서 확인하세요.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
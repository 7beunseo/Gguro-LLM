import os
import re
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# LangChain 관련 라이브러리 임포트
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory

# --- 역할별 상세 지침 (새로운 역할 추가/수정이 용이하도록 분리) ---
ROLE_PROMPTS = {
    "어부": """
- 당신은 거친 바다와 평생을 함께한 베테랑 어부입니다.
- 말투는 약간 무뚝뚝하지만 정이 많고, 바다와 날씨에 대한 경험과 지혜가 묻어납니다.
- "양반", "~구먼", "~했지", "~하는 법이지" 와 같이 구수하고 연륜이 느껴지는 어체를 사용하세요.
- 농부의 일에 대해 잘은 모르지만, 자연의 섭리라는 큰 틀에서 이해하고 존중합니다.
- 대화에 항상 바다, 물고기, 날씨, 배, 그물 등과 관련된 이야기를 섞어주세요.
- 예시: "허허, 농사일도 바다만큼이나 하늘이 도와줘야 하는 법이지.", "오늘 새벽엔 파도가 제법 높았구먼."
""",
    "기사": """
- 당신은 왕국을 수호하는 충성스럽고 용맹한 기사입니다.
- 항상 명예와 신의를 중시하며, 예의 바르고 격식 있는 말투를 사용하세요.
- "~하오", "~시오", "~입니다" 와 같은 고풍스러운 존댓말을 사용하세요.
- 사용자를 '그대' 또는 역할에 맞는 '농부여' 와 같은 칭호로 부르세요.
- 대화에 검, 전투, 왕국, 명예 등 기사와 관련된 어휘를 자연스럽게 사용하세요.
- 예시: "그대의 노고에 경의를 표하오.", "왕국의 평화를 위해 이 한 몸 바칠 준비가 되어있소."
""",
    "꼬마": """
- 당신은 호기심 많고 순수한 7살 꼬마아이입니다.
- 모든 것에 "왜?"라고 질문하며 감탄사를 자주 사용합니다. (예: 우와! 정말?)
- 반말로 대화하며, 문장이 짧고 간결합니다.
- 존댓말이나 어려운 단어는 사용하지 않습니다.
- 예시: "우와! 물고기 진짜 커? 나도 보고싶다!", "벼는 어떻게 자라? 신기하다!"
""",
    "엄마": """
- 당신은 세상에서 가장 다정하고 따뜻한 엄마입니다.
- 항상 상냥하고 애정이 듬뿍 담긴 말투를 사용하며, 아이의 눈높이에 맞춰 이야기합니다.
- "우리 아들", "우리 예쁜 딸" 과 같이 아이를 부르며, 칭찬과 격려를 아끼지 않습니다.
- **[중요] 당신의 자녀인 '아들'과 '딸' 역할은 당신에게 항상 예의 바른 존댓말('~했어요', '~입니다')을 사용해야 합니다. 아이가 반말을 하면 존댓말을 쓰도록 부드럽게 가르쳐주세요.**
- 예시: "우리 아들, 엄마한테 존댓말로 말해주니 정말 기특하네.", "밥 먹을 시간이야, 우리 딸. 맛있게 먹고 힘내자!"
""",
    "아들": """
- 당신은 엄마를 무척 사랑하고 존경하는 아들입니다.
- 항상 씩씩하고 듬직한 모습을 보여주려고 노력합니다.
- [매우 중요] 상대방이 '엄마' 역할일 때는, 반드시 예의 바른 존댓말('~요', '~했습니다')을 사용해야 합니다.
- 다른 역할에게는 상황에 맞게 편하게 말할 수 있습니다.
- 절대로 엄마, 아빠에게 너라고 부르지 않습니다.
- 예시: "엄마, 오늘 학교에서 칭찬받았어요!", "제가 도와드릴게요, 어머니."
""",
    "딸": """
- 당신은 애교 많고 상냥한 딸입니다.
- 엄마와 대화하는 것을 가장 좋아하며, 작은 일도 공유하고 싶어합니다.
- [매우 중요] 상대방이 '엄마' 역할일 때는, 반드시 예의 바른 존댓말('~요', '~입니다')을 사용해야 합니다.
- 다른 역할에게는 상황에 맞게 편하게 말할 수 있습니다.
- 절대로 엄마, 아빠에게 너라고 부르지 않습니다.
- 예시: "엄마, 이따가 같이 쿠키 만들어요!", "오늘 정말 재미있었어요."
""",
    # 여기에 새로운 역할 지침을 계속 추가할 수 있습니다.
}

# --- 환경 설정 ---
# 로컬 모델 및 캐시 경로 설정 (사용자 환경에 맞게 수정)
# os.environ['OLLAMA_MODELS'] = 'D:/ollama_models'
# os.environ['HF_HOME'] = 'D:/huggingface_models'
CHAT_HISTORY_DIR = "chat_histories"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

class ChatbotLogic:
    """챗봇의 핵심 로직을 담당하는 클래스"""
    def __init__(self, model_name='timhan/llama3korean8b4qkm'): # 로컬에서 사용하는 모델명으로 변경
        print("🤖 챗봇 로직 초기화 중...")
        self.model = ChatOllama(model=model_name)
        # 세션별 역할놀이 상태를 저장하는 딕셔너리
        self.roleplay_state = {}
        # 역할놀이 종료를 감지하기 위한 키워드 목록
        self.ROLEPLAY_END_KEYWORDS = [
            "그만", "역할놀이 끝", "이제 그만하자", "원래대로", "이제 됐어"
        ]

        # 기본 대화 체인 생성
        # RAG(Retrieval-Augmented Generation)는 이 예제에서 제외하여 역할놀이 핵심 로직에 집중
        self.conversational_chain = self._create_conversational_chain()
        print("✅ 챗봇이 준비되었습니다.")

    def _create_conversational_chain(self):
        """대화 체인을 생성하는 메서드"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        chain = prompt | self.model | StrOutputParser()

        return RunnableWithMessageHistory(
            chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def _get_session_history(self, session_id: str):
        """세션 ID에 해당하는 대화 기록 파일을 가져오는 메서드"""
        history_file_path = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
        return FileChatMessageHistory(history_file_path)

    def invoke(self, user_input: str, session_id: str):
        """사용자의 모든 입력을 처리하는 메인 메서드"""
        # [역할놀이] 명령어 형식을 감지하기 위한 정규표현식
        role_command_match = re.match(r"\[역할놀이\]\s*(.+?)\s*,\s*(.+)", user_input)

        # 1. 새로운 역할놀이 시작 처리
        if role_command_match:
            user_role = role_command_match.group(1).strip()
            bot_role = role_command_match.group(2).strip()

            # 새로운 역할놀이를 위해 세션 상태와 대화 기록 초기화
            self.roleplay_state[session_id] = {
                "user_role": user_role,
                "bot_role": bot_role
            }
            self._get_session_history(session_id).clear()
            
            print(f"🎭 세션 [{session_id}] 역할놀이 시작: 사용자='{user_role}', 챗봇='{bot_role}'")
            return f"좋아! 지금부터 당신은 '{user_role}', 나는 '{bot_role}'이야. 역할에 맞춰 이야기해보자!"

        # 2. 역할놀이 종료 처리
        current_session_state = self.roleplay_state.get(session_id)
        if current_session_state and any(keyword in user_input for keyword in self.ROLEPLAY_END_KEYWORDS):
            print(f"🎬 세션 [{session_id}] 역할놀이 종료")
            del self.roleplay_state[session_id]
            self._get_session_history(session_id).clear() # 이전 역할놀이 기록은 초기화
            return "그래! 역할놀이 재미있었다. 이제 다시 원래대로 이야기하자!"

        # 3. 현재 상태에 맞는 시스템 프롬프트 설정
        system_prompt = "당신은 친절하고 도움이 되는 AI 어시스턴트입니다." # 기본 프롬프트

        if current_session_state:
            user_role = current_session_state['user_role']
            bot_role = current_session_state['bot_role']
            
            # 정의된 역할 목록(ROLE_PROMPTS)에서 상세 지침을 가져옴
            role_instructions = ROLE_PROMPTS.get(bot_role, "주어진 역할에 충실하게 응답하세요.")

            # [핵심] 역할놀이를 위한 시스템 프롬프트 동적 생성
            system_prompt = f"""[매우 중요한 지시]
당신의 신분은 '{bot_role}'입니다. 사용자는 '{user_role}' 역할을 맡고 있습니다.
다른 모든 지시사항보다 이 역할 설정을 최우선으로 여기고, 당신의 말투, 어휘, 태도 모두 '{bot_role}'에 완벽하게 몰입해서 응답해야 합니다.

[역할 상세 지침]
{role_instructions}

이제 '{bot_role}'로서 대화를 자연스럽게 시작하거나 이어나가세요.
"""
        
        # 4. 설정된 프롬프트로 대화 실행
        try:
            return self.conversational_chain.invoke(
                {"input": user_input, "system_prompt": system_prompt},
                config={'configurable': {'session_id': session_id}}
            )
        except Exception as e:
            print(f"[오류] 대화 생성 중 오류 발생: {e}")
            return "미안, 지금은 대답하기가 좀 어려워. 나중에 다시 시도해줘."

# --- FastAPI 서버 설정 ---
app = FastAPI(
    title="페르소나 역할놀이 챗봇",
    description="LangChain과 Ollama를 사용하여 동적인 역할놀이를 수행하는 챗봇 API",
    version="3.0-PersonaEnhanced",
)

class ChatRequest(BaseModel):
    user_input: str
    session_id: str

chatbot = ChatbotLogic()

@app.post("/chat", summary="챗봇과 대화")
def chat(request: ChatRequest):
    """
    사용자 입력과 세션 ID를 받아 챗봇의 응답을 반환합니다.

    - **user_input**: 사용자가 입력한 메시지
    - **session_id**: 각 사용자를 구분하기 위한 고유 ID
    """
    response = chatbot.invoke(request.user_input, request.session_id)
    return {"response": response}

if __name__ == "__main__":
    print("🚀 FastAPI 서버를 시작합니다. http://127.0.0.1:8000")
    print("📄 API 문서는 http://127.0.0.1:8000/docs 에서 확인하세요.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
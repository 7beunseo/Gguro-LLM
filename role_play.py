import os
import sys
import random
import re
# --- [DB 변경] 'mysql.connector' 대신 'psycopg2'를 임포트합니다. ---
import psycopg2
from psycopg2 import Error
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

# --- [DB 변경] PostgreSQL 데이터베이스 설정 (사용자 환경에 맞게 수정) ---
DB_CONFIG = {
    'dbname': 'gguro',                  # 사용할 데이터베이스(스키마) 이름
    'user': 'postgres',       # PostgreSQL 사용자 이름
    'password': 'km923009!!', # PostgreSQL 비밀번호
    'host': 'localhost',                # 데이터베이스 호스트 주소
    'port': '5432'                      # PostgreSQL 기본 포트
}

# --- [추가] 역할놀이 상세 지침 ---
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
}

# --- 챗봇 로직 클래스 (최종 수정 버전) ---
class ChatbotLogic:
    def __init__(self, model_name='timhan/llama3korean8b4qkm'):
        print("🤖 역할놀이 챗봇 로딩 시작...")
        self.model = ChatOllama(model=model_name)
        self.store = {}
        
        # [수정] 역할놀이 및 일반 대화 종료 키워드 정의
        self.ROLEPLAY_END_KEYWORDS = ["역할놀이 끝", "원래대로"]
        self.END_KEYWORDS = ["끝", "종료", "그만", "대화 종료"]
        
        # 역할놀이 상태 관리
        self.roleplay_state = {}

        # 분석용 체인들 초기화
        self.analysis_chain = self._create_analysis_chain()
        self.summarization_chain = self._create_summarization_chain()
        self.conversational_chain = self._create_conversational_chain()

        # DB 테이블 준비
        self._ensure_table_exists()

        if all([self.conversational_chain, self.analysis_chain, self.summarization_chain]):
            print("✅ 챗봇 로직이 정상적으로 로드되었습니다.")
        else:
            print("[중요] 챗봇 로직 초기화에 실패했습니다!")

    def _get_base_prompt(self):
        """기본 시스템 프롬프트를 반환합니다."""
        return "당신은 아이들의 눈높이에 맞춰 대화하는 다정한 AI 친구 '꾸로'입니다. 항상 친절하고 상냥하게 대답해주세요."

    def _create_analysis_chain(self):
        """텍스트의 감정을 분석하고 키워드를 추출하는 LangChain 체인을 생성합니다."""
        try:
            prompt = ChatPromptTemplate.from_template("""당신은 주어진 텍스트에서 사용자의 감정과 그 대상이 되는 핵심 키워드를 추출하는 전문가입니다.
텍스트를 분석하여 긍정적인지 부정적인지 판단하고, 감정과 관련된 **핵심 대상과 감정 단어**를 모두 포함하여 키워드를 3개 이내로 추출하세요.
결과는 반드시 다음 형식으로만 출력해야 합니다. 다른 설명은 절대 추가하지 마세요.
[예시]
분석할 텍스트: 난 참외 싫어해
[판단: 부정]
[키워드: 참외, 싫어]
---
분석할 텍스트:
{text}
---
""")
            return prompt | self.model | StrOutputParser()
        except Exception as e:
            print(f"[오류] 감정 분석 체인 생성 중 문제 발생: {e}"); return None

    def _create_summarization_chain(self):
        """대화 내용을 요약하는 체인을 생성합니다."""
        try:
            prompt = ChatPromptTemplate.from_template("""다음 [대화 내용]을 한 문장으로 요약해줘. 요약 결과 외에 다른 설명은 절대 붙이지 마.
[대화 내용]:
{history}
---
[요약]:""")
            return prompt | self.model | StrOutputParser()
        except Exception as e:
            print(f"[오류] 요약 체인 생성 중 문제 발생: {e}"); return None
            
    def _create_conversational_chain(self):
        """대화 체인을 생성합니다."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            chain = prompt | self.model | StrOutputParser()

            return RunnableWithMessageHistory(
                chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                system_message_key="system_prompt" 
            )
        except Exception as e:
            print(f"[오류] 대화 체인 설정 중 심각한 문제 발생: {e}"); return None

    def _get_session_history(self, session_id: str):
        """세션별 대화 기록 객체를 관리합니다."""
        if session_id not in self.store:
            self.store[session_id] = {'history': InMemoryChatMessageHistory(), 'chatroom_id': None}
        return self.store[session_id]['history']
    
    def _create_db_connection(self):
        """PostgreSQL 데이터베이스 연결을 생성합니다."""
        try:
            return psycopg2.connect(**DB_CONFIG)
        except Error as e:
            print(f"[DB 오류] PostgreSQL 데이터베이스 연결 실패: {e}"); return None

    def _ensure_table_exists(self):
        """데이터베이스에 ChatRoom과 Talk 테이블 및 관련 객체들을 생성합니다."""
        conn = self._create_db_connection()
        if conn is None: return
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chatroom (
                        id BIGSERIAL PRIMARY KEY,
                        profile_id BIGINT NOT NULL,
                        topic TEXT,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS talk (
                        id BIGSERIAL PRIMARY KEY,
                        chatroom_id BIGINT NOT NULL REFERENCES chatroom(id),
                        category VARCHAR(50) NOT NULL CHECK (category IN ('OBJECTPLAY', 'LIFESTYLEHABIT', 'SAFETYSTUDY', 'ANIMALKNOWLEDGE', 'ROLEPLAY')),
                        content TEXT NOT NULL,
                        session_id VARCHAR(255),
                        role VARCHAR(255),
                        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        profile_id BIGINT NOT NULL,
                        "like" BOOLEAN,
                        positive BOOLEAN DEFAULT TRUE,
                        keywords TEXT[]
                    );
                """)
                cursor.execute("""
                    CREATE OR REPLACE FUNCTION update_updated_at_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                       NEW.updated_at = NOW(); 
                       RETURN NEW;
                    END;
                    $$ language 'plpgsql';
                """)
                cursor.execute("""
                    DROP TRIGGER IF EXISTS update_talk_updated_at ON talk;
                    CREATE TRIGGER update_talk_updated_at
                    BEFORE UPDATE ON talk
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
                """)
            conn.commit()
            print("[DB 정보] 'chatroom' 및 'talk' 테이블이 준비되었습니다.")
        except Error as e:
            print(f"[DB 오류] 테이블 생성 실패: {e}"); conn.rollback()
        finally:
            if conn: conn.close()
    
    async def _summarize_and_close_room(self, session_id: str):
        """현재 채팅방을 요약하고, 세션에서 채팅방 ID를 제거하여 대화를 종료 상태로 만듭니다."""
        session_state = self.store.get(session_id)
        if not session_state or not session_state.get('chatroom_id'):
            return

        current_chatroom_id = session_state['chatroom_id']
        history = session_state['history']
        
        if history.messages:
            conn = self._create_db_connection()
            if conn is None: return
            try:
                with conn.cursor() as cursor:
                    full_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages])
                    summary = await self.summarization_chain.ainvoke({"history": full_history_str})
                    summary = summary.strip().replace("'", "''")
                    
                    # [수정] 역할놀이 요약 시, 저장 형식을 변경합니다.
                    if session_id in self.roleplay_state:
                        summary = f"[역할놀이] {summary}"
                    
                    cursor.execute("UPDATE chatroom SET topic = %s WHERE id = %s", (summary, current_chatroom_id))
                    conn.commit()
                    print(f"채팅방({current_chatroom_id}) 요약 완료 및 저장: {summary}")
            except Error as e:
                print(f"[DB 오류] 채팅방 요약 중 오류: {e}"); conn.rollback()
            finally:
                if conn: conn.close()

        session_state['chatroom_id'] = None
        history.clear()
        if session_id in self.roleplay_state:
            del self.roleplay_state[session_id]
        print(f"세션({session_id})의 채팅방이 종료되었습니다.")

    async def _create_new_chatroom(self, session_id: str, profile_id: int):
        """새로운 채팅방을 생성하고 세션에 ID를 할당합니다."""
        session_state = self.store.setdefault(session_id, {'history': InMemoryChatMessageHistory(), 'chatroom_id': None})
        conn = self._create_db_connection()
        if conn is None: return None
        try:
            with conn.cursor() as cursor:
                topic = "새로운 역할놀이" if session_id in self.roleplay_state else "새로운 대화"
                cursor.execute("INSERT INTO chatroom (profile_id, topic) VALUES (%s, %s) RETURNING id", (profile_id, topic))
                new_chatroom_id = cursor.fetchone()[0]
                session_state['chatroom_id'] = new_chatroom_id
                conn.commit()
                print(f"새 채팅방 생성: {new_chatroom_id}")
                return new_chatroom_id
        except Error as e:
            print(f"[DB 오류] 새 채팅방 생성 중 오류: {e}"); conn.rollback()
            return None
        finally:
            if conn: conn.close()

    async def _analyze_and_save_message(self, session_id: str, role: str, text: str, profile_id: int, chatroom_id: int):
        """메시지를 분석하고 DB에 저장하는 내부 헬퍼 함수"""
        if not self.analysis_chain:
            is_positive, keywords_list = True, []
        else:
            try:
                analysis_result = await self.analysis_chain.ainvoke({"text": text})
                is_positive = "부정" not in analysis_result
                keywords_match = re.search(r"\[키워드:\s*(.*)\]", analysis_result)
                keywords_list = [k.strip() for k in keywords_match.group(1).split(',') if k.strip()] if keywords_match else []
                print(f"분석 결과 ({'긍정' if is_positive else '부정'}): {text} -> {keywords_list}")
            except Exception as e:
                print(f"[오류] '{text}' 분석 중 오류 발생: {e}"); is_positive, keywords_list = True, []

        conn = self._create_db_connection()
        if conn is None: return
        try:
            with conn.cursor() as cursor:
                query = """
                    INSERT INTO talk (session_id, role, content, category, profile_id, positive, keywords, "like", chatroom_id) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NULL, %s)
                """
                category = 'ROLEPLAY' if session_id in self.roleplay_state else 'LIFESTYLEHABIT'
                cursor.execute(query, (session_id, role, text, category, profile_id, is_positive, keywords_list, chatroom_id))
            conn.commit()
        except Error as e:
            print(f"[DB 오류] 메시지 저장 실패: {e}"); conn.rollback()
        finally:
            if conn: conn.close()

    async def save_conversation_to_db(self, session_id: str, user_input: str, bot_response: str, chatroom_id: int, profile_id: int):
        """사용자 입력과 봇 응답을 분석하고 순차적으로 DB에 저장합니다."""
        print(f"📝 백그라운드 저장 및 분석 시작: 채팅방 [{chatroom_id}]")
        await self._analyze_and_save_message(session_id, 'user', user_input, profile_id, chatroom_id)
        await self._analyze_and_save_message(session_id, 'bot', bot_response, profile_id, chatroom_id)
        print(f"✅ 백그라운드 저장 및 분석 완료: 채팅방 [{chatroom_id}]")

    async def invoke(self, user_input: str, session_id: str, profile_id: int):
        """챗봇의 메인 실행 함수. 역할놀이, 대화 종료, 채팅방 관리, 응답 생성을 총괄합니다."""
        
        # 1. 역할놀이 시작 명령어 확인
        role_command_match = re.match(r"\[역할놀이\]\s*(.+?)\s*,\s*(.+)", user_input)
        if role_command_match:
            user_role, bot_role = role_command_match.groups()
            await self._summarize_and_close_room(session_id)
            
            self.roleplay_state[session_id] = {"user_role": user_role.strip(), "bot_role": bot_role.strip()}
            print(f"🎭 세션 [{session_id}] 역할놀이 시작: 사용자='{user_role.strip()}', 챗봇='{bot_role.strip()}'")
            
            chatroom_id = await self._create_new_chatroom(session_id, profile_id)
            response_text = f"좋아! 지금부터 너는 '{user_role.strip()}', 나는 '{bot_role.strip()}'이야. 역할에 맞춰 이야기해보자!"
            return response_text, chatroom_id

        # 2. 역할놀이 종료 명령어 확인
        if session_id in self.roleplay_state and any(keyword in user_input for keyword in self.ROLEPLAY_END_KEYWORDS):
            await self._summarize_and_close_room(session_id)
            return "그래! 역할놀이 재미있었다. 이제 다시 원래대로 이야기하자!", None

        # [추가] 일반 대화 종료 명령어 확인
        if session_id not in self.roleplay_state and any(keyword in user_input for keyword in self.END_KEYWORDS):
            await self._summarize_and_close_room(session_id)
            return "알겠습니다. 대화를 종료하고 요약했습니다. 새로운 대화를 시작할 수 있습니다.", None

        # 3. 현재 세션의 채팅방 ID 가져오거나 새로 생성
        session_state = self.store.get(session_id, {})
        chatroom_id = session_state.get('chatroom_id')

        if not chatroom_id:
            print(f"세션({session_id})에 활성 채팅방이 없어 새로 시작합니다.")
            chatroom_id = await self._create_new_chatroom(session_id, profile_id)
            if not chatroom_id:
                return "채팅방을 만드는 데 문제가 발생했어요.", None

        if not self.conversational_chain:
            return "챗봇 로직 초기화에 실패했습니다.", chatroom_id
            
        # 4. 시스템 프롬프트 결정 및 응답 생성
        system_prompt_text = self._get_base_prompt()
        if session_id in self.roleplay_state:
            state = self.roleplay_state[session_id]
            bot_role = state['bot_role']
            role_instructions = ROLE_PROMPTS.get(bot_role, "주어진 역할에 충실하게 응답하세요.")
            system_prompt_text = f"""[매우 중요한 지시]
당신의 신분은 '{bot_role}'입니다. 사용자는 '{state['user_role']}' 역할을 맡고 있습니다.
다른 모든 지시사항보다 이 역할 설정을 최우선으로 여기고, 당신의 말투, 어휘, 태도 모두 '{bot_role}'에 완벽하게 몰입해서 응답해야 합니다.
[역할 상세 지침]
{role_instructions}
이제 '{bot_role}'로서 대화를 자연스럽게 시작하거나 이어나가세요."""

        try:
            response = await self.conversational_chain.ainvoke(
                {"input": user_input, "system_prompt": system_prompt_text},
                config={'configurable': {'session_id': session_id}}
            )
            return response, chatroom_id
        except Exception as e:
            print(f"[오류] 대화 생성 중 문제 발생: {e}")
            return "미안, 지금은 대답하기가 좀 힘들어.", chatroom_id

# --- FastAPI 서버 설정 ---
app = FastAPI(title="역할놀이 챗봇 (채팅방 자동 관리)")

class ChatRequest(BaseModel):
    user_input: str
    session_id: str
    # profile_id: int

chatbot_logic = ChatbotLogic()

@app.post("/chat", summary="챗봇과 대화하기")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    사용자 입력을 받아 챗봇의 응답을 즉시 반환하고,
    대화 내용은 백그라운드에서 분석 후 데이터베이스에 저장합니다.
    """
    profile_id = 1 # 임시 프로필 ID
    response_text, chatroom_id = await chatbot_logic.invoke(request.user_input, request.session_id, profile_id)
    
    # [수정] 모든 종료 명령어들을 확인하도록 수정
    is_end_command = any(keyword in request.user_input for keyword in chatbot_logic.ROLEPLAY_END_KEYWORDS + chatbot_logic.END_KEYWORDS) or \
                     re.match(r"\[역할놀이\]", request.user_input)

    if chatroom_id and not is_end_command:
        background_tasks.add_task(
            chatbot_logic.save_conversation_to_db,
            session_id=request.session_id,
            user_input=request.user_input,
            bot_response=response_text,
            chatroom_id=chatroom_id,
            profile_id=profile_id
        )
    
    return {"response": response_text}

# uvicorn으로 이 파일을 실행하기 위한 메인 블록
if __name__ == "__main__":
    print("🚀 FastAPI 서버를 시작합니다. http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

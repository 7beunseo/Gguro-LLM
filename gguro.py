import os
import sys
import random
import re
import psycopg2
from psycopg2 import Error
from datetime import date
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn
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

# --- 환경 설정 ---
os.environ['OLLAMA_MODELS'] = 'D:/ollama_models'
os.environ['HF_HOME'] = 'D:/huggingface_models'

# --- DB 설정 ---
DB_CONFIG = {
    'dbname': 'gguro',
    'user': 'postgres',
    'password': 'km923009!!',
    'host': 'localhost',
    'port': '5432'
}

# --- 역할놀이 프롬프트 ---
ROLE_PROMPTS = {
    "어부": """- 당신은 거친 바다와 평생을 함께한 베테랑 어부입니다. 말투는 약간 무뚝뚝하지만 정이 많고, 바다와 날씨에 대한 경험과 지혜가 묻어납니다. "양반", "~구먼", "~했지", "~하는 법이지" 와 같이 구수하고 연륜이 느껴지는 어체를 사용하세요. 농부의 일에 대해 잘은 모르지만, 자연의 섭리라는 큰 틀에서 이해하고 존중합니다. 대화에 항상 바다, 물고기, 날씨, 배, 그물 등과 관련된 이야기를 섞어주세요. 예시: "허허, 농사일도 바다만큼이나 하늘이 도와줘야 하는 법이지.", "오늘 새벽엔 파도가 제법 높았구먼." """,
    "기사": """- 당신은 왕국을 수호하는 충성스럽고 용맹한 기사입니다. 항상 명예와 신의를 중시하며, 예의 바르고 격식 있는 말투를 사용하세요. "~하오", "~시오", "~입니다" 와 같은 고풍스러운 존댓말을 사용하세요. 사용자를 '그대' 또는 역할에 맞는 '농부여' 와 같은 칭호로 부르세요. 대화에 검, 전투, 왕국, 명예 등 기사와 관련된 어휘를 자연스럽게 사용하세요. 예시: "그대의 노고에 경의를 표하오.", "왕국의 평화를 위해 이 한 몸 바칠 준비가 되어있소." """,
    "꼬마": """- 당신은 호기심 많고 순수한 7살 꼬마아이입니다. 모든 것에 "왜?"라고 질문하며 감탄사를 자주 사용합니다. (예: 우와! 정말?) 반말로 대화하며, 문장이 짧고 간결합니다. 존댓말이나 어려운 단어는 사용하지 않습니다. 예시: "우와! 물고기 진짜 커? 나도 보고싶다!", "벼는 어떻게 자라? 신기하다!" """,
    "엄마": """- 당신은 세상에서 가장 다정하고 따뜻한 엄마입니다. 항상 상냥하고 애정이 듬뿍 담긴 말투를 사용하며, 아이의 눈높이에 맞춰 이야기합니다. "우리 아들", "우리 예쁜 딸" 과 같이 아이를 부르며, 칭찬과 격려를 아끼지 않습니다. **[중요] 당신의 자녀인 '아들'과 '딸' 역할은 당신에게 항상 예의 바른 존댓말('~했어요', '~입니다')을 사용해야 합니다. 아이가 반말을 하면 존댓말을 쓰도록 부드럽게 가르쳐주세요.** 예시: "우리 아들, 엄마한테 존댓말로 말해주니 정말 기특하네.", "밥 먹을 시간이야, 우리 딸. 맛있게 먹고 힘내자!" """,
    "아들": """- 당신은 엄마를 무척 사랑하고 존경하는 아들입니다. 항상 씩씩하고 듬직한 모습을 보여주려고 노력합니다. [매우 중요] 상대방이 '엄마' 역할일 때는, 반드시 예의 바른 존댓말('~요', '~했습니다')을 사용해야 합니다. 다른 역할에게는 상황에 맞게 편하게 말할 수 있습니다. 절대로 엄마, 아빠에게 너라고 부르지 않습니다. 예시: "엄마, 오늘 학교에서 칭찬받았어요!", "제가 도와드릴게요, 어머니." """,
    "딸": """- 당신은 애교 많고 상냥한 딸입니다. 엄마와 대화하는 것을 가장 좋아하며, 작은 일도 공유하고 싶어합니다. [매우 중요] 상대방이 '엄마' 역할일 때는, 반드시 예의 바른 존댓말('~요', '~입니다')을 사용해야 합니다. 다른 역할에게는 상황에 맞게 편하게 말할 수 있습니다. 절대로 엄마, 아빠에게 너라고 부르지 않습니다. 예시: "엄마, 이따가 같이 쿠키 만들어요!", "오늘 정말 재미있었어요." """,
}

# --- 데이터 및 세션 관리 클래스 ---
class DatabaseManager:
    def __init__(self, model):
        self.model = model
        self.store = {}
        self.analysis_chain = self._create_analysis_chain()
        self.summarization_chain = self._create_summarization_chain()
        self._ensure_table_exists()

    def _create_db_connection(self):
        try:
            return psycopg2.connect(**DB_CONFIG)
        except Error as e:
            print(f"[DB 오류] PostgreSQL 연결 실패: {e}"); return None

    def _get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = {'history': InMemoryChatMessageHistory(), 'chatroom_id': None, 'type': 'conversation', 'roleplay_state': None, 'quiz_state': None}
        return self.store[session_id]['history']
    
    def _create_analysis_chain(self):
        try:
            prompt = ChatPromptTemplate.from_template("""당신은 주어진 텍스트에서 사용자의 감정과 그 대상이 되는 핵심 키워드를 추출하는 전문가입니다.
[예시]
분석할 텍스트: 난 참외 싫어해
[판단: 부정]
[키워드: 참외, 싫어]
---
분석할 텍스트:
{text}
---""")
            return prompt | self.model | StrOutputParser()
        except Exception as e:
            print(f"[오류] 감정 분석 체인 생성 실패: {e}"); return None

    def _create_summarization_chain(self):
        try:
            prompt = ChatPromptTemplate.from_template("""다음 [대화 내용]을 한 문장으로 요약해줘. 요약 결과 외에 다른 설명은 절대 붙이지 마.
[대화 내용]:
{history}
---
[요약]:""")
            return prompt | self.model | StrOutputParser()
        except Exception as e:
            print(f"[오류] 요약 체인 생성 실패: {e}"); return None

    def _ensure_table_exists(self):
        conn = self._create_db_connection()
        if conn is None: return
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chatroom (
                        id BIGSERIAL PRIMARY KEY, profile_id BIGINT NOT NULL, topic TEXT, created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );""")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS talk (
                        id BIGSERIAL PRIMARY KEY, chatroom_id BIGINT NOT NULL REFERENCES chatroom(id), category VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL, session_id VARCHAR(255), role VARCHAR(255), created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP, profile_id BIGINT NOT NULL, "like" BOOLEAN,
                        positive BOOLEAN DEFAULT TRUE, keywords TEXT[]
                    );""")
                cursor.execute("""
                    CREATE OR REPLACE FUNCTION update_updated_at_column() RETURNS TRIGGER AS $$
                    BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
                    $$ language 'plpgsql';""")
                cursor.execute("""
                    DROP TRIGGER IF EXISTS update_talk_updated_at ON talk;
                    CREATE TRIGGER update_talk_updated_at BEFORE UPDATE ON talk FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();""")
            conn.commit()
            print("[DB 정보] 'chatroom' 및 'talk' 테이블 준비 완료.")
        except Error as e:
            print(f"[DB 오류] 테이블 생성 실패: {e}"); conn.rollback()
        finally:
            if conn: conn.close()

    async def summarize_and_close_room(self, session_id: str):
        session_state = self.store.get(session_id)
        if not session_state or not session_state.get('chatroom_id'):
            print(f"세션({session_id})에 종료할 채팅방이 없습니다.")
            return

        current_chatroom_id = session_state['chatroom_id']
        history = session_state['history']
        
        summary = ""
        room_type = session_state.get('type', 'conversation')
        quiz_info = session_state.get('quiz_state')

        if room_type == 'quiz' and quiz_info:
            summary = f"[퀴즈] {quiz_info['quiz_item']['question']}"
        elif history.messages:
            full_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages])
            summary_text = await self.summarization_chain.ainvoke({"history": full_history_str})
            if room_type == 'roleplay' and session_state.get('roleplay_state'):
                summary = f"[역할놀이] {summary_text.strip()}"
            else:
                summary = f"[일상대화] {summary_text.strip()}"

        if summary:
            conn = self._create_db_connection()
            if conn is None: return
            try:
                with conn.cursor() as cursor:
                    cursor.execute("UPDATE chatroom SET topic = %s WHERE id = %s", (summary, current_chatroom_id))
                    conn.commit()
                print(f"채팅방({current_chatroom_id}) 요약 완료: {summary}")
            except Error as e:
                print(f"[DB 오류] 채팅방 요약 실패: {e}"); conn.rollback()
            finally:
                if conn: conn.close()

        self.store[session_id] = {'history': InMemoryChatMessageHistory(), 'chatroom_id': None, 'type': 'conversation', 'roleplay_state': None, 'quiz_state': None}
        print(f"세션({session_id})이 완전히 종료 및 초기화되었습니다.")

    async def create_new_chatroom(self, session_id: str, profile_id: int, room_type: str):
        await self.summarize_and_close_room(session_id)
        session_state = self.store.setdefault(session_id, {})
        session_state['history'] = InMemoryChatMessageHistory()
        session_state['type'] = room_type

        conn = self._create_db_connection()
        if conn is None: return None
        try:
            with conn.cursor() as cursor:
                topic_map = {'quiz': "새로운 퀴즈", 'roleplay': "새로운 역할놀이", 'conversation': "새로운 대화"}
                topic = topic_map.get(room_type, "새로운 대화")
                cursor.execute("INSERT INTO chatroom (profile_id, topic) VALUES (%s, %s) RETURNING id", (profile_id, topic))
                new_chatroom_id = cursor.fetchone()[0]
                session_state['chatroom_id'] = new_chatroom_id
                conn.commit()
                print(f"새 채팅방 생성 (타입: {room_type}, ID: {new_chatroom_id})")
                return new_chatroom_id
        except Error as e:
            print(f"[DB 오류] 새 채팅방 생성 실패: {e}"); conn.rollback(); return None
        finally:
            if conn: conn.close()

    async def save_conversation_to_db(self, session_id: str, user_input: str, bot_response: str, chatroom_id: int, profile_id: int):
        session_state = self.store.get(session_id, {})
        if not self.analysis_chain:
            is_positive, keywords_list = True, []
        else:
            try:
                analysis_result = await self.analysis_chain.ainvoke({"text": user_input})
                is_positive = "부정" not in analysis_result
                keywords_match = re.search(r"\[키워드:\s*(.*)\]", analysis_result)
                keywords_list = [k.strip() for k in keywords_match.group(1).split(',') if k.strip()] if keywords_match else []
            except Exception as e:
                print(f"[오류] 분석 중 오류 발생: {e}"); is_positive, keywords_list = True, []

        conn = self._create_db_connection()
        if conn is None: return
        try:
            with conn.cursor() as cursor:
                query = """
                    INSERT INTO talk (session_id, role, content, category, profile_id, positive, keywords, "like", chatroom_id) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NULL, %s)
                """
                category_map = {'quiz': 'SAFETYSTUDY', 'roleplay': 'ROLEPLAY', 'conversation': 'LIFESTYLEHABIT'}
                category = category_map.get(session_state.get('type', 'conversation'), 'LIFESTYLEHABIT')
                
                cursor.execute(query, (session_id, 'user', user_input, category, profile_id, is_positive, keywords_list, chatroom_id))
                cursor.execute(query, (session_id, 'bot', bot_response, category, profile_id, True, [], chatroom_id))
            conn.commit()
            print(f"✅ 채팅방[{chatroom_id}] 대화 저장 완료")
        except Error as e:
            print(f"[DB 오류] 메시지 저장 실패: {e}"); conn.rollback()
        finally:
            if conn: conn.close()

# --- 역할놀이 로직 클래스 ---
class RolePlayLogic:
    def __init__(self, model, db_manager):
        self.model = model
        self.db_manager = db_manager
        self.conversational_chain = self._create_conversational_chain()
    
    def _create_conversational_chain(self):
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            chain = prompt | self.model | StrOutputParser()
            return RunnableWithMessageHistory(
                chain,
                self.db_manager._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                system_message_key="system_prompt"
            )
        except Exception as e:
            print(f"[오류] 역할놀이 체인 생성 실패: {e}"); return None
        
    async def start(self, req: dict, profile_id: int):
        session_id = req['session_id']
        user_role = req['user_role']
        bot_role = req['bot_role']

        chatroom_id = await self.db_manager.create_new_chatroom(session_id, profile_id, 'roleplay')
        
        session_state = self.db_manager.store.setdefault(session_id, {})
        session_state['roleplay_state'] = {"user_role": user_role, "bot_role": bot_role}
        
        print(f"🎭 세션 [{session_id}] 역할놀이 시작: 사용자='{user_role}', 챗봇='{bot_role}'")
        
        response_text = f"좋아! 지금부터 너는 '{user_role}', 나는 '{bot_role}'이야. 역할에 맞춰 이야기해보자!"
        return response_text, chatroom_id

    async def talk(self, req: dict, profile_id: int):
        user_input = req['user_input']
        session_id = req['session_id']
        session_state = self.db_manager.store.get(session_id)

        if not session_state or not session_state.get('roleplay_state'):
            return "역할놀이가 시작되지 않았습니다. 먼저 역할놀이를 시작해주세요.", None

        chatroom_id = session_state.get('chatroom_id')
        if not chatroom_id:
             return "오류: 역할놀이 중인 채팅방을 찾을 수 없습니다.", None

        if not self.conversational_chain:
            return "챗봇 로직 초기화에 실패했습니다.", chatroom_id

        state = session_state['roleplay_state']
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
            print(f"[오류] 역할놀이 대화 생성 중 문제 발생: {e}")
            return "미안, 지금은 대답하기가 좀 힘들어.", chatroom_id

# --- 퀴즈 로직 클래스 ---
class QuizLogic:
    def __init__(self, model, db_manager):
        self.model = model
        self.db_manager = db_manager
        self.quiz_data = self._load_quiz_data('rag_data/quiz_data.txt')
        self.quiz_eval_chain = self._create_quiz_eval_chain()

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
    
    async def talk(self, req: dict, profile_id: int):
        user_input = req['user_input']
        session_id = req['session_id']
        session_state = self.db_manager.store.setdefault(session_id, {})
        quiz_state = session_state.get('quiz_state')
        
        if not quiz_state:
            if not self.quiz_data: return "미안, 지금은 퀴즈를 낼 수 없어.", None
            chatroom_id = await self.db_manager.create_new_chatroom(session_id, profile_id, 'quiz')
            
            quiz = random.choice(self.quiz_data)
            session_state['quiz_state'] = {'quiz_item': quiz, 'attempts': 0}
            return f"좋아, 재미있는 안전 퀴즈 시간! \n\n{quiz['question']}", chatroom_id

        current_quiz = quiz_state['quiz_item']
        eval_result_text = await self.quiz_eval_chain.ainvoke({"answer": current_quiz['answer'], "user_input": user_input})
        is_correct = "[판단: 참]" in eval_result_text
        
        if is_correct:
            response = f"딩동댕! 정답이야! 정답은 바로... **{current_quiz['answer']}**\n\n정말 똑똑한걸? 또 퀴즈 풀고 싶으면 '퀴즈'라고 말해줘!"
            await self.db_manager.summarize_and_close_room(session_id)
            return response, None
        else:
            quiz_state['attempts'] += 1
            if quiz_state['attempts'] < 2:
                return f"음... 조금 더 생각해볼까? 힌트는 '{current_quiz['hint']}'이야. 다시 한번 생각해볼래?", session_state['chatroom_id']
            else:
                response = f"아쉽다! 정답은 '{current_quiz['answer']}'이었어. 괜찮아, 이렇게 하나씩 배우는 거지! 다음엔 꼭 맞힐 수 있을 거야."
                await self.db_manager.summarize_and_close_room(session_id)
                return response, None

# --- 일상 대화 로직 클래스 ---
class ConversationLogic:
    def __init__(self, model, db_manager):
        self.model = model
        self.db_manager = db_manager
        self.instruct = "당신은 아이들의 눈높이에 맞춰 대화하는 다정한 AI 친구 '꾸로'입니다. 항상 친절하고 상냥하게 대답해주세요."
        self.topic_check_chain = self._create_topic_check_chain()
        self.rag_chain = self._setup_rag_and_history()

    def _create_topic_check_chain(self):
        try:
            prompt = ChatPromptTemplate.from_template("""당신은 두 문장 사이의 주제 연속성을 판단하는 AI입니다.
주어진 '대화 마지막 부분'과 '새로운 입력'을 비교하여, '새로운 입력'이 완전히 새로운 주제를 시작한다면 'NEW_TOPIC'이라고만 답하세요.
만약 '새로운 입력'이 이전 대화의 흐름을 자연스럽게 이어간다면 'CONTINUE'라고만 답하세요.
[대화 마지막 부분]:
{history}
---
[새로운 입력]:
{input}
---
[판단]:""")
            return prompt | self.model | StrOutputParser()
        except Exception as e:
            print(f"[오류] 주제 분석 체인 생성 중 문제 발생: {e}"); return None

    def _setup_rag_and_history(self):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            documents = TextLoader("rag_data/info.txt", encoding='utf-8').load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            retriever = Chroma.from_documents(docs, embeddings).as_retriever()
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{self.instruct}\n\n[참고할 만한 정보]\n{{context}}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            rag_chain_main = (RunnablePassthrough.assign(context=lambda x: retriever.get_relevant_documents(x["input"]))| prompt| self.model| StrOutputParser())
            return RunnableWithMessageHistory(rag_chain_main, self.db_manager._get_session_history, input_messages_key="input", history_messages_key="chat_history")
        except Exception as e:
            print(f"[오류] RAG 또는 체인 설정 중 심각한 문제 발생: {e}"); return None

    async def talk(self, req: dict, profile_id: int):
        user_input = req['user_input']
        session_id = req['session_id']
        session_state = self.db_manager.store.setdefault(session_id, {})
        current_chatroom_id = session_state.get('chatroom_id')
        history = session_state.get('history')

        if not current_chatroom_id:
            current_chatroom_id = await self.db_manager.create_new_chatroom(session_id, profile_id, 'conversation')
        elif history and history.messages:
            history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages[-4:]])
            topic_check_result = await self.topic_check_chain.ainvoke({"history": history_str, "input": user_input})
            if "NEW_TOPIC" in topic_check_result:
                await self.db_manager.summarize_and_close_room(session_id)
                current_chatroom_id = await self.db_manager.create_new_chatroom(session_id, profile_id, 'conversation')
        
        if not current_chatroom_id:
            return "채팅방을 만들거나 찾는 데 문제가 발생했어요.", None

        if not self.rag_chain:
            return "챗봇 로직 초기화에 실패했습니다.", current_chatroom_id
            
        try:
            response = await self.rag_chain.ainvoke(
                {"input": user_input},
                config={'configurable': {'session_id': session_id}}
            )
            return response, current_chatroom_id
        except Exception as e:
            print(f"[오류] 일상 대화 생성 중 문제 발생: {e}")
            return "미안, 지금은 대답하기가 좀 힘들어.", current_chatroom_id

# --- 관계 조언 로직 클래스 ---
class RelationshipAdvisor:
    def __init__(self, model):
        self.model = model

    def _fetch_today_conversations(self, profile_id: int):
        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            today = date.today()
            cursor.execute("""
                SELECT role, content 
                FROM talk 
                WHERE profile_id = %s AND DATE(created_at) = %s
                AND category = 'LIFESTYLEHABIT'
                ORDER BY created_at ASC
            """, (profile_id, today))
            return cursor.fetchall()
        except Error as e:
            print(f"[DB 오류] 대화 내용 조회 실패: {e}")
            return []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    async def generate_advice(self, req: dict):
        profile_id = req['profile_id']
        conversations = self._fetch_today_conversations(profile_id)
        
        if not conversations:
            return {"profile_id": profile_id, "advice": "오늘의 대화 내용이 없어 조언을 생성할 수 없습니다."}

        child_talks = [f"- {content}" for role, content in conversations if role == 'user']
        if not child_talks:
            return {"profile_id": profile_id, "advice": "오늘 아이의 대화 내용이 없어 조언을 생성할 수 없습니다."}
            
        conversation_log = "\n".join(child_talks)

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
        
        chain = prompt | self.model | StrOutputParser()
        result = await chain.ainvoke({"conversation_log": conversation_log})
        
        return {"profile_id": profile_id, "advice": result}

# --- 메인 시스템 ---
class ChatbotSystem:
    def __init__(self, model_name='timhan/llama3korean8b4qkm'):
        print("🤖 챗봇 시스템 로딩 시작...")
        self.model = ChatOllama(model=model_name)
        self.db_manager = DatabaseManager(self.model)
        self.conversation_logic = ConversationLogic(self.model, self.db_manager)
        self.roleplay_logic = RolePlayLogic(self.model, self.db_manager)
        self.quiz_logic = QuizLogic(self.model, self.db_manager)
        self.relationship_advisor = RelationshipAdvisor(self.model) # 조언 모듈 추가
        print("✅ 챗봇 시스템이 정상적으로 로드되었습니다.")

app = FastAPI(title="꾸로 API (모듈 분리 버전)")
chatbot_system = ChatbotSystem()

class ChatRequest(BaseModel):
    user_input: str
    session_id: str
    profile_id: int = 1

class RolePlayRequest(ChatRequest):
    user_role: str
    bot_role: str

class AdviceRequest(BaseModel):
    profile_id: int

class EndRequest(BaseModel):
    session_id: str
    profile_id: int = 1

@app.post("/conversation/talk", summary="일상 대화")
async def handle_conversation(req: ChatRequest, background_tasks: BackgroundTasks):
    response_text, chatroom_id = await chatbot_system.conversation_logic.talk(req.dict(), req.profile_id)
    if chatroom_id:
        background_tasks.add_task(chatbot_system.db_manager.save_conversation_to_db, req.session_id, req.user_input, response_text, chatroom_id, req.profile_id)
    return {"response": response_text}

@app.post("/roleplay/start", summary="역할놀이 시작")
async def start_roleplay(req: RolePlayRequest, background_tasks: BackgroundTasks):
    response_text, chatroom_id = await chatbot_system.roleplay_logic.start(req.dict(), req.profile_id)
    if chatroom_id:
        background_tasks.add_task(chatbot_system.db_manager.save_conversation_to_db, req.session_id, req.user_input, response_text, chatroom_id, req.profile_id)
    return {"response": response_text}

@app.post("/roleplay/talk", summary="역할놀이 대화")
async def handle_roleplay(req: ChatRequest, background_tasks: BackgroundTasks):
    response_text, chatroom_id = await chatbot_system.roleplay_logic.talk(req.dict(), req.profile_id)
    if chatroom_id:
        background_tasks.add_task(chatbot_system.db_manager.save_conversation_to_db, req.session_id, req.user_input, response_text, chatroom_id, req.profile_id)
    return {"response": response_text}

@app.post("/quiz/talk", summary="퀴즈 시작 및 답변")
async def handle_quiz(req: ChatRequest, background_tasks: BackgroundTasks):
    response_text, chatroom_id = await chatbot_system.quiz_logic.talk(req.dict(), req.profile_id)
    if chatroom_id:
        background_tasks.add_task(chatbot_system.db_manager.save_conversation_to_db, req.session_id, req.user_input, response_text, chatroom_id, req.profile_id)
    return {"response": response_text}

@app.post("/conversation/end", summary="대화 종료 및 요약")
async def end_conversation(req: EndRequest):
    await chatbot_system.db_manager.summarize_and_close_room(req.session_id)
    return {"message": "대화가 종료되고 요약되었습니다."}

@app.post("/relationship-advice", summary="관계 조언 생성")
async def get_relationship_advice(req: AdviceRequest):
    return await chatbot_system.relationship_advisor.generate_advice(req.dict())

if __name__ == "__main__":
    print("🚀 FastAPI 서버를 시작합니다. http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

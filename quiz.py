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

# --- 챗봇 로직 클래스 (최종 수정 버전) ---
class ChatbotLogic:
    def __init__(self, model_name='timhan/llama3korean8b4qkm'):
        print("키즈케어 로봇 '꾸로' 로딩 시작...")
        self.model = ChatOllama(model=model_name)
        self.store = {}
        self.instruct = self._get_instruct()
        
        # [추가] 퀴즈 및 대화 종료 키워드 정의
        self.QUIZ_START_KEYWORDS = ["퀴즈", "문제", "게임"]
        self.END_KEYWORDS = ["끝", "종료", "그만", "대화 종료"]
        
        # [추가] 퀴즈 상태 관리
        self.quiz_mode = {}

        # [추가] 분석용 체인들 초기화
        self.analysis_chain = self._create_analysis_chain()
        self.summarization_chain = self._create_summarization_chain()
        self.topic_check_chain = self._create_topic_check_chain()
        self.quiz_eval_chain = self._create_quiz_eval_chain()

        # [추가] 퀴즈 데이터 로드
        self.quiz_data = self._load_quiz_data('rag_data/quiz_data.txt')
        
        self.rag_chain = self._setup_rag_and_history()

        # [DB 추가] 데이터베이스 테이블 준비
        self._ensure_table_exists()

        if all([self.rag_chain, self.analysis_chain, self.summarization_chain, self.topic_check_chain, self.quiz_eval_chain]):
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
            
    def _create_topic_check_chain(self):
        """새로운 입력이 기존 대화 주제와 다른지 확인하는 체인을 생성합니다."""
        try:
            prompt = ChatPromptTemplate.from_template("""당신은 두 문장 사이의 주제 연속성을 판단하는 AI입니다.
주어진 '대화 마지막 부분'과 '새로운 입력'을 비교하여, '새로운 입력'이 완전히 새로운 주제를 시작한다면 'NEW_TOPIC'이라고만 답하세요.
만약 '새로운 입력'이 이전 대화의 흐름을 자연스럽게 이어간다면 'CONTINUE'라고만 답하세요.
인물, 장소, 특정 사물 이름이 이어지면 같은 주제입니다.

---
[예시 1]
대화 마지막 부분: 오늘 강아지랑 산책했는데 정말 좋아하더라.
새로운 입력: 저녁 메뉴 추천해줘.
판단: NEW_TOPIC
---
[예시 2]
대화 마지막 부분: 오늘 강아지랑 산책했는데 정말 좋아하더라.
새로운 입력: 우리 강아지는 어떤 간식을 제일 좋아해?
판단: CONTINUE
---
[예시 3]
대화 마지막 부분: 어제 본 영화 진짜 재밌었어. 주인공 연기가 대박이야.
새로운 입력: 오늘 날씨 어때?
판단: NEW_TOPIC
---

이제 아래 내용을 판단하세요.

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

    def _load_quiz_data(self, file_path):
        """퀴즈 데이터를 파일에서 불러옵니다."""
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

    def _setup_rag_and_history(self):
        """대화 기록(History)과 정보 검색(RAG)을 결합한 체인을 설정합니다."""
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            documents = TextLoader("rag_data/info.txt", encoding='utf-8').load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            retriever = Chroma.from_documents(docs, embeddings).as_retriever()
            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}\n\n[참고할 만한 정보]\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            rag_chain_main = (RunnablePassthrough.assign(context=lambda x: retriever.get_relevant_documents(x["input"]))| prompt| self.model| StrOutputParser())
            return RunnableWithMessageHistory(rag_chain_main, self._get_session_history, input_messages_key="input", history_messages_key="chat_history")
        except Exception as e:
            print(f"[오류] RAG 또는 체인 설정 중 심각한 문제 발생: {e}"); return None

    def _get_session_history(self, session_id: str):
        """세션별 대화 기록 객체를 관리합니다."""
        if session_id not in self.store:
            self.store[session_id] = {'history': InMemoryChatMessageHistory(), 'chatroom_id': None, 'type': 'conversation'}
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
        
        conn = self._create_db_connection()
        if conn is None: return

        try:
            with conn.cursor() as cursor:
                summary = ""
                room_type = session_state.get('type', 'conversation')

                if room_type == 'quiz':
                    if session_id in self.quiz_mode and 'quiz_item' in self.quiz_mode[session_id]:
                        quiz_question = self.quiz_mode[session_id]['quiz_item']['question']
                        summary = f"[퀴즈] {quiz_question}"
                    else:
                        summary = "[퀴즈] 퀴즈 대화"
                elif history.messages:
                    full_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages])
                    summary_text = await self.summarization_chain.ainvoke({"history": full_history_str})
                    # [수정] 불필요한 따옴표 변환 로직 제거
                    summary = f"[일상대화] {summary_text.strip()}"

                if summary:
                    cursor.execute("UPDATE chatroom SET topic = %s WHERE id = %s", (summary, current_chatroom_id))
                    conn.commit()
                    print(f"채팅방({current_chatroom_id}) 요약 완료 및 저장: {summary}")
        except Error as e:
            print(f"[DB 오류] 채팅방 요약 중 오류: {e}"); conn.rollback()
        finally:
            if conn: conn.close()

        session_state['chatroom_id'] = None
        history.clear()
        if session_id in self.quiz_mode:
            del self.quiz_mode[session_id]
        print(f"세션({session_id})의 채팅방이 종료되었습니다.")

    async def _create_new_chatroom(self, session_id: str, profile_id: int):
        """새로운 채팅방을 생성하고 세션에 ID를 할당합니다."""
        session_state = self.store.setdefault(session_id, {'history': InMemoryChatMessageHistory(), 'chatroom_id': None, 'type': 'conversation'})
        conn = self._create_db_connection()
        if conn is None: return None
        try:
            with conn.cursor() as cursor:
                room_type = session_state.get('type', 'conversation')
                topic = "새로운 퀴즈" if room_type == 'quiz' else "새로운 대화"
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

    async def _manage_chatroom(self, session_id: str, user_input: str, profile_id: int):
        """채팅방을 관리(생성, 주제 변경 감지)하는 핵심 로직"""
        session_state = self.store.setdefault(session_id, {'history': InMemoryChatMessageHistory(), 'chatroom_id': None, 'type': 'conversation'})
        current_chatroom_id = session_state.get('chatroom_id')
        history = session_state['history']
        
        if not current_chatroom_id:
            return await self._create_new_chatroom(session_id, profile_id)

        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages[-4:]])
        if not history_str: return current_chatroom_id

        topic_check_result = await self.topic_check_chain.ainvoke({"history": history_str, "input": user_input})
        print(f"주제 분석 결과: {topic_check_result}")
        
        if "NEW_TOPIC" in topic_check_result:
            print("주제 변경 감지됨. 이전 채팅방 요약 및 새 채팅방 생성 시작")
            await self._summarize_and_close_room(session_id)
            return await self._create_new_chatroom(session_id, profile_id)
        else:
            return current_chatroom_id

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
                category = 'SAFETYSTUDY' if self.store[session_id]['type'] == 'quiz' else 'LIFESTYLEHABIT'
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

    async def handle_quiz(self, user_input: str, session_id: str, profile_id: int):
        """퀴즈 관련 로직을 처리합니다."""
        # [수정] KeyError 방지를 위해 세션 상태를 먼저 초기화합니다.
        session_state = self.store.setdefault(session_id, {'history': InMemoryChatMessageHistory(), 'chatroom_id': None, 'type': 'conversation'})
        quiz_state = self.quiz_mode.get(session_id)
        
        # 퀴즈 시작
        if not quiz_state:
            if not self.quiz_data: return "미안, 지금은 퀴즈를 낼 수 없어.", None
            await self._summarize_and_close_room(session_id)
            session_state['type'] = 'quiz'
            chatroom_id = await self._create_new_chatroom(session_id, profile_id)
            
            quiz = random.choice(self.quiz_data)
            self.quiz_mode[session_id] = {'quiz_item': quiz, 'attempts': 0}
            return f"좋아, 재미있는 안전 퀴즈 시간! \n\n{quiz['question']}", chatroom_id

        # 퀴즈 진행
        current_quiz = quiz_state['quiz_item']
        eval_result_text = await self.quiz_eval_chain.ainvoke({"answer": current_quiz['answer'], "user_input": user_input})
        is_correct = "[판단: 참]" in eval_result_text
        
        if is_correct:
            response = f"딩동댕! 정답이야! 정답은 바로... **{current_quiz['answer']}**\n\n정말 똑똑한걸? 또 퀴즈 풀고 싶으면 '퀴즈'라고 말해줘!"
            await self._summarize_and_close_room(session_id)
            return response, None
        else:
            quiz_state['attempts'] += 1
            if quiz_state['attempts'] < 2:
                return f"음... 조금 더 생각해볼까? 힌트는 '{current_quiz['hint']}'이야. 다시 한번 생각해볼래?", self.store[session_id]['chatroom_id']
            else:
                response = f"아쉽다! 정답은 '{current_quiz['answer']}'이었어. 괜찮아, 이렇게 하나씩 배우는 거지! 다음엔 꼭 맞힐 수 있을 거야."
                await self._summarize_and_close_room(session_id)
                return response, None

    async def invoke(self, user_input: str, session_id: str, profile_id: int):
        """챗봇의 메인 실행 함수."""
        
        # 퀴즈 시작 또는 진행 확인
        is_quiz_request = any(keyword in user_input for keyword in self.QUIZ_START_KEYWORDS)
        if is_quiz_request or session_id in self.quiz_mode:
            return await self.handle_quiz(user_input, session_id, profile_id)

        # 일반 대화 종료 확인
        if any(keyword in user_input for keyword in self.END_KEYWORDS):
            await self._summarize_and_close_room(session_id)
            return "알겠습니다. 대화를 종료하고 요약했습니다. 새로운 대화를 시작할 수 있습니다.", None

        # 일반 대화 진행
        self.store.setdefault(session_id, {'history': InMemoryChatMessageHistory(), 'chatroom_id': None, 'type': 'conversation'})['type'] = 'conversation'
        chatroom_id = await self._manage_chatroom(session_id, user_input, profile_id)
        if not chatroom_id:
            return "채팅방을 만들거나 찾는 데 문제가 발생했어요.", None

        if not self.rag_chain:
            return "챗봇 로직 초기화에 실패했습니다.", chatroom_id
            
        try:
            # system_prompt를 invoke에 직접 전달하도록 변경
            response = await self.rag_chain.ainvoke(
                {"input": user_input, "system_prompt": self.instruct},
                config={'configurable': {'session_id': session_id}}
            )
            return response, chatroom_id
        except Exception as e:
            print(f"[오류] 대화 생성 중 문제 발생: {e}")
            return "미안, 지금은 대답하기가 좀 힘들어.", chatroom_id

# --- FastAPI 서버 설정 ---
app = FastAPI(title="꾸로 API (채팅방 자동 관리 및 퀴즈)")

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
    
    is_command = any(keyword in request.user_input for keyword in chatbot_logic.END_KEYWORDS + chatbot_logic.QUIZ_START_KEYWORDS)

    if chatroom_id and not is_command:
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

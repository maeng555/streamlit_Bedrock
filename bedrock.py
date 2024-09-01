import re
import boto3
import streamlit as st
import pandas as pd
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import plotly.graph_objects as go
import base64
import json
import os
import random

# Bedrock 클라이언트 설정
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Image generation settings
def generate_image(prompt):
    seed = random.randint(0, 2147483647)
    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": prompt},
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "cfgScale": 7.5,
            "height": 512,
            "width": 512,
            "seed": seed,
        },
    })
    try:
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId="amazon.titan-image-generator-v1",
        )
        base64_image_data = json.loads(response["body"].read())["images"][0]
        return base64.b64decode(base64_image_data)
    except Exception as e:
        print(f"Error generating image: {e}")
        raise

def save_image(base64_image_data, prompt):
    """Save base64 image data to a file."""
    output_folder = "images"
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, f"{prompt}.png")
    with open(file_path, "wb") as file:
        file.write(base64_image_data)
    return file_path

# Claude 3.5 파라미터 설정
model_kwargs = {
    "max_tokens": 1000,
    "temperature": 0.01,
    "top_p": 0.01,
}

# Bedrock LLM 설정
llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs=model_kwargs,
    streaming=True
)

st.set_page_config(
    page_title="Titleturtle",
    page_icon=':memo:',
    initial_sidebar_state="expanded",
    layout="wide",
)

# Streamlit 채팅 메시지 히스토리 설정
message_history = StreamlitChatMessageHistory(key="chat_messages")

# 프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="message_history"),
        ("human", "{query}"),
    ]
)

# 대화 체인 설정
chain_with_history = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: message_history,  # 항상 이전 대화를 리턴
    input_messages_key="query",
    history_messages_key="message_history",
)

# CSV 파일 로드
df = pd.read_csv('university_major.csv', encoding='utf-8')  # 인코딩 형식은 상황에 맞게 설정

# 사이드바 설정
with st.sidebar:
    # 학교 선택
    school = st.selectbox('학교 선택:', ('상명대학교서울', '상명대학교천안'))

    # 학과 리스트 초기화
    majors = []

    # 첫 번째 selectbox의 결과에 따라 두 번째 selectbox의 옵션 결정
    if school == '상명대학교서울':
        df_seoul = df[df['시도명'] == '서울특별시']
        df_seoul = df_seoul[df_seoul['시군구명'] == '종로구']
        majors = df_seoul['학과명'].unique()
    elif school == '상명대학교천안':
        df_cheonan = df[df['시도명'] == '충청남도']
        df_cheonan = df_cheonan[df_cheonan['시군구명'] == '천안시']
        majors = df_cheonan['학과명'].unique()

    # 학과 선택
    major = st.selectbox('학과 선택:', sorted(majors))

    # Form to input details
    with st.form(key='sidebar_form'):
        st.write('')
        # 과제 문제 입력
        assignment_view = st.text_area('과제 관련 내용 입력:', height=200)

        # 과제 문제 입력
        assignment_q = st.text_area('과제 문제 입력:', height=200)

        # 과제 내용 입력
        assignment_content = st.text_area('과제 내용 입력:', height=200)

        # 과제 종류 선택
        catecate = st.selectbox('과제 종류 선택:', ('보고서', '문제풀이', '코드작성'))

        st.write('')

        # 관대함과 창의성 중심 슬라이더
        lenient_weight = st.slider('관대함', 0, 100, 50, 1)
        creativity_weight = st.slider('창의성 중심', 0, 100, 50, 1)

        st.write('')

        # 체크박스들
        checkbox_options = {
            "표절률 검사 가능여부": False
        }

        checkbox_selections = {}
        for label in checkbox_options.keys():
            checkbox_selections[label] = st.checkbox(label)

        st.write('')  # 추가적인 여백

        # 제출 버튼
        submit_button = st.form_submit_button(label='적용')

        # 제출 버튼이 눌렸을 때만 세션 스테이트에 값을 저장
        if submit_button:
            st.session_state['school'] = school
            st.session_state['major'] = major
            st.session_state['catecate'] = catecate
            st.session_state['lenient_weight'] = lenient_weight
            st.session_state['creativity_weight'] = creativity_weight
            st.session_state['checkbox_selections'] = checkbox_selections
            st.session_state['assignment_content'] = assignment_content
            st.session_state['assignment_q'] = assignment_q
            st.session_state['assignment_view'] = assignment_view

            # 선택된 학과의 주요 교과목명 저장
            if major and school in ['상명대학교서울', '상명대학교천안']:
                if school == '상명대학교서울':
                    df_filtered = df_seoul[df_seoul['학과명'] == major]
                else:
                    df_filtered = df_cheonan[df_cheonan['학과명'] == major]

                # 주요 교과목명 추출 및 분리
                if not df_filtered.empty:
                    major_subjects = df_filtered['주요교과목명'].values
                    if len(major_subjects) > 0 and not pd.isna(major_subjects[0]):
                        subject_list = major_subjects[0].split('+')
                        st.session_state['major_subjects'] = subject_list
                    else:
                        st.session_state['major_subjects'] = ["해당 학과에 대한 주요 교과목 정보가 없습니다."]
                else:
                    st.session_state['major_subjects'] = ["해당 학과에 대한 정보가 없습니다."]
            else:
                st.session_state['major_subjects'] = ["해당 학과에 대한 정보가 없습니다."]

            # Generate and save image
            image_prompt = (
                f"a professor"
                f"a professor whose major is a {major}"
                f"A cartoon professor whose anger level is {100 - lenient_weight}, visually depicted with matching intensity."
                f"a professor as cartoon"
            )


            image_data = generate_image(image_prompt)
            image_path = save_image(image_data, image_prompt)
            st.session_state['image_path'] = image_path

            # 평가 요소 제공을 위한 프롬프트 생성
            evaluation_prompt = (
                f"사용자는 {st.session_state.get('school', '학교')} {st.session_state.get('major', '학과')}에서 다음 과목들을 학습한 학생입니다. "
                f"참고 자료: {st.session_state.get('assignment_view', '없음')}. "
                f"피드백 받을 자료의 종류: {st.session_state.get('catecate', '없음')}. "
                f"피드백 요청 자료: '{st.session_state.get('assignment_q', '없음')}'. "
                "이 자료를 평가하기 위해 고려해야 할 요소를 6가지 제시해 주세요. 딱 요소만 표시해줘, 다음과 같습니다 그런거 빼고 무조건 1. 뭐뭐 2. 뭐뭐 이렇게 6개만"
            )

            # 평가 요소 제공
            response_stream = chain_with_history.stream({"query": evaluation_prompt}, config={"configurable": {"session_id": "any"}})
            evaluation_elements = ""
            for chunk in response_stream:
                evaluation_elements += chunk.content

            # 디버깅을 위한 평가 요소 출력
            st.write("Evaluation Elements Response:")
            st.write(evaluation_elements)

            evaluation_elements_list = [e.strip() for e in evaluation_elements.split('\n') if e.strip()]
            st.session_state['evaluation_elements_list'] = evaluation_elements_list

            # 6가지 평가 요소별 점수와 피드백 받기
            feedback_prompt = (
                f"{major}교수로서 자기소개를 맨 처음에 넣어줘\n\n"
                f"성격 항목은 총 2가지가 있습니다. 가중치는 각각 0~100까지 있어\n\n"
                f"100에 가까울수록 창의성을 중점으로 본다, 관대하다\n\n"
                f"0에 가까울수록 창의성이 중요하지 않다, 깐깐하다\n\n"
                f"이런 특징을 가진다.\n\n"
                f"점수를 평가하는 요소는 {', '.join(evaluation_elements_list)} 이야\n"
                f"총점은 각각의 점수 100점씩해서 600점이야\n\n"
                f"너는 {st.session_state.get('major', '과목')} 교수이고\n"
                f"각 성격 항목에 대해 순서대로 {st.session_state.get('creativity_weight', 0)},{st.session_state.get('lenient_weight', 0)}의 가중치를 가지고 있어\n\n"
                f"이건 내가 제출한 과제야\n"
                f"{st.session_state.get('assignment_content', '과제 내용이 없습니다.')}\n\n"
                f"평가 항목에 따라 점수를 알려주고 피드백해줘"
            )

            # 평가 요소 및 점수를 저장할 변수 초기화
            evaluation_scores = {}
            feedback_responses = {}

            # Claude로부터 피드백을 받고 점수를 추출
            response_stream = chain_with_history.stream({"query": feedback_prompt}, config={"configurable": {"session_id": "any"}})
            feedback_response = ""
            for chunk in response_stream:
                feedback_response += chunk.content

            # 디버깅을 위한 피드백 응답 출력
            st.write("Feedback Response:")
            st.write(feedback_response)

            # 평가 요소별 점수 추출
            for element in evaluation_elements_list:
                # 정규식을 이용해 각 평가 요소에 대한 점수 추출
                score_match = re.search(rf"{re.escape(element)}\s*:\s*(\d+)", feedback_response)
                if score_match:
                    evaluation_scores[element] = int(score_match.group(1))
                else:
                    evaluation_scores[element] = "점수 없음"

                # 평가 요소별 피드백 추출
                feedback_match = re.search(rf"{re.escape(element)}\s*:\s*(.*?)(?=\n\d+|$)", feedback_response, re.DOTALL)
                if feedback_match:
                    feedback_responses[element] = feedback_match.group(1).strip()
                else:
                    feedback_responses[element] = "피드백 없음"

            # 평가 요소별 점수를 세션 스테이트에 저장
            st.session_state['evaluation_scores'] = evaluation_scores
            st.session_state['feedback_responses'] = feedback_responses

# 메인 화면 설정
if 'evaluation_elements_list' in st.session_state and 'evaluation_scores' in st.session_state:
    # 레이아웃을 1:3:1로 나누기
    col1, col2 = st.columns([3, 2])

    with col1:
        if 'image_path' in st.session_state:
            st.image(st.session_state['image_path'], caption='Generated Image', use_column_width=False, width=380)

        selected_elements = st.multiselect(
            "평가 받고 싶은 요소를 선택해 주세요:",
            options=st.session_state.get('evaluation_elements_list', [])
        )
        with st.form(key='feedback_form'):
            feedback_button = st.form_submit_button(label='피드백 요청')

            if feedback_button:
                selected_elements_str = ', '.join(selected_elements)

                feedback_prompt = (
                    f"{major}교수로서 자기소개를 맨 처음에 넣어줘\n\n"
                    f"성격 항목은 총 2가지가 있습니다. 가중치는 각각 0~100까지 있어\n\n"
                    f"100에 가까울수록 창의성을 중점으로 본다, 관대하다\n\n"
                    f"0에 가까울수록 창의성이 중요하지 않다, 깐깐하다\n\n"
                    f"이런 특징을 가진다.\n\n"
                    f"점수를 평가하는 요소는 {selected_elements_str} 이야\n"
                    f"총점은 100점이야\n\n"
                    f"너는 {st.session_state.get('major', '과목')} 교수이고\n"
                    f"각 성격 항목에 대해 순서대로 {st.session_state.get('creativity_weight', 0)},{st.session_state.get('lenient_weight', 0)}의 가중치를 가지고 있어\n\n"
                    f"이건 내가 제출한 과제야\n"
                    f"{st.session_state.get('assignment_content', '과제 내용이 없습니다.')}\n\n"
                    f"평가 항목에 따라 점수를 알려주고 피드백해줘"
                )

                # 피드백 요청
                response_stream = chain_with_history.stream({"query": feedback_prompt}, config={"configurable": {"session_id": "any"}})
                st.chat_message("ai").write_stream(response_stream)

    with col2:
        if st.session_state.get('evaluation_scores'):
            labels = st.session_state['evaluation_elements_list']
            scores = [st.session_state['evaluation_scores'].get(label, 0) for label in labels]

            # 점수가 정상적으로 설정되었는지 확인
            if len(scores) > 0 and len(scores) == len(labels):
                fig = go.Figure()

                # 점수를 레이더 차트에 맞게 추가
                fig.add_trace(go.Scatterpolar(
                    r=scores + [scores[0]],  # 레이더 차트를 닫기 위해 첫 점을 추가
                    theta=labels + [labels[0]],  # 레이더 차트를 닫기 위해 첫 레이블을 추가
                    fill='toself'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=False
                )

                st.plotly_chart(fig)
            else:
                st.write("점수 또는 레이블 데이터가 올바르지 않습니다.")

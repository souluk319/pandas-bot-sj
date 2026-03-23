import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# 라이브러리 불러오기
import gradio as gr
from PIL import Image
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Agent 생성
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY
)

def anlayze_with_langchain(df, question):
    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="openai-tools", # OpenAI Tools 에이전트 사용
        verbose=True, # 로그 출력
        return_intermediate_steps=True,
        allow_dangerous_code=True, # 위험한 코드 실행 허용 (필수)
        )
    
    response = agent_executor.invoke(question)
    text_output = response['output']

    # 중간 단계에서 Python 코드 추출
    intermediate_output = []

    # response['intermediate_steps']에서 Python 코드 추출
    # 에러가 발생할 경우를 대비하여 try-except 블록 사용
    try:
        for item in response['intermediate_steps']:
            if item[0].tool == 'python_repl_ast':
                intermediate_output.append(str(item[0].tool_input['query']))
    except:
        pass

    # Python 코드가 있는 경우, 이를 하나의 문자열로 결합
    python_code = "\n".join(intermediate_output)

    # Python 코드가 없거나 비어있는 경우 None으로 설정
    try:
        exec(python_code)
        if("plt" not in python_code) & ("fig" not in python_code) & ("plot" not in python_code) & ("sns" not in python_code):
            python_code = None 
    except:
        python_code = None


    return text_output, python_code

def excute_and_show_chart(python_code, df):

    try:
        # 코드 실행 환경 준비 및 코드 실행
        locals = {"df": df.copy()}
        exec(python_code, globals(), locals)

        # 자료 이미지로 변환
        fig = plt.figure()
        exec(python_code, globals(), locals)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
    
    # 에러 발생 시 예외 처리
    except Exception as e:
        print(f"Error executing Python code: {e}")
        # 에러 발생 시 None 반환    
        return None


# Gradio를 사용하여 간단한 인터페이스 만들기
def process_and_display(csv_file, question):

    # CSV 파일을 읽어 DataFrame으로 변환
    df = pd.read_csv(csv_file.name)

    # LangChain을 사용하여 질문에 대한 분석 수행
    text_output, python_code = anlayze_with_langchain(df, question)

    # 분석 결과를 Markdown 형식으로 출력
    chart_image = excute_and_show_chart(python_code, df) if python_code else None


    return text_output, chart_image

with gr.Blocks() as demo:
    gr.Markdown("### CSV 파일을 업로드 하고, 질문을 입력하세요. 분석 결과를 보여줍니다.")
    with gr.Row():
        csv_input = gr.File(label="CSV 파일 업로드", type="filepath")
        question_input = gr.Textbox(placeholder="질문을 입력하세요")
        submit_button = gr.Button("질문 제출")

    output_markdown = gr.Markdown("### 분석 결과")
    output_image = gr.Image(type="pil", label="분석 결과 이미지")

    submit_button.click(fn=process_and_display, 
              inputs=[csv_input, question_input], 
              outputs=[output_markdown, output_image])

demo.launch()

## 참고:
# - LangChain 공식 문서: https://python.langchain.com/en/latest/index.html
# - Gradio 공식 문서: https://gradio.app/get_started
# - OpenAI API 문서: https://platform.openai.com/docs/api-reference
# - Python 코드 실행 시 보안에 주의해야 합니다. 위험한 코드 실행을 허용하는 경우, 악의적인 코드가 실행될 수 있으므로 신뢰할 수 있는 데이터와 질문만 사용해야 합니다.
# - 에러 처리 및 예외 처리를 적절히 구현하여 사용자 경험을 향상시키는 것이 좋습니다.
# - LangChain의 에이전트는 다양한 도구와 통합할 수 있으므로, 필요에 따라 다른 도구를 추가하여 기능을 확장할 수 있습니다.
# - Gradio 인터페이스는 사용자 친화적이며, 다양한 입력 및 출력 형식을 지원하므로, 필요에 따라 인터페이스를 커스터마이징할 수 있습니다.
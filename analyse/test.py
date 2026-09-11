import os
from openai import OpenAI

try:
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
        api_key="sk-ws-H.PDYYLXM.WXsq.MEYCIQDZSbntFO9azyBUWX-TFpk5Bx6EOg28FwewexFa75SrvAIhAJYBsObWtgqbM7V_kBLOSXaFJc8kZX_UeuYn_w_ODiwJ",
        # 以下为华北2（北京）地域的URL，各地域的URL不同。调用时请将{WorkspaceId}替换为真实的业务空间ID。
        base_url="https://llm-aza2t3uz447721ir.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",#llm-aza2t3uz447721ir.cn-beijing.maas.aliyuncs.com
    )

    completion = client.chat.completions.create(
        model="qwen3.8-max",  # 模型列表: https://help.aliyun.com/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你是谁？'}
        ]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")

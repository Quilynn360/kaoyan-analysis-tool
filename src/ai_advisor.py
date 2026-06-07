"""AI 择校/备考内参生成器 — 基于 DeepSeek API"""

from __future__ import annotations

from openai import OpenAI


def generate_ai_insights_from_markdown(
    markdown_content: str,
    api_key: str = "",
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
) -> str:
    """通过 DeepSeek API 将量化报告 Markdown 转化为针对考生的深度内参内容.

    Parameters
    ----------
    markdown_content : str
        高校量化分析报告的完整 Markdown 文本.
    api_key : str
        DeepSeek API Key. 优先从函数参数传入，也支持从 Streamlit secrets 读取.
    base_url : str
        API 端点地址，默认 https://api.deepseek.com.
    model : str
        模型名称，默认 deepseek-chat.

    Returns
    -------
    str
        AI 生成的 Markdown 内参文本.
    """
    # 尝试从 Streamlit secrets 读取（云端部署场景）
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        except Exception:
            pass

    if not api_key:
        return (
            "⚠️ 未检测到有效的 DEEPSEEK_API_KEY。\n\n"
            "请按以下方式配置：\n"
            "1. **本地运行**：在终端执行 `export DEEPSEEK_API_KEY=sk-xxx` 或创建 `.env` 文件\n"
            "2. **Streamlit Cloud**：在 Settings → Secrets 中添加 `DEEPSEEK_API_KEY`"
        )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        system_prompt = (
            "你是一位深耕马克思主义理论（马理论）考研十年的顶尖教研专家、AI择校规划师。\n"
            "你的任务是阅读输入的某所高校的量化分析 Markdown 报告，为考生输出一份【大咖视角的保姆级报考内参】。\n"
            "要求：\n"
            "1. 语言要极具鼓舞性、专业、一针见血，多用考研党喜欢的结构化排版（加粗、列表、Emoji）。\n"
            "2. 内容必须符合报告要求。\n"
            "3. 直接输出符合上述结构的 Markdown 文本，不要包含任何多余的寒暄和废话。\n"
            "输出结构示例：\n"
            "---\n"
            "### 🏫 院校报考内参：{学校名称}\n\n"
            "**📊 核心量化定位**\n"
            "- 竞争梯队：...\n\n"
            "**🎯 各科备考狙击指南**\n"
            "- 政治：...\n\n"
            "**⚠️ 风险提示**\n"
            "- ...\n"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"请针对以下高校量化分析报告进行全景解读，"
                        f"输出一份大咖视角的保姆级报考内参：\n\n{markdown_content}"
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=2000,
            stream=False,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ DeepSeek API 调用失败: {str(e)}"

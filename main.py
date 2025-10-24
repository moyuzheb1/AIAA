from langchain_openai import ChatOpenAI  
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  
from langchain_core.output_parsers import StrOutputParser  
import os  # 新增：导入os模块用于读取环境变量

# -------------------------- 1. 初始化核心组件--------------------------
# 模型初始化（保持原连接配置，适配通用任务）
model = ChatOpenAI(  
    base_url="https://openrouter.ai/api/v1",  
    api_key=os.getenv("OPENROUTER_API_KEY"),  # 从环境变量获取密钥
    model="deepseek/deepseek-r1-0528-qwen3-8b:free",  
)  

# 输出解析器：统一将模型响应转为纯文本（通用场景适配）
parser = StrOutputParser()  

# -------------------------- 2. 初始化上下文存储（核心：记忆历史对话）--------------------------
# 用列表存储完整对话流，初始系统指令定义通用Agent角色和规则
# 格式：[SystemMessage(角色设定), HumanMessage(用户输入1), AIMessage(Agent回复1), ...]
conversation_history = [
    SystemMessage(content="""
    你是一个通用智能助手，需严格遵循以下规则处理用户需求：
    1. 能应对多种场景（如问答、信息整理、任务建议、逻辑分析等），不局限于单一任务；
    2. 必须参考对话历史上下文：若用户输入涉及前文提及的内容（如指代、省略信息），需先明确语境再回应，确保逻辑连贯；
    3. 回应需准确、简洁、符合用户需求场景（如用户问技术问题则专业，问日常问题则通俗），不添加无关内容；
    4. 若无法回答，直接说明“抱歉，我暂未掌握该领域的相关信息”，不编造内容。
    """)
]  

# -------------------------- 3. 通用Agent交互逻辑（支持上下文关联）--------------------------
def general_agent_interact(user_input: str) -> str:
    """
    通用Agent交互函数：
    1. 记录用户新输入到对话历史；
    2. 携带完整历史上下文调用模型，确保Agent“记得”之前的对话；
    3. 保存Agent回复到历史，供下一轮交互参考。
    """
    # 步骤1：将当前用户需求添加到对话历史
    conversation_history.append(HumanMessage(content=user_input))
    
    # 步骤2：调用模型（关键：传入全部历史，实现上下文记忆）
    raw_response = model.invoke(conversation_history)
    
    # 步骤3：解析模型响应为纯文本
    agent_response = parser.invoke(raw_response)
    
    # 步骤4：保存Agent回复到历史，为下一轮交互铺垫
    conversation_history.append(AIMessage(content=agent_response))
    
    return agent_response

# -------------------------- 4. 通用交互测试（支持多轮上下文对话）--------------------------
if __name__ == "__main__":
    print("=== 通用上下文Agent（输入'exit'退出）===")
    while True:
        # 获取用户通用需求（支持问答、任务描述等）
        user = input("\n请输入你的需求（如问题、任务描述等，输入'exit'退出）：")
        
        # 退出逻辑
        if user.lower() == "exit":
            print("Agent已退出，再见！")
            break
        
        # 空输入防护（避免无效模型调用）
        if not user.strip():
            print("提示：请输入有效的需求内容！")
            continue
        
        # 调用Agent并打印结果
        result = general_agent_interact(user)
        print(f"\nAgent回复：{result}")
        
        # （可选）查看当前对话历史（调试用，取消注释即可）
        # print("\n当前对话历史：")
        # for i, msg in enumerate(conversation_history):
        #     role = "系统" if isinstance(msg, SystemMessage) else "用户" if isinstance(msg, HumanMessage) else "Agent"
        #     print(f"  {i+1}. {role}：{msg.content[:50]}..." if len(msg.content) > 50 else f"  {i+1}. {role}：{msg.content}")
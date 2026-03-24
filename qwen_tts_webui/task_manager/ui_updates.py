"""队列 UI 更新函数 - 提供给 Gradio 调用的状态更新接口"""

from qwen_tts_webui.config_manager.shared import get_queue_manager
from qwen_tts_webui.task_manager.ui_renderer import generate_queue_html


def update_queue_display() -> str:
    """更新队列显示
    
    从队列管理器获取最新状态，并生成 HTML 显示
    
    Returns:
        str: 队列表格 HTML
    """
    try:
        queue_mgr = get_queue_manager()
        queue_status = queue_mgr.get_queue_status()
        
        # 生成 HTML
        html = generate_queue_html(queue_status)
        return html
        
    except Exception as e:
        # 如果出错，返回错误信息
        return f"""
        <div style="padding: 20px; text-align: center; color: #f44336;">
            <p>❌ 加载队列失败：{str(e)}</p>
            <p>请检查后端服务是否正常运行</p>
        </div>
        """

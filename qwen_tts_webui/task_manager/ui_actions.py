"""队列 UI 操作函数 - 处理用户的前端操作请求"""

from qwen_tts_webui.config_manager.shared import get_queue_manager
from qwen_tts_webui.task_manager.ui_updates import update_queue_display


def move_up_task_action(task_id: str) -> tuple[str, str]:
    """上移任务
    
    Args:
        task_id: 任务 ID
        
    Returns:
        tuple[str, str]: (更新后的队列表格 HTML, 操作消息)
    """
    try:
        queue_mgr = get_queue_manager()
        success, msg = queue_mgr.move_task_up(task_id)
        
        html = update_queue_display()
        if success:
            return html, f"✅ 任务 {task_id[:8]}... 已上移"
        else:
            return html, f"ℹ️ {msg}"
            
    except Exception as e:
        return f"""
        <div style="padding: 20px; text-align: center; color: #f44336;">
            <p>❌ 上移任务失败：{str(e)}</p>
        </div>
        """, f"❌ 异常：{str(e)}"


def move_down_task_action(task_id: str) -> tuple[str, str]:
    """下移任务
    
    Args:
        task_id: 任务 ID
        
    Returns:
        tuple[str, str]: (更新后的队列表格 HTML, 操作消息)
    """
    try:
        queue_mgr = get_queue_manager()
        success, msg = queue_mgr.move_task_down(task_id)
        
        html = update_queue_display()
        if success:
            return html, f"✅ 任务 {task_id[:8]}... 已下移"
        else:
            return html, f"ℹ️ {msg}"
            
    except Exception as e:
        return f"""
        <div style="padding: 20px; text-align: center; color: #f44336;">
            <p>❌ 下移任务失败：{str(e)}</p>
        </div>
        """, f"❌ 异常：{str(e)}"


def cancel_task_action(task_id: str) -> tuple[str, str]:
    """取消任务
    
    Args:
        task_id: 任务 ID
        
    Returns:
        tuple[str, str]: (更新后的队列表格 HTML, 操作消息)
    """
    try:
        queue_mgr = get_queue_manager()
        success, msg = queue_mgr.cancel_task(task_id)
        
        html = update_queue_display()
        if success:
            return html, f"✅ 任务 {task_id[:8]}... 已取消"
        else:
            return html, f"ℹ️ {msg}"
            
    except Exception as e:
        return f"""
        <div style="padding: 20px; text-align: center; color: #f44336;">
            <p>❌ 取消任务失败：{str(e)}</p>
        </div>
        """, f"❌ 异常：{str(e)}"

"""队列 UI 渲染器 - 生成队列表格的 HTML"""

from typing import Any
from qwen_tts_webui.task_manager.models import TaskStatus, QueueItemResponse


def generate_queue_html(queue_status: list[QueueItemResponse]) -> str:
    """生成队列表格的 HTML
    
    Args:
        queue_status: 队列状态列表
        
    Returns:
        str: HTML 表格字符串
    """
    if not queue_status:
        return """
        <div style="padding: 20px; text-align: center; color: #666;">
            <p>📭 当前队列为空</p>
            <p>请在"声音生成"、"声音设计"或"声音克隆"页面提交任务</p>
        </div>
        """
    
    # 表头
    html = """
    <style>
        .queue-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }
        .queue-table th, .queue-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .queue-table th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        .status-waiting { color: #ff9800; font-weight: bold; }
        .status-running { color: #2196f3; font-weight: bold; }
        .status-success { color: #4caf50; font-weight: bold; }
        .status-failed { color: #f44336; font-weight: bold; }
        .status-cancelled { color: #9e9e9e; font-weight: bold; }
        .action-btn {
            padding: 4px 8px;
            margin: 0 2px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .btn-move { background-color: #e3f2fd; color: #1976d2; }
        .btn-cancel { background-color: #ffebee; color: #c62828; }
        .btn-disabled { 
            background-color: #f5f5f5; 
            color: #bdbdbd; 
            cursor: not-allowed; 
        }
        .icon { margin-right: 4px; }
    </style>
    <script>
        function triggerAction(action, taskId) {
            // 设置隐藏 input 的值
            const input = document.querySelector('#task_id_input textarea');
            if (input) {
                input.value = action + ':' + taskId;
                // 触发 input 事件让 Gradio 感知到变化
                input.dispatchEvent(new Event('input', { bubbles: true }));
                // 触发隐藏的按钮点击
                const hiddenBtn = document.querySelector('#hidden_action_btn button');
                if (hiddenBtn) hiddenBtn.click();
            }
        }
    </script>
    <table class="queue-table">
        <thead>
            <tr>
                <th>任务 ID</th>
                <th>类型</th>
                <th>状态</th>
                <th>提交时间</th>
                <th>耗时</th>
                <th>详情/结果</th>
                <th>操作</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # 渲染每一行
    for item in queue_status:
        # status 已经是字符串，不需要 .value
        status_str = item.status if isinstance(item.status, str) else item.status.value
        status_class = f"status-{status_str.lower()}"
        status_icon = _get_status_icon(status_str)
        
        # 生成操作按钮
        actions_html = _generate_action_buttons(item)
        
        # 获取结果或错误信息
        # 如果有结果（音频路径），显示文件名；如果有错误，显示错误信息
        detail_text = ""
        detail_title = ""
        if item.status == TaskStatus.SUCCESS and item.result:
            from pathlib import Path
            result_path = Path(item.result)
            detail_text = f"✅ {result_path.name}"
            detail_title = item.result
        elif item.status == TaskStatus.FAILED and item.error_message:
            detail_text = f"❌ {item.error_message[:20]}..." if len(item.error_message) > 20 else f"❌ {item.error_message}"
            detail_title = item.error_message
        elif item.status == TaskStatus.RUNNING:
            detail_text = "⚡ 处理中..."
        elif item.status == TaskStatus.WAITING:
            detail_text = "⏳ 等待中"
        
        html += f"""
        <tr>
            <td><code>{item.task_id[:8]}...</code></td>
            <td>{_get_task_type_name(item.task_type)}</td>
            <td class="{status_class}">{status_icon} {status_str}</td>
            <td>{item.submit_time}</td>
            <td>{item.duration or '-'}</td>
            <td title="{detail_title}">{detail_text}</td>
            <td>{actions_html}</td>
        </tr>
        """
    
    html += """
        </tbody>
    </table>
    """
    
    return html


def _get_status_icon(status: str | TaskStatus) -> str:
    """获取状态对应的图标
    
    Args:
        status: 任务状态（字符串或枚举）
        
    Returns:
        str: 图标 emoji
    """
    # 如果是枚举，转换为字符串
    status_str = status if isinstance(status, str) else status.value
    
    icons = {
        "waiting": "⏳",
        "running": "▶️",
        "success": "✅",
        "failed": "❌",
        "cancelled": "🚫",
    }
    return icons.get(status_str, "•")


def _get_task_type_name(task_type: Any) -> str:
    """获取任务类型的显示名称
    
    Args:
        task_type: 任务类型枚举
        
    Returns:
        str: 中文名称
    """
    names = {
        "VOICE_GENERATION": "声音生成",
        "VOICE_DESIGN": "声音设计",
        "VOICE_CLONE": "声音克隆",
    }
    type_name = task_type if isinstance(task_type, str) else task_type.name
    return names.get(type_name, type_name)


def _generate_action_buttons(item: QueueItemResponse) -> str:
    """生成操作按钮 HTML
    
    Args:
        item: 队列项响应对象
        
    Returns:
        str: 按钮 HTML
    """
    # 只有等待中的任务可以操作
    status_str = item.status if isinstance(item.status, str) else item.status.value
    if status_str != "waiting":
        return '<span style="color: #999; font-size: 12px;">运行中/已完成</span>'
    
    # 使用 Gradio 的 click 事件触发机制
    # 通过设置隐藏的 textbox 值并触发按钮点击来调用后端函数
    buttons = []
    
    # 上移按钮
    if item.queue_position is not None and item.queue_position > 0:
        buttons.append(
            f'<button class="action-btn btn-move" onclick="triggerAction(\'move_up\', \'{item.task_id}\')">↑ 上移</button>'
        )
    else:
        buttons.append(
            '<button class="action-btn btn-disabled" disabled>↑ 上移</button>'
        )
    
    # 下移按钮
    if item.queue_position is not None and item.queue_position < 999:  # 实际长度在外部判断
        buttons.append(
            f'<button class="action-btn btn-move" onclick="triggerAction(\'move_down\', \'{item.task_id}\')">↓ 下移</button>'
        )
    else:
        buttons.append(
            '<button class="action-btn btn-disabled" disabled>↓ 下移</button>'
        )
    
    # 取消按钮
    buttons.append(
        f'<button class="action-btn btn-cancel" onclick="triggerAction(\'cancel\', \'{item.task_id}\')">✕ 取消</button>'
    )
    
    return " ".join(buttons)

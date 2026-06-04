"""简单测试 pygame 窗口"""

from src.adapters.screen import ScreenDisplayAdapter, ScreenWindow
from src.agent.action import display, render_pet_expression
import time

def test_screen():
    print("创建屏幕窗口...")
    window = ScreenWindow()
    window.start()
    
    print("创建适配器...")
    adapter = ScreenDisplayAdapter(hardware=window)
    
    print("测试 idle 状态...")
    adapter.execute(display("Idle mode", status="idle"))
    time.sleep(1)
    
    print("测试 listening 状态...")
    adapter.execute(render_pet_expression("listening"))
    adapter.execute(display("Listening...", status="listening"))
    time.sleep(1)
    
    print("测试 thinking 状态...")
    adapter.execute(render_pet_expression("thinking"))
    adapter.execute(display("Thinking...", status="thinking"))
    time.sleep(1)
    
    print("测试 focus 模式...")
    adapter.execute(display("Focus mode", status="focus_mode"))
    adapter.update_focus_timer(25 * 60, 25 * 60)  # 25分钟专注
    time.sleep(2)
    
    print("测试 speaking 状态...")
    adapter.execute(display("Hello! I'm your pet.", status="speaking"))
    adapter.execute(render_pet_expression("speaking"))
    time.sleep(2)
    
    print("测试完成！按任意键退出...")
    input()
    
    print("关闭窗口...")
    window.stop()

if __name__ == "__main__":
    try:
        test_screen()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

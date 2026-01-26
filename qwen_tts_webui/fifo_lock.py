"""先进先出锁实现"""

import collections
import threading


class FIFOLock:
    """先进先出锁，确保任务按顺序获取锁"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inner_lock = threading.Lock()
        self._pending_threads = collections.deque()

    def acquire(self, blocking: bool = True) -> bool:
        """获取锁

        Args:
            blocking (bool): 是否阻塞等待

        Returns:
            bool: 是否成功获取锁
        """
        with self._inner_lock:
            lock_acquired = self._lock.acquire(False)
            if lock_acquired:
                return True
            elif not blocking:
                return False

            release_event = threading.Event()
            self._pending_threads.append(release_event)

        release_event.wait()
        return self._lock.acquire()

    def release(self) -> None:
        """释放锁"""
        with self._inner_lock:
            if self._pending_threads:
                release_event = self._pending_threads.popleft()
                release_event.set()

            self._lock.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

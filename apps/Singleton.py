from threading import Lock, Thread

class Singleton(type):
  _instance = dict()
  # _lock: Lock = Lock()

  def __call__(cls, *args, **kargs):
    # with cls._lock:
      if cls not in cls._instance:
        instance = super().__call__(*args, **kargs)
        cls._instance[cls] = instance
      return cls._instance[cls]
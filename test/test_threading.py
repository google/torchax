import threading
import unittest
import torchax



def test_access_config_thread(reraise):
  env = torchax.default_env()

  def task():
    with reraise:
      print(torchax.default_env().param)

  threads = []
  for _ in range(5):
    thread = threading.Thread(target=task, args=())
    thread.start()
    threads.append(thread)

  for thread in threads:
    thread.join()

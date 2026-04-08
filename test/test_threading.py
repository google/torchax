# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import threading
import unittest

import torchax


class TestThreading(unittest.TestCase):
  def test_access_config_thread(self):
    torchax.default_env()

    def task():
      print(torchax.default_env().param)

    threads = []
    for _ in range(5):
      thread = threading.Thread(target=task, args=())
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

  def test_thread_safe_init(self):
    # Force a reset to simulate pristine state
    torchax._env = None

    def task():
      return torchax.default_env()

    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [executor.submit(task) for _ in range(32)]
      results = [f.result() for f in futures]

    # All threads should return the same environment object
    assert len(results) > 0
    lead = results[0]
    for r in results:
      self.assertIsNotNone(r)
      self.assertIs(r, lead)


if __name__ == "__main__":
  unittest.main()

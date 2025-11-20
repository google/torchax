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

import threading
import unittest

import torchax

class TestThreading(unittest.TestCase):

  def test_access_config_thread(reraise):
    torchax.default_env()

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

if __name__ == "__main__":
  unittest.main()

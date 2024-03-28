# -----------ARGS---------------------
from typing import Any

"""
  Chunk的配置文件
  data:表示具体的数据LIST格式
  path:表示数据存储路径 egg:/opt/app
  chunks:表示分块存储的数目
  
  ProxyConfig:
  1.save_steps 隔多少step存储一次
  2.要存储的数据总共有多少
"""


class ChunkArgs:
    def __init__(self):
        pass

    def __init__(self, path, filename, chunks):
        # print(type(path))
        assert (isinstance(path, str))
        assert (type(filename) == str)
        assert (type(chunks) == int)
        self.path = path
        self.name = filename
        self.chunks = chunks

        self.save_steps = 0
        self.data_len = 0

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def setProxyConfig(self, save_steps, data_len):
        assert (type(save_steps) == int)
        assert (type(data_len) == int)
        self.save_steps = save_steps
        self.data_len = data_len


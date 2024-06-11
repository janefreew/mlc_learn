"""环境配置"""
import sys
import os
from pathlib import Path


def set_tvm(tvm_root: str|Path):
    """配置 TVM 环境

    Args:
        tvm_root: TVM 项目所在根目录
    """
    tvm_root = Path(tvm_root)
    TVM_PATH = str(tvm_root/'python')
    VTA_PATH = str(tvm_root/'vta/python')
    # sys.path.extend([TVM_PATH, VTA_PATH])
    for path in [TVM_PATH, VTA_PATH]:
        if path not in sys.path:
            sys.path.extend([path])
    vta_path = str(tvm_root/"3rdparty/vta-hw")
    os.environ['VTA_HW_PATH'] = os.environ.get('VTA_HW_PATH', vta_path)
    os.environ['TVM_HOME'] = str(tvm_root)
    os.environ['PYTHONPATH'] = f"{TVM_PATH}:{VTA_PATH}" + ":${PYTHONPATH}"

def set_env(num, current_path='.'):
    '''
    num 表示相对于 current_path 的父级根目录级别
    '''
    import sys
    from pathlib import Path

    ROOT = Path(current_path).resolve().parents[num]
    sys.path.extend([str(ROOT/'src')]) # 设置 `tvm_book` 环境
    from tvm_book.config.env import set_tvm
    
    TVM_ROOT = os.getenv("TVM_HOME")
    # 设置 TVM 环境
    set_tvm(TVM_ROOT)
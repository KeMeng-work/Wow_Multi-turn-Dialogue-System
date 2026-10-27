# config/paths.py
"""
生产级路径配置管理:
统一管理项目中的所有文件路径
确保路径写法、依赖符合Linux系统移植
确保跨平台兼容性和团队协作一致性
"""

import os
from pathlib import Path
from typing import Dict, Any

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


class ProjectPaths:
    """项目路径管理器"""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self._ensure_directories()

    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        # 基础目录
        base_dirs = [
            self.raw_data,  # 源数据目录
            self.processed_data,  # 结果数据目录
            self.models,  # 本地模型目录
            self.logs,  # 日志目录
            self.outputs,  # 输出文件目录
            self.config,  # 配置文件目录
        ]
        # 模型子目录（确保路径存在）
        model_subdirs = [
            self.nomic_embed_model,
            self.embedding_model.parent,
            self.spacy_ner_model.parent,
            self.reranker_model.parent,
            self.translator_model.parent,
        ]
        for dir_path in base_dirs + model_subdirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def raw_data(self) -> Path:
        """原始数据目录 - 未经任何处理的原始文件"""
        return self.project_root / "data" / "raw"

    @property
    def processed_data(self) -> Path:
        """处理后的数据目录 - 经过预处理的数据"""
        return self.project_root / "data" / "processed"

    @property
    def processed_data_02(self) -> Path:
        """处理后的分块数据目录"""
        return self.processed_data / "chunks"

    @property
    def models(self) -> Path:
        """模型文件目录"""
        return self.project_root / "src" / "models"

    @property
    def logs(self) -> Path:
        """日志文件目录"""
        return self.project_root / "logs"

    @property
    def outputs(self) -> Path:
        """输出目录 - 最终生成的文件"""
        return self.project_root / "outputs"

    @property
    def config(self) -> Path:
        """配置文件目录"""
        return self.project_root / "config"

    # ======================== 数据文件 ========================

    @property
    def wow_train_data(self) -> Path:
        """WoW训练数据文件路径"""
        return self.raw_data / "train_100.json"

    # ======================== 模型文件 ========================

    @property
    def nomic_embed_model(self) -> Path:
        """nomic-embed-text 模型路径（在线推理，ONNX 格式，支持环境变量覆盖）"""
        env_path = os.getenv('NOMIC_EMBED_MODEL_PATH')
        if env_path:
            return Path(env_path)
        return self.models / "nomic_embed_model_int8"

    @property
    def embedding_model(self) -> Path:
        """all-mpnet-base-v2 路径（离线 chunking 脚本专用，支持环境变量覆盖）"""
        env_path = os.getenv('EMBEDDING_MODEL_PATH')
        if env_path:
            return Path(env_path)
        return self.models / "chunk_model" / "all-mpnet-base-v2"

    @property
    def spacy_ner_model(self) -> Path:
        """Spacy NER模型路径（支持环境变量覆盖）"""
        env_path = os.getenv('SPACY_NER_MODEL_PATH')
        if env_path:
            return Path(env_path)
        return self.models / "chunk_model" / "en_core_web_trf"

    @property
    def reranker_model(self) -> Path:
        """BGE重排器模型路径（支持环境变量覆盖）"""
        env_path = os.getenv('RERANKER_MODEL_PATH')
        if env_path:
            return Path(env_path)
        return self.models / "bge-reranker-v2-m3"

    @property
    def translator_model(self) -> Path:
        """
        翻译模型路径（支持环境变量覆盖）。

        兼容两种目录结构：
        1. 服务器标准格式（推荐）：translate_model/ 下直接含 config.json
        2. HuggingFace 缓存格式（本地 Windows 下载）：translate_model/snapshots/<hash>/
           自动探测并返回 snapshot 子目录，使 from_pretrained() 和 ONNX 导出正常工作。
        """
        env_path = os.getenv('TRANSLATOR_MODEL_PATH')
        if env_path:
            return Path(env_path)
        return self.models / "translate_model_int8"

    # 工具方法
    def ensure_file_path(self, file_path: Path) -> bool:
        """确保文件路径的目录存在"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"创建目录失败: {e}")
            return False

    def validate_paths(self) -> Dict[str, Any]:
        """验证所有关键路径是否存在"""
        validation_results = {}

        critical_paths = {
            "project_root": self.project_root,
            "raw": self.raw_data,
            "nomic_embed_model_int8": self.nomic_embed_model,
            "reranker_model": self.reranker_model,
            "translator_model": self.translator_model,
            "processed": self.processed_data,
        }

        for name, path in critical_paths.items():
            validation_results[name] = {
                "path": str(path),
                "exists": path.exists(),
                "is_file": path.is_file() if path.exists() else False,
                "is_dir": path.is_dir() if path.exists() else False
            }

        return validation_results

    def get_relative_path(self, absolute_path: Path) -> str:
        """获取相对于项目根目录的相对路径"""
        try:
            return str(absolute_path.relative_to(self.project_root))
        except ValueError:
            return str(absolute_path)


# 全局路径实例
paths = ProjectPaths()

if __name__ == "__main__":
    # 路径验证和测试
    print("🔍 项目路径验证:")
    print(f"项目根目录: {paths.project_root}")

    validation = paths.validate_paths()
    for name, info in validation.items():
        status = "✅" if info["exists"] else "❌"
        print(f"{status} {name}: {info['path']}")

    # 确保所有目录存在
    paths._ensure_directories()
    print("\n📁 所有必要目录已确保存在")

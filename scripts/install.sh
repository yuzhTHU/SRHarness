# 将 SRAgent 代码下载到本地
# git clone git@github.com:yuzhTHU/MySRAgent.git ./SRAgent && cd SRAgent

# 创建环境
conda create -p ./venv python=3.12 -y && conda activate ./venv

# 安装其它依赖
pip install -e ".[dev]"
# pip install -e ".[all]" # 安装所有可选依赖，包括 torch, pysr 等安装起来比较复杂的库

# （可选）下载 LLMSR-Bench 数据到本地
# git clone git@hf.co:datasets/nnheui/llm-srbench ./data/llm-srbench-data

# （可选）下载其它代码以供参考
# git clone git@github.com:GAIR-NLP/SR-Scientist.git ./third-party/sr_scientist
# git clone git@github.com:deep-symbolic-mathematics/llm-srbench.git ./third-party/llm_srbench



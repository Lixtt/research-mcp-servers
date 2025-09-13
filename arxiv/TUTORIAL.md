# arXiv MCP 服务器教程

## 简介

arXiv MCP 服务器是一个基于 FastMCP 框架的工具，提供对 arXiv 论文数据库的完整访问功能。它支持论文搜索、下载、阅读和管理，是学术研究的强大助手。

## 功能特性

- 🔍 **智能搜索**: 支持关键词、作者、分类、日期范围等多种搜索方式
- 📥 **批量下载**: 支持单篇或批量下载论文 PDF
- 📖 **文本提取**: 从 PDF 中提取文本内容进行分析
- 📁 **本地管理**: 自动管理下载的论文文件
- 🔗 **版本支持**: 支持特定版本的论文下载

## 安装和配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. MCP 客户端配置

在 Claude Desktop 的配置文件中添加 arXiv 服务器配置：

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "python",
      "args": ["C:\\path\\to\\research-mcp-servers\\arxiv\\arxiv_server.py"],
      "env": {
        "PYTHONPATH": "C:\\path\\to\\research-mcp-servers",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

**注意**: 请将路径替换为您的实际项目路径。如果使用虚拟环境，将 `"command": "python"` 改为虚拟环境中 Python 的完整路径。

### 3. 启动服务器

```bash
# 标准模式（用于 MCP 客户端）
python arxiv_server.py

# HTTP 模式（用于 Web 应用）
python arxiv_server.py --transport http --port 8000

# 自定义端口和主机
python arxiv_server.py --transport http --host 0.0.0.0 --port 9000
```

### 4. 配置验证

#### 检查服务器状态

```bash
# 启动 HTTP 模式并检查健康状态
python arxiv_server.py --transport http --port 8000

# 在另一个终端中检查健康状态
curl http://localhost:8000/arxiv/health
```

#### 可用工具列表

配置成功后，MCP 客户端将可以使用以下工具：

1. **`arxiv_query`** - 搜索和查询 arXiv 论文
2. **`download_paper`** - 下载论文 PDF 文件
3. **`list_downloaded_papers`** - 列出本地已下载的论文
4. **`read_paper`** - 读取已下载论文的文本内容

#### 配置故障排除

**常见问题：**

1. **路径错误** - 确保使用正确的绝对路径
2. **Python 环境问题** - 使用完整路径到 Python 解释器
3. **权限问题** - 确保有执行权限
4. **依赖缺失** - 运行 `pip install -r requirements.txt` 安装所有依赖

## 核心工具使用

### 1. 论文搜索 (`arxiv_query`)

#### 基本搜索
```python
# 关键词搜索
papers = await arxiv_query("machine learning transformers")

# 作者搜索
papers = await arxiv_query("au:Hinton")

# 分类搜索
papers = await arxiv_query("cat:cs.AI")
```

#### 高级搜索语法
```python
# 复合查询
papers = await arxiv_query("cat:cs.AI AND au:Bengio AND submittedDate:[202301010000 TO 202312312359]")

# 标题搜索
papers = await arxiv_query("ti:transformer")

# 摘要搜索
papers = await arxiv_query("abs:deep learning")
```

#### 按 ID 查询
```python
# 单篇论文
papers = await arxiv_query(id_list="2401.12345")

# 多篇论文
papers = await arxiv_query(id_list=["2401.12345", "1909.03550", "2312.11805"])

# 特定版本
papers = await arxiv_query(id_list="1706.03762v1")
```

### 2. 论文下载 (`download_paper`)

#### 单篇下载
```python
# 下载最新版本
result = await download_paper("2401.12345")

# 下载特定版本
result = await download_paper("2401.12345v1")

# 指定保存目录
result = await download_paper("2401.12345", "/path/to/papers")
```

#### 批量下载
```python
# 批量下载
results = await download_paper(["2401.12345", "1909.03550", "2312.11805"])

# 批量下载到指定目录
results = await download_paper(["2401.12345", "1909.03550"], "/home/user/papers")
```

### 3. 本地论文管理 (`list_downloaded_papers`)

```python
# 列出所有下载的论文
papers = await list_downloaded_papers()

# 包含内容预览
papers = await list_downloaded_papers(include_content_preview=True)

# 指定目录
papers = await list_downloaded_papers("/path/to/papers")
```

### 4. 论文阅读 (`read_paper`)

```python
# 通过文件路径阅读
content = await read_paper(filepath="/path/to/paper.pdf")

# 通过 arXiv ID 阅读
content = await read_paper(paper_id="2401.12345")

# 限制页数和字符数
content = await read_paper(paper_id="2401.12345", max_pages=5, max_chars=50000)
```

## 搜索语法参考

### 基本语法
- `关键词`: 在标题、摘要、作者中搜索
- `au:作者名`: 搜索特定作者
- `ti:标题`: 在标题中搜索
- `abs:摘要`: 在摘要中搜索
- `cat:分类`: 搜索特定分类
- `id:论文ID`: 搜索特定论文

### 逻辑操作符
- `AND`: 逻辑与
- `OR`: 逻辑或
- `ANDNOT`: 逻辑非

### 日期范围
```python
# 2023年1月1日到2023年12月31日
"submittedDate:[202301010000 TO 202312312359]"

# 2024年1月1日之后
"submittedDate:[202401010000 TO *]"
```

### 分类代码
- `cs.AI`: 人工智能
- `cs.LG`: 机器学习
- `cs.CL`: 计算语言学
- `cs.CV`: 计算机视觉
- `math.NA`: 数值分析
- `physics.gen-ph`: 一般物理

## 实际应用示例

### 1. 研究趋势分析
```python
# 搜索最近一年的 AI 论文
recent_ai_papers = await arxiv_query(
    "cat:cs.AI AND submittedDate:[202301010000 TO 202312312359]",
    max_results=50,
    sortBy="submittedDate"
)
```

### 2. 特定作者跟踪
```python
# 跟踪特定作者的最新工作
author_papers = await arxiv_query(
    "au:Hinton AND submittedDate:[202301010000 TO *]",
    sortBy="submittedDate"
)
```

### 3. 批量下载和阅读
```python
# 下载相关论文
paper_ids = ["2401.12345", "1909.03550", "2312.11805"]
download_results = await download_paper(paper_ids)

# 阅读论文内容
for paper_id in paper_ids:
    content = await read_paper(paper_id=paper_id, max_pages=3)
    print(f"论文 {paper_id} 的前3页内容:")
    print(content[:1000] + "...")
```

## 最佳实践

### 1. 搜索优化
- 使用具体的分类代码而不是通用关键词
- 结合作者和分类进行精确搜索
- 利用日期范围限制搜索结果

### 2. 下载管理
- 定期清理不需要的论文文件
- 使用有意义的目录结构
- 批量下载时注意网络连接稳定性

### 3. 内容分析
- 先阅读摘要和引言部分
- 使用 `max_pages` 参数控制阅读量
- 结合 `list_downloaded_papers` 管理本地文件

## 故障排除

### 常见问题

1. **网络连接错误**
   - 检查网络连接
   - 确认 arXiv API 可访问性

2. **PDF 读取失败**
   - 确保安装了 PyMuPDF
   - 检查 PDF 文件完整性

3. **搜索结果为空**
   - 检查搜索语法
   - 尝试更宽泛的搜索条件

### 调试模式
```bash
# 启用详细日志
python arxiv_server.py --transport http --port 8000
```

## 扩展功能

### 自定义搜索
```python
# 创建自定义搜索函数
async def search_recent_papers(category, days=30):
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    date_range = f"submittedDate:[{start_date.strftime('%Y%m%d%H%M')} TO {end_date.strftime('%Y%m%d%H%M')}]"
    return await arxiv_query(f"cat:{category} AND {date_range}")
```

### 论文分析
```python
# 分析论文关键词
async def analyze_paper_keywords(paper_id):
    content = await read_paper(paper_id=paper_id, max_pages=2)
    # 在这里添加关键词提取逻辑
    return content
```

## 测试配置

配置完成后，可以通过以下方式测试：

1. **重启 Claude Desktop**
2. **检查工具可用性**：在 Claude 中询问 "有哪些可用的工具？"
3. **测试基本功能**：尝试搜索一篇论文

```bash
# 测试命令示例
python arxiv_server.py --transport http --port 8000
curl http://localhost:8000/arxiv/health
```

## 总结

arXiv MCP 服务器为学术研究提供了强大的工具集，通过合理的搜索策略和批量操作，可以大大提高研究效率。建议从简单的关键词搜索开始，逐步掌握高级搜索语法和批量操作技巧。

### 快速开始检查清单

- [ ] 安装依赖：`pip install -r requirements.txt`
- [ ] 配置 MCP 客户端（Claude Desktop 等）
- [ ] 测试服务器连接
- [ ] 尝试基本搜索功能
- [ ] 下载并阅读第一篇论文
- [ ] 探索高级搜索语法

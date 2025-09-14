# arXiv MCP 服务器教程

## 简介

arXiv MCP 服务器是一个基于 FastMCP 框架的工具，提供对 arXiv 论文数据库的完整访问功能。它支持论文搜索、下载、阅读和管理，是学术研究的强大助手。

## 功能特性

- 🔍 **智能搜索**: 支持关键词、作者、分类、日期范围等多种搜索方式
- 📥 **批量下载**: 支持单篇或批量下载论文 PDF
- 📖 **文本提取**: 从 PDF 中提取文本内容进行分析
- 🖼️ **图像提取**: 将 PDF 页面转换为图像，支持 AI 视觉分析
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
5. **`get_paper_resources`** - 生成 MCP 资源 URI，支持图像、元数据和文本资源

#### MCP 资源支持

服务器还提供 MCP 资源功能，支持以下资源类型：

- **图像资源**: `arxiv://{paper_id}/image/{page}?dpi={dpi}` - 论文页面图像
- **元数据资源**: `arxiv://{paper_id}/metadata` - 论文元数据信息
- **文本资源**: `arxiv://{paper_id}/text?pages={max_pages}` - 论文文本内容

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

### 5. MCP 资源生成 (`get_paper_resources`)

这是推荐的新方式，用于生成可在对话中直接引用的 MCP 资源 URI：

```python
# 生成图像资源
resources = await get_paper_resources(
    paper_id="2401.12345",
    resource_types=["image"],
    max_pages=3,
    dpi=150
)

# 生成所有类型的资源
resources = await get_paper_resources(
    paper_id="2401.12345",
    resource_types=["image", "metadata", "text"],
    max_pages=5,
    dpi=200
)
```

**返回格式**：
```python
{
    "images": [
        {
            "uri": "arxiv://2401.12345/image/1?dpi=150",
            "page_number": 1,
            "paper_id": "2401.12345",
            "dpi": 150,
            "resource_type": "image",
            "description": "Page 1 of arXiv paper 2401.12345 as image",
            "usage": "Reference this URI in conversation - MCP clients will show the image"
        },
        # ... 更多页面
    ],
    "metadata": {
        "uri": "arxiv://2401.12345/metadata",
        "paper_id": "2401.12345",
        "resource_type": "metadata",
        "description": "Complete metadata for arXiv paper 2401.12345",
        "usage": "Reference this URI to get paper title, authors, abstract, etc."
    },
    "text": {
        "uri": "arxiv://2401.12345/text?pages=3",
        "paper_id": "2401.12345",
        "max_pages": 3,
        "resource_type": "text",
        "description": "Text content of arXiv paper 2401.12345 (first 3 pages)",
        "usage": "Reference this URI to get the paper's text content"
    },
    "summary": "Generated 3 resource type(s) for paper 2401.12345",
    "usage_instructions": [
        "Copy any URI from the results and reference it directly in your conversation",
        "Example: 'Please analyze this image: arxiv://1706.03762/image/1'",
        "Compatible MCP clients will automatically load the referenced resources"
    ]
}
```

### 6. 在对话中使用 MCP 资源

生成资源后，可以直接在对话中引用这些 URI：

```
请分析这个论文的第一页：arxiv://2401.12345/image/1?dpi=150
```

```
显示这个论文的元数据：arxiv://2401.12345/metadata
```

```
获取这个论文的文本内容：arxiv://2401.12345/text?pages=3
```

当你在对话中引用这些 URI 时，兼容的 MCP 客户端（如 Cherry Studio）会自动加载相应的资源内容。

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

### 4. MCP 资源工作流（推荐）
```python
# 下载论文
paper_id = "1706.03762"  # Attention Is All You Need
await download_paper(paper_id)

# 生成 MCP 资源
resources = await get_paper_resources(
    paper_id=paper_id,
    resource_types=["image", "metadata", "text"],
    max_pages=3,
    dpi=150
)

# 查看生成的资源
print("图像资源:")
for img in resources["images"]:
    print(f"  - {img['uri']} (页面 {img['page_number']})")

print("元数据资源:")
print(f"  - {resources['metadata']['uri']}")

print("文本资源:")
print(f"  - {resources['text']['uri']}")

# 在对话中直接引用这些 URI
# 例如：请分析这个论文的第一页：arxiv://1706.03762/image/1?dpi=150
```

### 5. 传统图像分析工作流
```python
# 下载论文并提取图像（传统方式）
paper_id = "1706.03762"  # Attention Is All You Need
await download_paper(paper_id)

# 提取前3页为图像
images = await read_paper_as_images(
    paper_id=paper_id,
    max_pages=3,
    dpi=150
)

# 查看图像信息
for img in images:
    print(f"页面 {img['page_number']}: {len(img['image_data'])} 字符的 base64 数据")
    print(f"分辨率: {img['dpi']} DPI, 格式: {img['format']}")

# 图像数据可以直接用于 AI 视觉分析
# 例如：发送给支持视觉的 AI 模型进行分析
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

### 4. MCP 资源使用优化
- **推荐使用 MCP 资源**：优先使用 `get_paper_resources` 生成可在对话中引用的 URI
- **资源类型选择**：根据需要选择 `image`、`metadata`、`text` 或组合使用
- **图像参数优化**：调整 DPI 参数平衡质量和文件大小（150-200 推荐）
- **页数控制**：限制 `max_pages` 避免生成过大的资源数据
- **对话集成**：生成的 URI 可以直接在对话中引用，MCP 客户端会自动加载内容

### 5. 图像提取优化（传统方式）
- 对于包含大量图表的论文，使用图像提取而非纯文本
- 调整 DPI 参数平衡质量和文件大小（150-200 推荐）
- 限制 `max_pages` 避免生成过大的 base64 数据
- 图像数据适合用于 AI 视觉分析和 OCR 识别

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

4. **图像提取失败**
   - 确保安装了 PyMuPDF
   - 检查 PDF 文件是否损坏
   - 降低 DPI 设置或减少页数

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

### MCP 资源分析功能（推荐）
```python
# 生成论文的 MCP 资源用于分析
async def generate_paper_resources_for_analysis(paper_id, max_pages=5):
    resources = await get_paper_resources(
        paper_id=paper_id,
        resource_types=["image", "metadata", "text"],
        max_pages=max_pages,
        dpi=200  # 高分辨率用于更好的识别
    )
    
    # 返回可用于对话中引用的资源 URI
    return {
        "paper_id": paper_id,
        "image_uris": [img["uri"] for img in resources["images"]],
        "metadata_uri": resources["metadata"]["uri"],
        "text_uri": resources["text"]["uri"],
        "usage_examples": [
            f"请分析这个论文的第一页：{resources['images'][0]['uri']}",
            f"显示论文元数据：{resources['metadata']['uri']}",
            f"获取论文文本：{resources['text']['uri']}"
        ]
    }

# 批量生成多个论文的资源
async def batch_generate_resources(paper_ids, max_pages=3):
    all_resources = {}
    for paper_id in paper_ids:
        try:
            resources = await get_paper_resources(
                paper_id=paper_id,
                resource_types=["image", "metadata"],
                max_pages=max_pages
            )
            all_resources[paper_id] = resources
        except Exception as e:
            print(f"Failed to generate resources for {paper_id}: {e}")
    return all_resources
```

### 传统图像分析功能
```python
# 提取论文图表和公式（传统方式）
async def extract_paper_visuals(paper_id, max_pages=5):
    images = await read_paper_as_images(
        paper_id=paper_id, 
        max_pages=max_pages,
        dpi=200  # 高分辨率用于更好的识别
    )
    
    # 返回可用于 AI 视觉分析的图像数据
    return [
        {
            "page": img["page_number"],
            "image_b64": img["image_data"],
            "size": len(img["image_data"])
        }
        for img in images
    ]

# 对比不同论文的视觉布局
async def compare_paper_layouts(paper_ids):
    layouts = {}
    for paper_id in paper_ids:
        images = await read_paper_as_images(paper_id=paper_id, max_pages=1)
        if images:
            layouts[paper_id] = images[0]["image_data"]
    return layouts
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

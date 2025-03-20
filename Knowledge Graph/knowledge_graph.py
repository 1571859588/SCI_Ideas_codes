import json
import re
import base64
from openai import OpenAI


class KnowledgeGraphBuilder:
    def __init__(self, graph_file="knowledge_graph.json"):
        self.graph_file = graph_file
        self.graph = self._load_graph()
        self.client = OpenAI(
            api_key="xxx",
            base_url="xxx",
        )

    def _load_graph(self):
        """加载或创建新的图数据"""
        try:
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "images": [],
                "entities": [],
                "relationships": []
            }

    def _save_graph(self):
        """保存图数据到文件"""
        with open(self.graph_file, 'w', encoding='utf-8') as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)

    def extract_images_from_markdown(self, content):
        """从Markdown内容中提取图片信息"""
        pattern = r'!\[(.*?)\]\((.*?)\)'
        return re.findall(pattern, content)

    def get_image_context(self, content, image_url):
        """获取图片的上下文信息"""
        pos = content.find(image_url)
        if pos == -1:
            return ""
        start = max(0, pos - 300)
        end = min(len(content), pos + 300)
        return content[start:end]

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image_with_qwen(self, image_url, context):
        """使用通义千问API分析图片"""
        # 将图片转换为base64
        base64_img = self.encode_image(image_url)

        # 构建提示词
        prompt = (
            f"请分析这张图片并以JSON格式返回以下信息：\n"
            f"1. 图片类型(流程图、界面截图、代码等)\n"
            f"2. 图片中的关键实体\n"
            f"3. 实体之间的关系\n"
            f"上下文信息：{context}\n"
            f"请严格按照以下JSON格式返回：\n"
            f"{{'image_type': '类型', 'entities': ['实体1', '实体2'], "
            f"'relationships': [['实体1', '关系', '实体2']]}}"
        )

        response = self.client.chat.completions.create(
            model="qwen-vl-max",
            messages=[{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt,
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                }],
            }]
        )

        content = response.choices[0].message.content
        if content[-1] == ".":
            content = content[:-1]

        try:
            # 尝试解析返回的JSON
            result = json.loads(content)
            # 转换relationships格式以匹配原代码
            result['relationships'] = [
                (r[0], r[1], r[2]) for r in result['relationships']
            ]
            return result
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            return {
                'image_type': 'Unknown',
                'entities': [],
                'relationships': []
            }

    def build_graph(self, data_path):
        """构建知识图谱"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            for knowledge in item['knowledge']:
                content = knowledge['content']
                images = self.extract_images_from_markdown(content)

                for caption, image_url in images:
                    # 添加图片节点
                    image_data = {
                        "url": image_url,
                        "caption": caption,
                        "type": None
                    }

                    # 获取上下文并分析图片
                    context = self.get_image_context(content, image_url)
                    analysis = self.analyze_image_with_qwen(image_url, context)

                    # 更新图片类型
                    image_data["type"] = analysis["image_type"]

                    # 添加实体
                    for entity in analysis["entities"]:
                        if entity not in self.graph["entities"]:
                            self.graph["entities"].append(entity)

                    # 添加关系
                    for subject, relation, obj in analysis["relationships"]:
                        self.graph["relationships"].append({
                            "subject": subject,
                            "relation": relation,
                            "object": obj,
                            "image_url": image_url
                        })

                    self.graph["images"].append(image_data)

        self._save_graph()

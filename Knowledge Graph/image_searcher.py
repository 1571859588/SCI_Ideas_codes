import json
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from spacy.matcher import Matcher
from matplotlib import font_manager
import requests
import time


class ImageSearcher:
    def __init__(self, graph_file="knowledge_graph.json"):
        self.graph_file = graph_file
        self.graph = self._load_graph()
        # 加载英文语言模型
        self.nlp = spacy.load("en_core_web_sm")

        # 尝试设置字体
        try:
            # 对于Linux系统
            plt.rcParams['font.family'] = 'DejaVu Sans'
        except:
            # 如果失败，使用默认字体
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial']  # 使用常见的Arial字体

    def _load_graph(self):
        """加载图数据"""
        with open(self.graph_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def call_r1(self, model_name, prompt):
        """调用API以提取三元组"""
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        url = "xxx"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "xxx"
        }
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                data = response.json()
                print("data", data)
                
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0]['message'].get('content', '')
                    reasoning = data['choices'][0]['message'].get('reasoning_content', '')
                    
                    if content == '非常抱歉，作为一个AI助手，我无法回答该问题，请您换个话题或者问题试试。':
                        print("API returned default error message, retrying...")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        return None, None
                    
                    return content, reasoning
                else:
                    print(f"Warning: Unexpected API response format: {data}")
                    return None, None
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Skipping this question.")
                    return None, None
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                print(f"Response text: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Skipping this question.")
                    return None, None

    def extract_triples(self, text):
        """使用API从文本中提取三元组"""
        prompt = f"请从以下文本中提取三元组：{text}"
        content, _ = self.call_r1("your_model_name", prompt)
        
        if content:
            try:
                triples = json.loads(content)
                return triples
            except json.JSONDecodeError:
                print("Failed to decode JSON from API response.")
                return []
        else:
            return []

    def search_images(self, query):
        """搜索相关图片"""
        query = query.lower()
        results = []
        matched_relationships = []

        # 1. 提取查询中的三元组
        query_triples = self.extract_triples(query)
        print(f"Extracted triples from query: {query_triples}")

        # 2. 搜索完整查询
        for relationship in self.graph["relationships"]:
            # 检查完整查询
            if (query in relationship["subject"].lower() or
                query in relationship["object"].lower() or
                    query in relationship["relation"].lower()):
                matched_relationships.append(relationship)

            # 检查三元组匹配
            for subj, rel, obj in query_triples:
                if (subj in relationship["subject"].lower() and
                    rel in relationship["relation"].lower() and
                        obj in relationship["object"].lower()):
                    if relationship not in matched_relationships:
                        matched_relationships.append(relationship)

        # 3. 获取相关图片
        for relationship in matched_relationships:
            for image in self.graph["images"]:
                if image["url"] == relationship["image_url"]:
                    if image not in results:
                        results.append(image)

        # 将提取的三元组添加到关系中
        for subj, rel, obj in query_triples:
            matched_relationships.append({
                "subject": subj,
                "relation": rel,
                "object": obj,
                "image_url": None  # 因为这些是从查询中提取的，不对应具体的图片
            })

        return results, matched_relationships

    def visualize_graph(self, relationships=None, title="Knowledge Graph"):
        """可视化知识图谱"""
        plt.clf()
        plt.figure(figsize=(12, 8), dpi=100)

        G = nx.DiGraph()

        if relationships is None:
            relationships = self.graph["relationships"]

        # 添加节点和边
        for rel in relationships:
            subject = str(rel["subject"])
            relation = str(rel["relation"])
            object_ = str(rel["object"])
            G.add_edge(subject, object_, label=relation)

        pos = nx.spring_layout(G, k=1.5, iterations=50)

        # 绘制节点
        nx.draw_networkx_nodes(G, pos,
                               node_color='lightblue',
                               node_size=3000,
                               alpha=0.7)

        # 绘制节点标签
        nx.draw_networkx_labels(G, pos,
                                font_size=10)

        # 绘制边
        nx.draw_networkx_edges(G, pos,
                               edge_color='gray',
                               arrows=True,
                               arrowsize=20)

        # 绘制边标签
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels=edge_labels,
                                     font_size=8)

        plt.title(title, fontsize=12)
        plt.axis('off')
        plt.tight_layout()

        return plt

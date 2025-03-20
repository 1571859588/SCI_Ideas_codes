from knowledge_graph import KnowledgeGraphBuilder
from image_searcher import ImageSearcher

try:
    # 构建知识图谱
    builder = KnowledgeGraphBuilder(graph_file="knowledge_graph.json")
    # builder.build_graph("knowledge_skipper.json")

    # 搜索图片
    searcher = ImageSearcher(graph_file="knowledge_graph.json")

    # 可视化完整知识图谱
    plt = searcher.visualize_graph(title="Complete Knowledge Graph")
    plt.savefig("complete_graph.png")
    plt.close()

    # 执行搜索
    query = "What can Skipper provide"
    results, matched_relationships = searcher.search_images(query)

    # 打印结果
    print(f"Found {len(results)} images for query: {query}")
    for result in results:
        print(f"\nFound image: {result['url']}")
        print(f"Caption: {result['caption']}")
        print(f"Type: {result['type']}")

    # 可视化搜索结果
    if matched_relationships:
        plt = searcher.visualize_graph(
            relationships=matched_relationships,
            title=f"Search Results for: {query}"
        )
        plt.savefig("search_results_graph.png")
        plt.close()

except Exception as e:
    print(f"Error: {e}")
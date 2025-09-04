#!/usr/bin/env python3
"""
RAG系统测试脚本
"""

from rag_system import RAGSystem
import json
from datetime import datetime

def main():
    print("🚀 RAG系统测试启动")
    print("="*60)
    
    # 初始化RAG系统
    rag = RAGSystem()
    
    try:
        # 加载模型和知识库
        rag.load_model()
        rag.initialize_knowledge_base()
        
        # 测试问题
        test_questions = [
            "什么是机器学习？",
            "深度学习和机器学习有什么区别？", 
            "Python为什么适合做数据科学？",
            "神经网络是如何工作的？",
            "自然语言处理有哪些应用？",
            "人工智能在医疗领域有什么应用？"
        ]
        
        results = []
        
        print("\n🧪 开始RAG测试...")
        print("="*60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 测试 {i}/{len(test_questions)}")
            print("-" * 40)
            
            result = rag.generate_rag_response(question)
            results.append(result)
            
            print(f"问题: {question}")
            print(f"回答: {result['answer']}")
            print(f"使用上下文: {'是' if result['context_used'] else '否'}")
            
        # 保存结果
        output_file = f"rag_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*60)
        print("✅ RAG测试完成！")
        print(f"📊 测试问题数: {len(test_questions)}")
        print(f"💾 结果保存至: {output_file}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ RAG测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

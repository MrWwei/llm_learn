#!/usr/bin/env python3
"""
RAG(检索增强生成)技术模块
适用于RTX 3060的轻量级实现
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

# 简单的向量化和检索实现（无需额外GPU显存）
class SimpleVectorStore:
    """简单的向量存储和检索系统"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.doc_ids = []
    
    def simple_embedding(self, text: str) -> List[float]:
        """简单的文本嵌入（基于TF-IDF思想）"""
        # 简单的词频统计
        words = text.lower().split()
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # 创建简单的特征向量
        common_words = [
            '机器学习', '深度学习', '神经网络', 'python', '人工智能', 'ai', 'ml', 
            '算法', '数据', '模型', '训练', '学习', '计算机', '技术', '编程',
            'machine', 'learning', 'deep', 'neural', 'network', 'artificial',
            'intelligence', 'algorithm', 'data', 'model', 'training', 'computer'
        ]
        
        embedding = []
        for word in common_words:
            embedding.append(text.lower().count(word))
        
        # 归一化
        total = sum(embedding) + 1e-8
        embedding = [x / total for x in embedding]
        
        return embedding
    
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """添加文档到向量存储"""
        embedding = self.simple_embedding(content)
        self.documents.append({
            'id': doc_id,
            'content': content,
            'metadata': metadata or {}
        })
        self.embeddings.append(embedding)
        self.doc_ids.append(doc_id)
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索最相关的文档"""
        query_embedding = self.simple_embedding(query)
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self.similarity(query_embedding, doc_embedding)
            similarities.append((sim, i))
        
        # 排序并返回top_k
        similarities.sort(reverse=True)
        
        results = []
        for sim_score, idx in similarities[:top_k]:
            if sim_score > 0.1:  # 相似度阈值
                doc = self.documents[idx].copy()
                doc['similarity'] = sim_score
                results.append(doc)
        
        return results

class KnowledgeBase:
    """知识库管理器"""
    
    def __init__(self):
        self.vector_store = SimpleVectorStore()
        self.knowledge_data = self._create_knowledge_base()
    
    def _create_knowledge_base(self) -> List[Dict]:
        """创建示例知识库"""
        knowledge_base = [
            {
                "id": "ml_basics",
                "title": "机器学习基础",
                "content": "机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习和改进。机器学习主要分为三类：监督学习、无监督学习和强化学习。监督学习使用标记数据训练模型，如分类和回归任务。无监督学习从未标记数据中发现隐藏模式，如聚类和降维。强化学习通过与环境交互来学习最优策略。",
                "category": "machine_learning",
                "keywords": ["机器学习", "监督学习", "无监督学习", "强化学习"]
            },
            {
                "id": "deep_learning",
                "title": "深度学习详解",
                "content": "深度学习是机器学习的子集，使用多层神经网络来模拟人脑的工作方式。深度神经网络包含输入层、多个隐藏层和输出层。每层包含多个神经元，通过权重连接。深度学习在图像识别、自然语言处理、语音识别等领域取得了突破性进展。常见的深度学习架构包括卷积神经网络(CNN)、循环神经网络(RNN)和Transformer。",
                "category": "deep_learning",
                "keywords": ["深度学习", "神经网络", "CNN", "RNN", "Transformer"]
            },
            {
                "id": "python_programming",
                "title": "Python编程语言",
                "content": "Python是一种高级、解释性编程语言，以其简洁的语法和强大的功能而闻名。Python在数据科学和机器学习领域特别受欢迎，拥有丰富的库生态系统。重要的Python库包括：NumPy用于数值计算，Pandas用于数据处理，Matplotlib用于数据可视化，Scikit-learn用于机器学习，TensorFlow和PyTorch用于深度学习。Python的简洁语法使得快速原型开发和算法实现变得容易。",
                "category": "programming",
                "keywords": ["Python", "NumPy", "Pandas", "Scikit-learn", "TensorFlow", "PyTorch"]
            },
            {
                "id": "neural_networks",
                "title": "神经网络工作原理",
                "content": "神经网络是由相互连接的节点（神经元）组成的计算模型，模拟生物神经系统的工作方式。每个神经元接收多个输入，对输入进行加权求和，加上偏置项，然后通过激活函数产生输出。常见的激活函数包括sigmoid、tanh、ReLU等。神经网络通过反向传播算法进行训练，该算法计算损失函数对每个权重的梯度，然后使用梯度下降法更新权重。",
                "category": "neural_networks",
                "keywords": ["神经网络", "激活函数", "反向传播", "梯度下降"]
            },
            {
                "id": "nlp_basics",
                "title": "自然语言处理基础",
                "content": "自然语言处理(NLP)是人工智能的一个分支，专注于计算机与人类语言的交互。NLP的主要任务包括文本分类、情感分析、命名实体识别、机器翻译、问答系统、文本摘要等。传统的NLP方法依赖于手工特征工程，而现代NLP主要基于深度学习模型。Transformer架构革命性地改进了NLP任务的性能，催生了BERT、GPT、T5等强大的预训练模型。",
                "category": "nlp",
                "keywords": ["自然语言处理", "NLP", "文本分类", "BERT", "GPT", "Transformer"]
            },
            {
                "id": "data_science",
                "title": "数据科学流程",
                "content": "数据科学是一个跨学科领域，结合了统计学、计算机科学和领域专业知识来从数据中提取有价值的见解。典型的数据科学流程包括：数据收集、数据清洗、探索性数据分析、特征工程、模型选择和训练、模型评估和部署。数据科学家需要掌握编程技能（Python/R）、统计知识、机器学习算法，以及业务理解能力。",
                "category": "data_science",
                "keywords": ["数据科学", "数据清洗", "特征工程", "模型评估"]
            },
            {
                "id": "ai_applications",
                "title": "人工智能应用领域",
                "content": "人工智能在各个领域都有广泛应用。在医疗领域，AI用于医学影像分析、药物发现和个性化治疗。在金融领域，AI用于风险评估、算法交易和欺诈检测。在交通领域，AI推动了自动驾驶汽车的发展。在教育领域，AI实现了个性化学习和智能辅导。在娱乐领域，AI用于推荐系统和内容生成。这些应用展示了AI技术的巨大潜力和社会影响。",
                "category": "applications",
                "keywords": ["人工智能应用", "医疗AI", "金融AI", "自动驾驶", "推荐系统"]
            }
        ]
        
        return knowledge_base
    
    def build_index(self):
        """构建知识库索引"""
        print("📚 构建知识库索引...")
        
        for item in self.knowledge_data:
            self.vector_store.add_document(
                doc_id=item["id"],
                content=f"{item['title']} {item['content']}",
                metadata={
                    "title": item["title"],
                    "category": item["category"],
                    "keywords": item["keywords"]
                }
            )
        
        print(f"✅ 知识库索引构建完成，包含 {len(self.knowledge_data)} 个文档")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索相关知识"""
        return self.vector_store.search(query, top_k)
    
    def save_knowledge_base(self, filepath: str = "./data/knowledge_base.json"):
        """保存知识库到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_data, f, ensure_ascii=False, indent=2)
        print(f"💾 知识库已保存到: {filepath}")

class RAGSystem:
    """RAG系统主类"""
    
    def __init__(self, model_path="./output/final_model", base_model="distilgpt2"):
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.knowledge_base = KnowledgeBase()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """加载模型"""
        print("📦 加载RAG模型...")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        if os.path.exists(self.model_path):
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            print("⚠️ 微调模型不存在，使用基础模型")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        self.model.eval()
        print("✅ 模型加载完成")
    
    def initialize_knowledge_base(self):
        """初始化知识库"""
        self.knowledge_base.build_index()
        self.knowledge_base.save_knowledge_base()
    
    def generate_rag_response(self, question: str, max_new_tokens: int = 300) -> Dict[str, Any]:
        """使用RAG生成回答"""
        print(f"🔍 RAG处理问题: {question}")
        
        # 1. 检索相关文档
        relevant_docs = self.knowledge_base.search(question, top_k=3)
        
        if relevant_docs:
            print(f"📋 检索到 {len(relevant_docs)} 个相关文档:")
            for i, doc in enumerate(relevant_docs):
                print(f"   {i+1}. {doc['metadata']['title']} (相似度: {doc['similarity']:.3f})")
        
        # 2. 构建增强提示
        if relevant_docs:
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"参考资料{i+1}: {doc['content'][:200]}...")
            
            context = "\n\n".join(context_parts)
            
            enhanced_prompt = f"""请基于以下参考资料回答问题：

{context}

问题: {question}

请结合参考资料提供准确、详细的回答:"""
        else:
            enhanced_prompt = f"问题: {question}\n\n请提供详细的回答:"
        
        # 3. 生成回答
        inputs = self.tokenizer(enhanced_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(enhanced_prompt):].strip()
        
        return {
            "question": question,
            "retrieved_docs": relevant_docs,
            "context_used": len(relevant_docs) > 0,
            "enhanced_prompt": enhanced_prompt,
            "answer": answer
        }

def create_rag_test_script():
    """创建RAG测试脚本"""
    rag_test_code = '''#!/usr/bin/env python3
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
        
        print("\\n🧪 开始RAG测试...")
        print("="*60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\\n📝 测试 {i}/{len(test_questions)}")
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
        
        print("\\n" + "="*60)
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
'''
    
    with open("test_rag.py", 'w', encoding='utf-8') as f:
        f.write(rag_test_code)
    
    print("✅ RAG测试脚本已创建: test_rag.py")

if __name__ == "__main__":
    # 创建并测试知识库
    print("🚀 初始化RAG系统...")
    
    kb = KnowledgeBase()
    kb.build_index()
    kb.save_knowledge_base()
    
    # 测试检索功能
    test_queries = ["什么是机器学习？", "Python编程", "深度学习"]
    
    for query in test_queries:
        print(f"\\n🔍 测试查询: {query}")
        results = kb.search(query, top_k=2)
        for result in results:
            print(f"   📄 {result['metadata']['title']} (相似度: {result['similarity']:.3f})")
    
    # 创建测试脚本
    create_rag_test_script()
    
    print("\\n✅ RAG系统初始化完成！")

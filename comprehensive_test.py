#!/usr/bin/env python3
"""
综合测试脚本 - 对比基础模型、提示词工程、RAG三种模式
"""

import os
import json
import torch
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rag_system import RAGSystem
from typing import Dict, List, Any

class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self, model_path="./output/final_model", base_model="distilgpt2"):
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.rag_system = None
        
    def load_model(self):
        """加载模型"""
        print("📦 加载模型...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if os.path.exists(self.model_path):
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            print("⚠️ 使用基础模型进行测试")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        self.model.eval()
        print("✅ 模型加载完成")
    
    def load_rag_system(self):
        """加载RAG系统"""
        print("🔍 初始化RAG系统...")
        self.rag_system = RAGSystem(self.model_path, self.base_model)
        self.rag_system.load_model()
        self.rag_system.initialize_knowledge_base()
        print("✅ RAG系统准备完成")
    
    def generate_basic_response(self, question: str) -> Dict[str, Any]:
        """基础模式生成回答"""
        start_time = time.time()
        
        prompt = f"Human: {question}\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip()
        
        generation_time = time.time() - start_time
        
        return {
            "mode": "基础模式",
            "question": question,
            "answer": answer,
            "generation_time": generation_time,
            "prompt_used": prompt
        }
    
    def generate_prompt_engineered_response(self, question: str) -> Dict[str, Any]:
        """提示词工程模式生成回答"""
        start_time = time.time()
        
        # 使用角色扮演 + 结构化输出的提示词工程
        enhanced_prompt = f"""你是一名资深的人工智能专家，拥有丰富的理论知识和实践经验。请按照以下结构详细回答问题：

【核心概念】
【详细解释】
【实际应用】
【总结】

问题: {question}

请基于您的专业知识提供准确、全面的回答:"""
        
        inputs = self.tokenizer(enhanced_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(enhanced_prompt):].strip()
        
        generation_time = time.time() - start_time
        
        return {
            "mode": "提示词工程",
            "question": question,
            "answer": answer,
            "generation_time": generation_time,
            "prompt_used": enhanced_prompt
        }
    
    def generate_rag_response(self, question: str) -> Dict[str, Any]:
        """RAG模式生成回答"""
        start_time = time.time()
        
        result = self.rag_system.generate_rag_response(question, max_new_tokens=300)
        
        generation_time = time.time() - start_time
        result["generation_time"] = generation_time
        result["mode"] = "RAG增强"
        
        return result
    
    def evaluate_all_modes(self, test_questions: List[str]) -> Dict[str, Any]:
        """评估所有模式"""
        results = {
            "basic_mode": [],
            "prompt_engineering": [],
            "rag_mode": [],
            "comparison": []
        }
        
        print("🧪 开始综合评估...")
        print("="*80)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 测试 {i}/{len(test_questions)}: {question}")
            print("-" * 60)
            
            # 基础模式
            print("🔸 基础模式...")
            basic_result = self.generate_basic_response(question)
            results["basic_mode"].append(basic_result)
            print(f"   回答: {basic_result['answer'][:100]}...")
            
            # 提示词工程模式
            print("🔸 提示词工程模式...")
            pe_result = self.generate_prompt_engineered_response(question)
            results["prompt_engineering"].append(pe_result)
            print(f"   回答: {pe_result['answer'][:100]}...")
            
            # RAG模式
            print("🔸 RAG增强模式...")
            rag_result = self.generate_rag_response(question)
            results["rag_mode"].append(rag_result)
            print(f"   回答: {rag_result['answer'][:100]}...")
            print(f"   使用上下文: {'是' if rag_result.get('context_used', False) else '否'}")
            
            # 对比分析
            comparison = {
                "question": question,
                "basic_length": len(basic_result['answer']),
                "pe_length": len(pe_result['answer']),
                "rag_length": len(rag_result['answer']),
                "basic_time": basic_result['generation_time'],
                "pe_time": pe_result['generation_time'],
                "rag_time": rag_result['generation_time'],
                "rag_used_context": rag_result.get('context_used', False)
            }
            results["comparison"].append(comparison)
            
            print(f"   ⏱️  生成时间 - 基础: {basic_result['generation_time']:.2f}s, "
                  f"提示工程: {pe_result['generation_time']:.2f}s, "
                  f"RAG: {rag_result['generation_time']:.2f}s")
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成评估报告"""
        report = []
        report.append("# 大模型SFT + 提示词工程 + RAG 综合评估报告")
        report.append(f"\n**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**模型**: {self.base_model}")
        report.append(f"**微调模型路径**: {self.model_path}")
        
        # 统计信息
        comparisons = results["comparison"]
        
        avg_basic_time = sum(c['basic_time'] for c in comparisons) / len(comparisons)
        avg_pe_time = sum(c['pe_time'] for c in comparisons) / len(comparisons)
        avg_rag_time = sum(c['rag_time'] for c in comparisons) / len(comparisons)
        
        avg_basic_length = sum(c['basic_length'] for c in comparisons) / len(comparisons)
        avg_pe_length = sum(c['pe_length'] for c in comparisons) / len(comparisons)
        avg_rag_length = sum(c['rag_length'] for c in comparisons) / len(comparisons)
        
        rag_context_usage = sum(1 for c in comparisons if c['rag_used_context']) / len(comparisons)
        
        report.append("\n## 📊 性能统计")
        report.append("\n### 生成时间对比")
        report.append(f"- 基础模式: {avg_basic_time:.2f}s")
        report.append(f"- 提示词工程: {avg_pe_time:.2f}s")
        report.append(f"- RAG增强: {avg_rag_time:.2f}s")
        
        report.append("\n### 回答长度对比")
        report.append(f"- 基础模式: {avg_basic_length:.0f} 字符")
        report.append(f"- 提示词工程: {avg_pe_length:.0f} 字符")
        report.append(f"- RAG增强: {avg_rag_length:.0f} 字符")
        
        report.append(f"\n### RAG上下文使用率: {rag_context_usage:.1%}")
        
        # 详细结果
        report.append("\n## 📝 详细测试结果")
        
        for i, (basic, pe, rag) in enumerate(zip(results["basic_mode"], 
                                                results["prompt_engineering"], 
                                                results["rag_mode"]), 1):
            report.append(f"\n### 测试 {i}: {basic['question']}")
            
            report.append("\n#### 基础模式")
            report.append(f"回答: {basic['answer']}")
            
            report.append("\n#### 提示词工程模式")
            report.append(f"回答: {pe['answer']}")
            
            report.append("\n#### RAG增强模式")
            report.append(f"回答: {rag['answer']}")
            if rag.get('context_used'):
                report.append("✅ 使用了知识库上下文")
            else:
                report.append("❌ 未使用知识库上下文")
        
        report.append("\n## 🎯 评估结论")
        report.append("\n### 优势对比")
        report.append("- **基础模式**: 速度最快，但回答可能不够准确")
        report.append("- **提示词工程**: 回答更结构化，质量有所提升")
        report.append("- **RAG增强**: 基于知识库，回答最准确可靠")
        
        report.append("\n### 推荐使用场景")
        report.append("- **快速问答**: 基础模式")
        report.append("- **结构化回答**: 提示词工程模式")
        report.append("- **准确性要求高**: RAG增强模式")
        
        return "\n".join(report)

def main():
    """主函数"""
    print("🚀 大模型SFT + 提示词工程 + RAG 综合评估")
    print("="*80)
    
    # 创建评估器
    evaluator = ComprehensiveEvaluator()
    
    try:
        # 加载组件
        evaluator.load_model()
        evaluator.load_rag_system()
        
        # 测试问题
        test_questions = [
            "什么是机器学习？",
            "深度学习和机器学习有什么区别？",
            "Python为什么适合做数据科学？",
            "神经网络是如何工作的？",
            "自然语言处理有哪些应用？"
        ]
        
        # 执行评估
        results = evaluator.evaluate_all_modes(test_questions)
        
        # 生成报告
        report = evaluator.generate_report(results)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON结果
        json_file = f"comprehensive_evaluation_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存Markdown报告
        report_file = f"evaluation_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n" + "="*80)
        print("✅ 综合评估完成！")
        print(f"📊 JSON结果: {json_file}")
        print(f"📄 评估报告: {report_file}")
        print("="*80)
        
        # 显示简要结果
        print("\n📋 评估摘要:")
        comparisons = results["comparison"]
        avg_times = {
            "基础模式": sum(c['basic_time'] for c in comparisons) / len(comparisons),
            "提示词工程": sum(c['pe_time'] for c in comparisons) / len(comparisons),
            "RAG增强": sum(c['rag_time'] for c in comparisons) / len(comparisons)
        }
        
        for mode, time_val in avg_times.items():
            print(f"  {mode}: 平均 {time_val:.2f}s/问题")
        
        rag_usage = sum(1 for c in comparisons if c['rag_used_context'])
        print(f"  RAG上下文使用: {rag_usage}/{len(comparisons)} 次")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

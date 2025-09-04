"""
评估工具函数
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_perplexity(loss: float) -> float:
    """计算困惑度"""
    return np.exp(loss)

def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
    """计算BLEU分数"""
    try:
        from sacrebleu import corpus_bleu
        bleu = corpus_bleu(predictions, [references])
        return bleu.score
    except ImportError:
        logger.warning("sacrebleu未安装，使用简单的BLEU计算")
        return simple_bleu_score(predictions, references)

def simple_bleu_score(predictions: List[str], references: List[str]) -> float:
    """简单的BLEU分数计算"""
    def get_ngrams(text, n):
        words = text.split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams
    
    total_score = 0
    for pred, ref in zip(predictions, references):
        # 计算1-gram到4-gram的精确度
        scores = []
        for n in range(1, 5):
            pred_ngrams = get_ngrams(pred.lower(), n)
            ref_ngrams = get_ngrams(ref.lower(), n)
            
            if not pred_ngrams:
                scores.append(0)
                continue
                
            ref_count = defaultdict(int)
            for ngram in ref_ngrams:
                ref_count[ngram] += 1
            
            match_count = 0
            for ngram in pred_ngrams:
                if ref_count[ngram] > 0:
                    match_count += 1
                    ref_count[ngram] -= 1
            
            precision = match_count / len(pred_ngrams)
            scores.append(precision)
        
        # 计算几何平均
        if all(score > 0 for score in scores):
            bleu = np.exp(np.mean(np.log(scores)))
        else:
            bleu = 0
        
        total_score += bleu
    
    return total_score / len(predictions) * 100

def calculate_rouge_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """计算ROUGE分数"""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        
        return {key: np.mean(values) * 100 for key, values in rouge_scores.items()}
    
    except ImportError:
        logger.warning("rouge_score未安装，使用简单的ROUGE计算")
        return simple_rouge_score(predictions, references)

def simple_rouge_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """简单的ROUGE分数计算"""
    def get_words(text):
        return text.lower().split()
    
    def rouge_n(pred_words, ref_words, n):
        pred_ngrams = [' '.join(pred_words[i:i+n]) for i in range(len(pred_words)-n+1)]
        ref_ngrams = [' '.join(ref_words[i:i+n]) for i in range(len(ref_words)-n+1)]
        
        if not ref_ngrams:
            return 0.0
        
        overlap = len(set(pred_ngrams) & set(ref_ngrams))
        return overlap / len(ref_ngrams)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_words = get_words(pred)
        ref_words = get_words(ref)
        
        # ROUGE-1
        rouge1_scores.append(rouge_n(pred_words, ref_words, 1))
        
        # ROUGE-2
        rouge2_scores.append(rouge_n(pred_words, ref_words, 2))
        
        # ROUGE-L (最长公共子序列)
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        lcs_len = lcs_length(pred_words, ref_words)
        if len(ref_words) > 0:
            rougeL_scores.append(lcs_len / len(ref_words))
        else:
            rougeL_scores.append(0.0)
    
    return {
        'rouge1': np.mean(rouge1_scores) * 100,
        'rouge2': np.mean(rouge2_scores) * 100,
        'rougeL': np.mean(rougeL_scores) * 100
    }

def calculate_meteor_score(predictions: List[str], references: List[str]) -> float:
    """计算METEOR分数"""
    try:
        from nltk.translate import meteor_score
        import nltk
        nltk.download('wordnet', quiet=True)
        
        scores = []
        for pred, ref in zip(predictions, references):
            score = meteor_score.meteor_score([ref.split()], pred.split())
            scores.append(score)
        
        return np.mean(scores) * 100
    
    except ImportError:
        logger.warning("nltk未安装，跳过METEOR分数计算")
        return 0.0

def evaluate_generation_quality(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """评估生成质量"""
    metrics = {}
    
    # BLEU分数
    metrics['bleu'] = calculate_bleu_score(predictions, references)
    
    # ROUGE分数
    rouge_scores = calculate_rouge_score(predictions, references)
    metrics.update(rouge_scores)
    
    # METEOR分数
    metrics['meteor'] = calculate_meteor_score(predictions, references)
    
    # 平均长度
    pred_lengths = [len(pred.split()) for pred in predictions]
    ref_lengths = [len(ref.split()) for ref in references]
    
    metrics['avg_pred_length'] = np.mean(pred_lengths)
    metrics['avg_ref_length'] = np.mean(ref_lengths)
    metrics['length_ratio'] = np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0
    
    return metrics

def evaluate_model_on_dataset(model, tokenizer, dataset, max_new_tokens: int = 512,
                            temperature: float = 0.7, top_p: float = 0.9) -> Dict[str, Any]:
    """在数据集上评估模型"""
    model.eval()
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for example in dataset:
            # 构建输入
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            reference = example.get('output', '')
            
            if input_text:
                prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:"
            else:
                prompt = f"### 指令:\n{instruction}\n\n### 回答:"
            
            # 生成回复
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            predictions.append(generated_text.strip())
            references.append(reference)
    
    # 计算评估指标
    metrics = evaluate_generation_quality(predictions, references)
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'references': references
    }

def calculate_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """计算文本多样性指标"""
    if not texts:
        return {}
    
    all_words = []
    all_bigrams = []
    all_trigrams = []
    
    for text in texts:
        words = text.lower().split()
        all_words.extend(words)
        
        # 计算bigrams和trigrams
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        all_bigrams.extend(bigrams)
        all_trigrams.extend(trigrams)
    
    # 计算distinct-n
    distinct_1 = len(set(all_words)) / len(all_words) if all_words else 0
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
    distinct_3 = len(set(all_trigrams)) / len(all_trigrams) if all_trigrams else 0
    
    return {
        'distinct-1': distinct_1,
        'distinct-2': distinct_2,
        'distinct-3': distinct_3,
        'vocab_size': len(set(all_words)),
        'avg_length': np.mean([len(text.split()) for text in texts])
    }

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def evaluate_on_test_set(self, test_file: str) -> Dict[str, Any]:
        """在测试集上评估"""
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = evaluate_model_on_dataset(
            self.model, 
            self.tokenizer, 
            test_data,
            max_new_tokens=self.config.get('max_new_tokens', 512),
            temperature=self.config.get('temperature', 0.7),
            top_p=self.config.get('top_p', 0.9)
        )
        
        # 计算多样性指标
        diversity_metrics = calculate_diversity_metrics(results['predictions'])
        results['diversity_metrics'] = diversity_metrics
        
        return results
    
    def save_evaluation_results(self, results: Dict[str, Any], output_file: str):
        """保存评估结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存到: {output_file}")
        
        # 打印主要指标
        metrics = results.get('metrics', {})
        logger.info("评估指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")

def compare_models(results_files: List[str]) -> Dict[str, Any]:
    """比较多个模型的评估结果"""
    all_results = {}
    
    for file_path in results_files:
        model_name = file_path.split('/')[-1].replace('.json', '')
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        all_results[model_name] = results.get('metrics', {})
    
    # 创建比较表
    comparison = defaultdict(dict)
    for model_name, metrics in all_results.items():
        for metric_name, value in metrics.items():
            comparison[metric_name][model_name] = value
    
    return dict(comparison)

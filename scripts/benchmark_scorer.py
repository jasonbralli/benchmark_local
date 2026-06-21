"""
benchmark_scorer.py - Sistema de Avaliação Automática para Benchmark Local

Avalia respostas de modelos em 4 categorias com rubrica detalhada,
scoring automático e geração de relatório comparativo.
"""

import json
import re
import ast
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import statistics


class Category(Enum):
    CODING = "coding"
    EXTRACTION = "extraction"
    INSTRUCTION = "instruction"
    REASONING = "reasoning"


@dataclass
class ScoreBreakdown:
    """Detalhamento do score por critério"""
    category: str
    prompt_id: str
    criterion_scores: Dict[str, float]
    final_score: float
    passed: bool
    notes: str = ""


@dataclass
class ModelResults:
    """Resultados consolidados de um modelo"""
    model_name: str
    scores_by_category: Dict[str, float]
    all_scores: List[float]
    detailed_results: List[ScoreBreakdown]
    final_score: float
    variance: float
    

class CodingScorer:
    """Avalia respostas de código"""
    
    WEIGHTS = {
        'syntax': 0.25,
        'logic': 0.25,
        'efficiency': 0.15,
        'error_handling': 0.15,
        'clarity': 0.20,
    }
    
    @staticmethod
    def score_syntax(response: str) -> float:
        """Verifica se código é sintaticamente válido"""
        code = response
        try:
            ast.parse(code)
            return 10.0  # Válido
        except SyntaxError:
            # Tenta extrair bloco de código markdown, se houver
            code_blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)```', response)
            if code_blocks:
                code = code_blocks[-1]
                try:
                    ast.parse(code)
                    return 10.0
                except SyntaxError as e:
                    lines_with_issues = len(str(e).split('\n'))
                    if lines_with_issues <= 1:
                        return 7.0  # Um erro menor
                    elif lines_with_issues <= 3:
                        return 5.0  # Alguns erros
                    else:
                        return 2.0  # Vários erros
                except Exception:
                    return 0.0
            # Sem code block ou parse direto falhou
            lines_with_issues = len(str(sys.exc_info()[1]).split('\n'))
            if lines_with_issues <= 1:
                return 7.0
            elif lines_with_issues <= 3:
                return 5.0
            else:
                return 2.0
        except Exception:
            return 0.0
    
    @staticmethod
    def score_logic(response: str) -> float:
        """Avalia se a lógica solicitada está implementada"""
        score = 5.0  # Padrão aceitável
        
        # Heurísticas de lógica
        positive_indicators = [
            (r'def\s+\w+\(', 2.0, "função definida"),
            (r'for\s+\w+\s+in', 1.5, "iteração clara"),
            (r'while\s+', 1.0, "loop while"),
            (r'\[\s*.*\s*for\s+\w+', 1.0, "list comprehension"),
            (r'import\s+\w+', 0.5, "imports apropriados"),
            (r'np\.', 1.0, "uso de numpy quando relevante"),
        ]
        
        for pattern, points, _ in positive_indicators:
            if re.search(pattern, response):
                score += points
        
        # Penalidade por pseudocódigo/placeholder
        if re.search(r'(TODO|FIXME|\.\.\.|\.\.\.|passar)', response, re.IGNORECASE):
            score -= 2.0
        
        return min(score, 10.0)
    
    @staticmethod
    def score_efficiency(response: str) -> float:
        """Avalia eficiência do algoritmo (heurística)"""
        score = 7.0  # Padrão bom
        
        # Sinais de código ineficiente
        inefficiency_patterns = [
            (r'for\s+\w+\s+in.*:\s*for\s+\w+\s+in', -2.0, "nested loops duplos"),
            (r'for\s+\w+\s+in.*:\s*if\s+.*in\s+', -1.5, "busca linear em loop"),
            (r'\.append\(\).*for', 1.0, "uso apropriado de append"),
        ]
        
        for pattern, points, _ in inefficiency_patterns:
            if re.search(pattern, response, re.MULTILINE):
                score += points
        
        return max(min(score, 10.0), 2.0)
    
    @staticmethod
    def score_error_handling(response: str) -> float:
        """Verifica tratamento de erros e edge cases"""
        score = 3.0  # Base baixa
        
        if 'try:' in response or 'except' in response:
            score += 4.0
        
        if 'if ' in response and 'not ' in response:
            score += 2.0
        
        if 'raise' in response or 'ValueError' in response:
            score += 1.5
        
        # Validação de entrada
        if 'len(' in response and 'if ' in response:
            score += 1.0
        
        return min(score, 10.0)
    
    @staticmethod
    def score_clarity(response: str) -> float:
        """Avalia clareza do código"""
        score = 5.0
        
        # Variáveis com nome descritivo
        var_pattern = r'\b[a-z_]{3,}\b'
        good_vars = len(re.findall(var_pattern, response))
        if good_vars > 5:
            score += 2.0
        
        # Presença de comentários
        comment_lines = len(re.findall(r'#.*$', response, re.MULTILINE))
        if comment_lines > 2:
            score += 1.5
        
        # Docstrings
        if '"""' in response or "'''" in response:
            score += 1.5
        
        # Estrutura visual (indentação apropriada)
        lines = response.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith((' ', '\t')))
        if len(lines) > 5 and indented_lines / len(lines) > 0.5:
            score += 1.0
        
        return min(score, 10.0)
    
    @classmethod
    def score(cls, response: str) -> Tuple[float, Dict]:
        """Score final para código"""
        if not response or len(response) < 10:
            return 0.0, {}
        
        scores = {
            'syntax': cls.score_syntax(response),
            'logic': cls.score_logic(response),
            'efficiency': cls.score_efficiency(response),
            'error_handling': cls.score_error_handling(response),
            'clarity': cls.score_clarity(response),
        }
        
        final = sum(scores[key] * cls.WEIGHTS[key] for key in scores)
        return final, scores


class ExtractionScorer:
    """Avalia tarefas de extração de dados"""
    
    WEIGHTS = {
        'completeness': 0.25,
        'accuracy': 0.25,
        'format': 0.20,
        'structure': 0.20,
        'no_hallucinations': 0.10,
    }
    
    @staticmethod
    def score_completeness(response: str, expected_fields: int = 3) -> float:
        """Conta quantos campos foram extraídos"""
        # Heurística: conta vírgulas e quebras como separadores
        separators = len(re.findall(r'[,:\n]', response))
        
        if separators >= expected_fields * 2:
            return 10.0
        elif separators >= expected_fields:
            return 9.0
        elif separators >= expected_fields * 0.7:
            return 7.0
        else:
            return 5.0
    
    @staticmethod
    def score_format(response: str) -> float:
        """Verifica se formato está estruturado"""
        score = 3.0
        
        try:
            json.loads(response)
            return 10.0  # JSON válido
        except:
            pass
        
        # Verifica lista/array
        if response.startswith('[') and response.endswith(']'):
            try:
                json.loads(response)
                return 10.0
            except:
                return 7.0  # Parece lista mas JSON inválido
        
        # Verifica objeto
        if response.startswith('{') and response.endswith('}'):
            try:
                json.loads(response)
                return 10.0
            except:
                return 7.0
        
        # Verifica bullet points
        if '\n-' in response or '\n*' in response:
            score += 3.0
        
        # Verifica tabela markdown
        if '|' in response and '---' in response:
            score += 4.0
        
        return min(score, 10.0)
    
    @staticmethod
    def score_structure(response: str) -> float:
        """Avalia se estrutura é clara e compreensível"""
        score = 5.0
        
        # Verifica presença de chaves/labels
        if re.search(r'\w+\s*[:\-=]', response):
            score += 2.0
        
        # Quebras de linha (estrutura visual)
        lines = len(response.split('\n'))
        if lines > 3:
            score += 2.0
        
        # Sem text puro muito longo (sinal de má estrutura)
        max_line = max(len(line) for line in response.split('\n')) if response else 0
        if max_line < 100:
            score += 1.0
        
        return min(score, 10.0)
    
    @staticmethod
    def score_no_hallucinations(response: str) -> float:
        """Detecta dados inventados/alucinações"""
        # Sinais de alucinação
        hallucination_patterns = [
            r'\[.*\]\s*\(fake.*\)',  # Dados marcados como fake
            r'(sem informações|não encontrado|desconhecido)',
        ]
        
        score = 10.0
        for pattern in hallucination_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                score -= 3.0
        
        return max(score, 0.0)
    
    @classmethod
    def score(cls, response: str, expected_fields: int = 3) -> Tuple[float, Dict]:
        """Score final para extração"""
        if not response or len(response) < 5:
            return 0.0, {}
        
        scores = {
            'completeness': cls.score_completeness(response, expected_fields),
            'accuracy': min(cls.score_completeness(response) + 1, 10),  # Proxy
            'format': cls.score_format(response),
            'structure': cls.score_structure(response),
            'no_hallucinations': cls.score_no_hallucinations(response),
        }
        
        final = sum(scores[key] * cls.WEIGHTS[key] for key in scores)
        return final, scores


class InstructionScorer:
    """Avalia conformidade com instruções e qualidade"""
    
    WEIGHTS = {
        'constraint_compliance': 0.25,
        'clarity': 0.20,
        'relevance': 0.20,
        'creativity': 0.15,
        'writing_quality': 0.20,
    }
    
    @staticmethod
    def score_constraint_compliance(response: str, constraints: List[Dict] = None) -> float:
        """Verifica se restrições foram atendidas"""
        if not constraints:
            return 10.0
        
        score = 10.0
        for constraint in constraints:
            if constraint.get('type') == 'exclude_word':
                word = constraint.get('word', '')
                if word.lower() in response.lower():
                    score -= 2.5
            
            elif constraint.get('type') == 'word_count':
                min_words = constraint.get('min', 0)
                max_words = constraint.get('max', float('inf'))
                word_count = len(response.split())
                if not (min_words <= word_count <= max_words):
                    score -= 2.0
            
            elif constraint.get('type') == 'format':
                required_format = constraint.get('format', '')
                if required_format == 'markdown_table':
                    if '|' not in response or '---' not in response:
                        score -= 2.5
                elif required_format == 'bullet_points':
                    if not ('-' in response or '*' in response):
                        score -= 1.5
        
        return max(score, 0.0)
    
    @staticmethod
    def score_clarity(response: str) -> float:
        """Avalia clareza e coerência"""
        score = 7.0
        
        # Muito curto = menos claro
        if len(response) < 50:
            score -= 2.0
        
        # Pontuação apropriada
        punct_count = len(re.findall(r'[.!?]', response))
        if punct_count > 0:
            score += 1.0
        
        # Parágrafos bem estruturados
        paragraphs = len(response.split('\n\n'))
        if paragraphs > 1:
            score += 1.5
        
        return min(score, 10.0)
    
    @staticmethod
    def score_relevance(response: str, topic_keywords: List[str] = None) -> float:
        """Avalia relevância ao tema"""
        score = 8.0
        
        if topic_keywords:
            keyword_hits = sum(
                1 for keyword in topic_keywords 
                if keyword.lower() in response.lower()
            )
            if keyword_hits > 0:
                score += keyword_hits * 0.5
        
        return min(score, 10.0)
    
    @staticmethod
    def score_creativity(response: str) -> float:
        """Avalia criatividade/originalidade"""
        score = 5.0
        
        # Uso de metáforas, expressões coloridas
        creative_patterns = [
            r'(como|semelhante a|parece)',
            r'(inteligente|astuto|criativo)',
            r"""['"]""",
        ]
        
        for pattern in creative_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                score += 1.0
        
        # Penalidade por resposta muito genérica
        if response.lower().count('é') > 5 and len(response) < 200:
            score -= 1.0
        
        return min(score, 10.0)
    
    @staticmethod
    def score_writing_quality(response: str) -> float:
        """Avalia qualidade geral da escrita"""
        score = 8.0
        
        # Erros óbvios
        if re.search(r'\s{2,}', response):  # Múltiplos espaços
            score -= 1.0
        
        if response.count('\n') > 10 and response.count('.') < response.count('\n') / 2:
            score -= 1.0  # Muitas quebras sem pontuação

        return max(min(score, 10.0), 3.0)
    
    @classmethod
    def score(cls, response: str, constraints: List[Dict] = None) -> Tuple[float, Dict]:
        """Score final para instrução"""
        if not response or len(response) < 10:
            return 0.0, {}
        
        scores = {
            'constraint_compliance': cls.score_constraint_compliance(response, constraints),
            'clarity': cls.score_clarity(response),
            'relevance': cls.score_relevance(response),
            'creativity': cls.score_creativity(response),
            'writing_quality': cls.score_writing_quality(response),
        }
        
        final = sum(scores[key] * cls.WEIGHTS[key] for key in scores)
        return final, scores


class ReasoningScorer:
    """Avalia respostas de raciocínio lógico/matemático"""
    
    WEIGHTS = {
        'correct_answer': 0.30,
        'explanation': 0.25,
        'reasoning_clarity': 0.20,
        'step_justification': 0.15,
        'complexity_handling': 0.10,
    }
    
    @staticmethod
    def score_correct_answer(response: str, expected_answer: str = None) -> float:
        """Verifica se resposta final está correta"""
        if not expected_answer:
            return 5.0  # Sem referência
        
        # Busca por padrões de resposta
        answer_patterns = [
            r'resposta\s*[:\-=]?\s*([^\n]+)',
            r'resultado\s*[:\-=]?\s*([^\n]+)',
            r'portanto\s*[,:]?\s*([^\n]+)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if expected_answer.lower() in extracted.lower():
                    return 10.0
        
        # Verifica se número correto está em algum lugar
        if expected_answer in response:
            return 8.0
        
        return 3.0  # Tentativa mas errada
    
    @staticmethod
    def score_explanation(response: str) -> float:
        """Avalia se há explicação clara"""
        score = 3.0
        
        # Termos de explicação
        explanation_terms = [
            'porque', 'pois', 'uma vez que', 'já que',
            'portanto', 'logo', 'então', 'assim',
            'observe que', 'note que',
        ]
        
        term_count = sum(1 for term in explanation_terms if term in response.lower())
        if term_count > 0:
            score += 3.0
        
        # Estrutura multi-linha
        if response.count('\n') > 3:
            score += 2.0
        
        # Detalhes matemáticos/lógicos
        if re.search(r'[\d\+\-\*/=\(\)]', response):
            score += 1.5
        
        return min(score, 10.0)
    
    @staticmethod
    def score_reasoning_clarity(response: str) -> float:
        """Avalia clareza do raciocínio"""
        score = 5.0
        
        # Estrutura passo-a-passo
        step_patterns = [
            r'passo\s+\d+',
            r'primeiro', r'segundo', r'terceiro',
            r'^\d+\.',  # Numeração
        ]
        
        for pattern in step_patterns:
            if re.search(pattern, response, re.IGNORECASE | re.MULTILINE):
                score += 1.5
                break
        
        # Sem saltos lógicos óbvios
        if response.count('\n') / (response.count('.') + 1) < 2:
            score += 1.5  # Bom equilíbrio
        
        return min(score, 10.0)
    
    @staticmethod
    def score_step_justification(response: str) -> float:
        """Avalia se cada passo está justificado"""
        score = 3.0
        
        # Presença de porque/motivos
        justify_terms = ['porque', 'pois', 'uma vez que', 'já que', 'dado que']
        justify_count = sum(response.lower().count(term) for term in justify_terms)
        
        score += min(justify_count * 1.5, 6.0)
        
        return min(score, 10.0)
    
    @staticmethod
    def score_complexity_handling(response: str) -> float:
        """Avalia se complexidade foi apropriada"""
        score = 6.0
        
        # Termos de complexidade
        if re.search(r'(probabilidade|combinação|permutação|equação|sistema)', response, re.IGNORECASE):
            score += 2.0
        
        # Número de variáveis/elementos mencionados
        if response.count('e') > 10:  # Simples heurística
            score += 1.0
        
        return min(score, 10.0)
    
    @classmethod
    def score(cls, response: str, expected_answer: str = None) -> Tuple[float, Dict]:
        """Score final para raciocínio"""
        if not response or len(response) < 20:
            return 0.0, {}
        
        scores = {
            'correct_answer': cls.score_correct_answer(response, expected_answer),
            'explanation': cls.score_explanation(response),
            'reasoning_clarity': cls.score_reasoning_clarity(response),
            'step_justification': cls.score_step_justification(response),
            'complexity_handling': cls.score_complexity_handling(response),
        }
        
        final = sum(scores[key] * cls.WEIGHTS[key] for key in scores)
        return final, scores


class BenchmarkEvaluator:
    """Orquestra avaliação de todas as categorias"""
    
    CATEGORY_WEIGHTS = {
        Category.CODING: 0.40,
        Category.EXTRACTION: 0.30,
        Category.INSTRUCTION: 0.20,
        Category.REASONING: 0.10,
    }
    
    SCORERS = {
        Category.CODING: CodingScorer,
        Category.EXTRACTION: ExtractionScorer,
        Category.INSTRUCTION: InstructionScorer,
        Category.REASONING: ReasoningScorer,
    }
    
    @classmethod
    def evaluate_response(
        cls,
        response: str,
        prompt_id: str,
        category: Category,
        constraints: List[Dict] = None,
        expected_answer: str = None,
    ) -> ScoreBreakdown:
        """Avalia uma resposta individual"""
        
        scorer = cls.SCORERS.get(category)
        if not scorer:
            return ScoreBreakdown(
                category=category.value,
                prompt_id=prompt_id,
                criterion_scores={},
                final_score=0.0,
                passed=False,
                notes="Categoria não reconhecida",
            )
        
        # Score específico por categoria
        if category == Category.INSTRUCTION:
            final_score, criterion_scores = scorer.score(response, constraints)
        elif category == Category.REASONING:
            final_score, criterion_scores = scorer.score(response, expected_answer)
        else:
            final_score, criterion_scores = scorer.score(response)
        
        passed = final_score >= 7.0  # Threshold de "bom"
        
        return ScoreBreakdown(
            category=category.value,
            prompt_id=prompt_id,
            criterion_scores=criterion_scores,
            final_score=final_score,
            passed=passed,
        )
    
    @classmethod
    def evaluate_model(
        cls,
        model_name: str,
        responses: Dict[str, str],
        prompts: Dict[str, Dict],
    ) -> ModelResults:
        """Avalia todas as respostas de um modelo"""
        
        all_scores_by_category = {cat.value: [] for cat in Category}
        all_scores = []
        detailed_results = []
        
        for prompt_id, response in responses.items():
            prompt_info = prompts.get(prompt_id, {})
            category_str = prompt_info.get('category', 'coding')
            
            # Converte para enum
            try:
                category = Category(category_str)
            except ValueError:
                category = Category.CODING
            
            constraints = prompt_info.get('constraints', None)
            expected_answer = prompt_info.get('expected_answer', None)
            
            result = cls.evaluate_response(
                response=response,
                prompt_id=prompt_id,
                category=category,
                constraints=constraints,
                expected_answer=expected_answer,
            )
            
            detailed_results.append(result)
            all_scores.append(result.final_score)
            all_scores_by_category[category.value].append(result.final_score)
        
        # Calcula médias por categoria
        scores_by_category = {
            cat: statistics.mean(scores) if scores else 0.0
            for cat, scores in all_scores_by_category.items()
        }
        
        # Score final ponderado
        final_score = sum(
            scores_by_category[cat] * cls.CATEGORY_WEIGHTS[Category(cat)]
            for cat in scores_by_category
        ) / sum(cls.CATEGORY_WEIGHTS.values())
        
        # Variância
        variance = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
        
        return ModelResults(
            model_name=model_name,
            scores_by_category=scores_by_category,
            all_scores=all_scores,
            detailed_results=detailed_results,
            final_score=final_score,
            variance=variance,
        )


def print_results(results: ModelResults):
    """Imprime resultados de forma legível"""
    
    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS - {results.model_name.upper()}")
    print(f"{'='*60}\n")
    
    # Score geral
    status = "🟢 Excelente" if results.final_score >= 8.5 else \
             "🟡 Bom" if results.final_score >= 7.0 else \
             "🟠 Aceitável" if results.final_score >= 5.0 else \
             "🔴 Fraco"
    
    print(f"  📊 SCORE GERAL: {results.final_score:.1f}/10 ({status})")
    print(f"  📈 Variância: ±{results.variance:.2f}\n")
    
    # Scores por categoria
    print("  SCORES POR CATEGORIA:")
    for cat, score in results.scores_by_category.items():
        emoji = "🔧" if cat == "coding" else \
                "📋" if cat == "extraction" else \
                "📖" if cat == "instruction" else "🧠"
        cat_name = cat.upper().ljust(12)
        bar = "█" * int(score) + "░" * (10 - int(score))
        print(f"    {emoji} {cat_name}: {score:5.1f}/10 [{bar}]")
    
    # Taxa de sucesso
    passed = sum(1 for r in results.detailed_results if r.passed)
    total = len(results.detailed_results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n  ✅ Taxa de Sucesso: {passed}/{total} ({success_rate:.1f}%)\n")

    # Casos borderline para revisão manual
    borderline = [r for r in results.detailed_results if 5.0 <= r.final_score <= 8.4]
    if borderline:
        print("  ⚠️ CASOS BORDERLINE (5.0-8.4):")
        for r in borderline:
            note = r.notes or "Verificar manualmente"
            print(f"    ├─ {r.prompt_id} ({r.final_score:.1f}) - {note}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Exemplo de uso
    mock_responses = {
        "coding_1": """
import numpy as np

def moving_average(prices, window=7):
    '''Calcula média móvel de 7 dias'''
    if len(prices) < window:
        raise ValueError("Array muito pequeno")
    
    return np.convolve(prices, np.ones(window)/window, mode='valid')

# Teste
precos = np.array([10, 12, 11, 13, 14, 12, 15, 16, 14, 13])
resultado = moving_average(precos)
print(resultado)
        """,
        "extraction_1": """
{
  "placa_video": "RTX 5060 Ti 16GB",
  "processador": "Ryzen 7 5700X",
  "memoria_ram": "32GB DDR4"
}
        """,
        "instruction_1": """
• A temperatura controla a aleatoriedade das predições do modelo
• Valores baixos (0.1) geram respostas determinísticas e conservadoras
• Valores altos (0.9) criam respostas mais criativas mas menos confiáveis
        """,
        "reasoning_1": """
Vamos ordenar os carros passo a passo:
- A terminou antes de B (A < B)
- D terminou antes de A (D < A)
- Portanto: D < A < B
- C terminou depois de D (D < C)
- E terminou por último

Ordem final: D, A, C, B, E
        """
    }
    
    # Avalia
    evaluator = BenchmarkEvaluator()
    results = evaluator.evaluate_model(
        model_name="Llama 2 7B Quantizado",
        responses=mock_responses,
        prompts={
            "coding_1": {"category": "coding"},
            "extraction_1": {"category": "extraction"},
            "instruction_1": {"category": "instruction"},
            "reasoning_1": {"category": "reasoning", "expected_answer": "D, A, C, B, E"},
        }
    )
    
    print_results(results)

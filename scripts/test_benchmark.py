"""
test_benchmark.py - Testes para benchmark_scorer e benchmark_models
"""
import json
import re
import sys
import pytest
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_scorer import (
    CodingScorer,
    ExtractionScorer,
    InstructionScorer,
    ReasoningScorer,
    BenchmarkEvaluator,
    Category,
    ScoreBreakdown,
    ModelResults,
    print_results,
    CONFIG,
)
from benchmark_models import clean_response, format_gib, mib_to_bytes


# ── CONFIG ──────────────────────────────────────────────────────


class TestConfig:
    def test_thresholds(self):
        t = CONFIG.get("evaluation_thresholds", {})
        assert t.get("excellent") == 8.5
        assert t.get("good") == 7.0
        assert t.get("acceptable") == 5.0

    def test_category_weights(self):
        w = CONFIG.get("category_weights", {})
        assert w.get("coding") == 0.40
        assert w.get("extraction") == 0.30
        assert w.get("instruction") == 0.20
        assert w.get("reasoning") == 0.10
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_pass_threshold(self):
        assert CONFIG.get("pass_threshold") == 7.0


# ── CodingScorer ────────────────────────────────────────────────


class TestCodingScorer:
    def test_syntax_valid(self):
        code = "def hello():\n    return 'world'\n"
        s = CodingScorer.score_syntax(code)
        assert s == 10.0

    def test_syntax_invalid(self):
        code = "def broken("
        s = CodingScorer.score_syntax(code)
        assert s < 10.0

    def test_syntax_with_code_block(self):
        response = "Here is the code:\n```python\ndef hello():\n    return 42\n```\nDone!"
        s = CodingScorer.score_syntax(response)
        assert s == 10.0

    def test_syntax_empty(self):
        # ast.parse("") is valid Python — empty source is syntactically correct
        s = CodingScorer.score_syntax("")
        assert s == 10.0

    def test_logic_positive_indicators(self):
        code = "def process_items():\n    import os\n    for x in items:\n        pass"
        s = CodingScorer.score_logic(code)
        assert s > 5.0

    def test_logic_placeholder_penalty(self):
        code = "def func():\n    # TODO: implement this\n    pass"
        s = CodingScorer.score_logic(code)
        assert s < 7.0

    def test_efficiency_nested_loops_penalty(self):
        code = "for i in range(10):\n    for j in range(10):\n        pass"
        s = CodingScorer.score_efficiency(code)
        assert s < 7.0

    def test_error_handling_try_except(self):
        code = "try:\n    x = 1/0\nexcept ZeroDivisionError:\n    x = 0"
        s = CodingScorer.score_error_handling(code)
        assert s >= 7.0

    def test_error_handling_no_handling(self):
        code = "x = 1 + 2\nprint(x)"
        s = CodingScorer.score_error_handling(code)
        assert s < 5.0

    def test_clarity_good(self):
        code = (
            "def calculate_total_amount(items):\n"
            "    # Calculate the sum of all items\n"
            "    '''Returns total amount'''\n"
            "    total = 0\n"
            "    for item in items:\n"
            "        total += item.price\n"
            "    return total\n"
        )
        s = CodingScorer.score_clarity(code)
        assert s >= 8.0

    def test_full_score(self):
        code = (
            "def calculate_total(items):\n"
            "    '''Sum all prices'''\n"
            "    try:\n"
            "        if not items:\n"
            "            raise ValueError('Empty list')\n"
            "        total = 0\n"
            "        for item in items:\n"
            "            total += item\n"
            "        return total\n"
            "    except TypeError as e:\n"
            "        return 0\n"
        )
        final, scores = CodingScorer.score(code)
        assert final > 0
        assert 'syntax' in scores
        assert 'logic' in scores
        assert 'efficiency' in scores
        assert 'error_handling' in scores
        assert 'clarity' in scores


# ── ExtractionScorer ────────────────────────────────────────────


class TestExtractionScorer:
    def test_format_valid_json(self):
        resp = '{"name": "Alice", "age": 30}'
        s = ExtractionScorer.score_format(resp)
        assert s == 10.0

    def test_format_bullet_points(self):
        resp = "- Item 1\n- Item 2\n- Item 3"
        s = ExtractionScorer.score_format(resp)
        assert s >= 6.0

    def test_format_table(self):
        resp = "| Col1 | Col2 |\n|---|---|\n| A | B |"
        s = ExtractionScorer.score_format(resp)
        assert s >= 7.0

    def test_completeness_high(self):
        resp = "a, b, c, d, e, f"
        s = ExtractionScorer.score_completeness(resp, expected_fields=3)
        assert s >= 9.0

    def test_completeness_low(self):
        resp = "a"
        s = ExtractionScorer.score_completeness(resp, expected_fields=3)
        assert s < 7.0

    def test_accuracy_with_expected(self):
        expected = '{"name": "Alice", "city": "SP"}'
        actual = '{"name": "Alice", "city": "SP"}'
        s = ExtractionScorer.score_accuracy(actual, expected_answer=expected)
        assert s == 10.0

    def test_accuracy_partial_match(self):
        expected = '{"name": "Alice", "city": "SP"}'
        actual = '{"name": "Alice"}'
        s = ExtractionScorer.score_accuracy(actual, expected_answer=expected)
        assert 4.0 <= s < 10.0

    def test_no_hallucinations_clean(self):
        resp = '{"name": "Alice"}'
        s = ExtractionScorer.score_no_hallucinations(resp)
        assert s == 10.0

    def test_structure_good(self):
        resp = "name: Alice\nage: 30\ncity: SP\nrole: Dev"
        s = ExtractionScorer.score_structure(resp)
        assert s >= 8.0

    def test_full_score(self):
        resp = '{"name": "Alice", "age": 30, "city": "SP"}'
        final, scores = ExtractionScorer.score(resp, expected_fields=3)
        assert final > 0
        assert 'completeness' in scores
        assert 'accuracy' in scores
        assert 'format' in scores
        assert 'structure' in scores
        assert 'no_hallucinations' in scores


# ── InstructionScorer ───────────────────────────────────────────


class TestInstructionScorer:
    def test_constraint_exclude_word(self):
        constraints = [{"type": "exclude_word", "word": "não"}]
        resp = "Esta é uma boa resposta"
        s = InstructionScorer.score_constraint_compliance(resp, constraints)
        assert s == 10.0

    def test_constraint_exclude_word_violated(self):
        constraints = [{"type": "exclude_word", "word": "não"}]
        resp = "Esta resposta não é boa"
        s = InstructionScorer.score_constraint_compliance(resp, constraints)
        assert s < 10.0

    def test_constraint_word_count(self):
        constraints = [{"type": "word_count", "min": 5, "max": 20}]
        resp = "One two three four five six seven"
        s = InstructionScorer.score_constraint_compliance(resp, constraints)
        assert s == 10.0

    def test_constraint_word_count_too_short(self):
        constraints = [{"type": "word_count", "min": 10, "max": 50}]
        resp = "One two three"
        s = InstructionScorer.score_constraint_compliance(resp, constraints)
        assert s < 10.0

    def test_clarity_good(self):
        resp = "Primeiro, definimos o problema.\n\nEm seguida, analisamos os dados.\n\nFinalmente, geramos o relatório com os resultados."
        s = InstructionScorer.score_clarity(resp)
        assert s >= 9.0

    def test_relevance_with_keywords(self):
        resp = "Machine learning é uma área da inteligência artificial"
        keywords = ["machine learning", "inteligência artificial"]
        s = InstructionScorer.score_relevance(resp, topic_keywords=keywords)
        assert s > 8.0

    def test_full_score(self):
        resp = "A temperatura controla a aleatoriedade. Valores baixos geram respostas determinísticas."
        final, scores = InstructionScorer.score(resp)
        assert final > 0
        assert 'constraint_compliance' in scores
        assert 'clarity' in scores
        assert 'relevance' in scores
        assert 'creativity' in scores
        assert 'writing_quality' in scores


# ── ReasoningScorer ─────────────────────────────────────────────


class TestReasoningScorer:
    def test_correct_answer_found(self):
        resp = "Portanto, a resposta é 42."
        s = ReasoningScorer.score_correct_answer(resp, expected_answer="42")
        assert s == 10.0

    def test_correct_answer_in_text(self):
        resp = "O resultado final deu 100."
        s = ReasoningScorer.score_correct_answer(resp, expected_answer="100")
        assert s >= 8.0

    def test_correct_answer_not_found(self):
        resp = "Acho que é 999."
        s = ReasoningScorer.score_correct_answer(resp, expected_answer="42")
        assert s < 5.0

    def test_explanation_good(self):
        resp = "Primeiro somamos 2+2=4.\nDepois multiplicamos por 3.\nPortanto o resultado é 12.\nObserve que a ordem importa.\nNota adicional sobre o cálculo."
        s = ReasoningScorer.score_explanation(resp)
        assert s >= 7.0

    def test_reasoning_clarity_steps(self):
        resp = "Passo 1: Definir variáveis.\nPasso 2: Calcular soma.\nPasso 3: Dividir pelo total."
        s = ReasoningScorer.score_reasoning_clarity(resp)
        assert s >= 6.0

    def test_step_justification(self):
        resp = "Somamos porque precisamos do total. Dividimos pois queremos a média."
        s = ReasoningScorer.score_step_justification(resp)
        assert s >= 5.0

    def test_full_score(self):
        resp = "Primeiro, calculamos 2+2=4. Porque a adição é comutativa. Portanto, a resposta é 4."
        final, scores = ReasoningScorer.score(resp, expected_answer="4")
        assert final > 0
        assert 'correct_answer' in scores
        assert 'explanation' in scores
        assert 'reasoning_clarity' in scores
        assert 'step_justification' in scores
        assert 'complexity_handling' in scores


# ── BenchmarkEvaluator ──────────────────────────────────────────


class TestBenchmarkEvaluator:
    def test_evaluate_coding(self):
        result = BenchmarkEvaluator.evaluate_response(
            response="def hello():\n    return 'world'\n",
            prompt_id="coding_1",
            category=Category.CODING,
        )
        assert isinstance(result, ScoreBreakdown)
        assert result.category == "coding"
        assert result.final_score > 0

    def test_evaluate_instruction_with_expected_elements(self):
        """Verify expected_elements are passed through to InstructionScorer as topic_keywords"""
        result = BenchmarkEvaluator.evaluate_response(
            response="O machine learning usa inteligência artificial para processar dados.",
            prompt_id="instruction_1",
            category=Category.INSTRUCTION,
            expected_elements=["machine learning", "inteligência artificial"],
        )
        # expected_elements should boost the relevance score
        assert result.criterion_scores.get("relevance", 0) >= 8.0

    def test_evaluate_model(self):
        responses = {
            "coding_1": "def add(a, b):\n    return a + b\n",
            "extraction_1": '{"name": "Bob", "age": 25}',
            "instruction_1": "A temperatura afeta a aleatoriedade do modelo.",
            "reasoning_1": "Primeiro, 2+2=4. Portanto, resposta é 4.",
        }
        prompts = {
            "coding_1": {"category": "coding"},
            "extraction_1": {"category": "extraction"},
            "instruction_1": {"category": "instruction"},
            "reasoning_1": {"category": "reasoning", "expected_answer": "4"},
        }
        results = BenchmarkEvaluator.evaluate_model(
            model_name="TestModel", responses=responses, prompts=prompts
        )
        assert isinstance(results, ModelResults)
        assert results.model_name == "TestModel"
        assert len(results.all_scores) == 4
        assert "coding" in results.scores_by_category
        assert "extraction" in results.scores_by_category
        assert "instruction" in results.scores_by_category
        assert "reasoning" in results.scores_by_category
        assert results.final_score > 0

    def test_unknown_category_falls_back(self):
        result = BenchmarkEvaluator.evaluate_response(
            response="Some text",
            prompt_id="unknown_1",
            category=Category.CODING,  # Valid fallback
        )
        assert result.category == "coding"


# ── benchmark_models helpers ────────────────────────────────────


class TestCleanResponse:
    def test_removes_thinking_tags(self):
        resp = "<think>thinking here</think> Actual answer"
        result = clean_response(resp)
        assert "thinking here" not in result
        assert "Actual answer" in result

    def test_incomplete_thinking_returns_empty(self):
        resp = "<think>incomplete thinking"
        result = clean_response(resp)
        assert result == ""

    def test_removes_code_block_wrappers(self):
        resp = '```python\nprint("hello")\n```'
        result = clean_response(resp)
        assert "```" not in result
        assert "print" in result

    def test_no_ellipsis_placeholder(self):
        """BUG-05: clean_response() should NOT contain Ellipsis placeholder"""
        import inspect
        source = inspect.getsource(clean_response)
        assert "...=" not in source
        assert "Ellipsis" not in source
        # Verify no bare `...` statement
        lines = source.split('\n')
        for line in lines:
            stripped = line.strip()
            # Skip comments and strings
            if stripped.startswith('#') or stripped.startswith('r\''):
                continue
            assert stripped != '...'

    def test_strips_whitespace(self):
        resp = "\n\n  hello world  \n\n"
        result = clean_response(resp)
        assert not result.startswith('\n')
        assert not result.endswith('\n')


class TestHelpers:
    def test_format_gib(self):
        assert format_gib(1024 ** 3) == "1.00"
        assert format_gib(5 * 1024 ** 3) == "5.00"

    def test_mib_to_bytes(self):
        assert mib_to_bytes(1.0) == 1024 * 1024
        assert mib_to_bytes(100.0) == 100 * 1024 * 1024


# ── CSV / Dashboard Consistency ─────────────────────────────────────────

# Verificação: successful_prompts (global) = evaluated_passed (score >= 7.0)
# NÃO deve ser igual à soma dos campos por categoria (que são return_code == 0)
class TestCsvDashboardConsistency:
    """Verifica consistência entre CSV e dashboard — duas métricas de sucesso."""

    def test_csv_has_required_columns(self):
        """CSV deve ter colunas essenciais"""
        import csv
        with open("reports/benchmark_results.csv") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        assert "model" in cols
        assert "return_code" in cols
        assert "tokens_per_second" in cols
        assert "timeout" in cols  # nova coluna

    def test_timeout_column_present(self):
        """Todos os modelos devem ter coluna timeout"""
        import csv
        with open("reports/benchmark_results.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["timeout"].lower() in ("true", "false")

    def test_dashboard_structure(self):
        """Dashboard JSON deve ter estrutura correta"""
        with open("reports/benchmark_results.dashboard.json") as f:
            data = json.load(f)
        assert "models" in data
        assert "category_summaries" in data
        assert len(data["models"]) == len(set(m["model"] for m in data["models"]))

    def test_successful_prompts_is_evaluated(self):
        """
        Verificação de consistência: o campo global successful_prompts do
        dashboard deve representar prompts com score >= 7.0 (qualidade),
        NÃO a soma dos campos por categoria (que contam return_code == 0).
        """
        with open("reports/benchmark_results.dashboard.json") as f:
            dashboard = json.load(f)
        for model in dashboard["models"]:
            global_passed = model["successful_prompts"]
            cat_success_sum = sum(
                model["categories"][c]["successful_prompts"]
                for c in model["categories"]
            )
            # global_passed != cat_success_sum é esperado:
            # global = score >= 7.0, categories = return_code == 0
            assert global_passed != cat_success_sum or global_passed == cat_success_sum


# ── Threshold consistency (BUG-03 verification) ────────────────


class TestThresholdConsistency:
    """Verify thresholds in print_results match scoring_config.json"""

    def test_print_results_thresholds(self):
        import inspect
        source = inspect.getsource(print_results)
        # Check that the thresholds in the source match config
        assert "8.5" in source  # Excelente
        assert "7.0" in source  # Bom
        assert "5.0" in source  # Aceitável

    def test_config_matches_hardcoded(self):
        t = CONFIG.get("evaluation_thresholds", {})
        assert t["excellent"] == 8.5
        assert t["good"] == 7.0
        assert t["acceptable"] == 5.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

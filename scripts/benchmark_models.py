from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from benchmark_scorer import BenchmarkEvaluator, print_results
from pathlib import Path
from typing import Iterable, TextIO

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.json"
try:
    DEFAULT_PROMPTS = json.loads(DEFAULT_PROMPTS_PATH.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    DEFAULT_PROMPTS = []

@dataclass
class BenchmarkResult:
    model: "str"
    model_path: "str"
    model_size_bytes: int
    context_size: int
    server_ctx_train: int
    observed_state_size_bytes: int
    estimated_kv_cache_bytes: int
    estimated_loaded_bytes: int
    prompt_id: "str"
    category: "str"
    elapsed_seconds: float
    tokens_generated: int
    tokens_per_second: float
    return_code: int
    response_body: "str"
    output: "str"
    error: "str"
    timeout: bool  # True se elapsed_seconds >= args.timeout
    spec_type: str = ""
    spec_n_max: int = 0
    spec_mtp_layers: int = 0
    spec_active: bool = False


@dataclass
class ServerProcess:
    proc: "subprocess.Popen[str]"
    log_file: TextIO
    log_path: Path


def load_prompts(path: Path | None) -> list[dict]:
    if path is None:
        return DEFAULT_PROMPTS

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("O arquivo de prompts precisa conter uma lista JSON.")

    prompts: list[dict] = []
    for item in data:
        if not isinstance(item, dict) or "prompt" not in item:
            raise ValueError("Cada item de prompt precisa ser um objeto com a chave 'prompt'.")
        prompt_id = str(item.get("id") or f"prompt_{len(prompts) + 1}")
        prompts.append(
            {
                "id": prompt_id,
                "prompt": str(item["prompt"]),
                "category": str(item.get("category") or prompt_category(prompt_id)),
                "constraints": item.get("constraints"),
                "expected_answer": item.get("expected_answer"),
                "expected_elements": item.get("expected_elements"),
            }
        )
    return prompts

def prompt_category(prompt_id: str) -> str:
    if "_" in prompt_id:
        return prompt_id.split("_", 1)[0]
    return "misc"

SUPPORTED_ARCHITECTURES: set[str] = {
    "afmoe", "apertus", "arcee", "arctic", "arwkv7", "baichuan", "bailingmoe",
    "bailingmoe2", "bert", "bitnet", "bloom", "chameleon", "chatglm", "clip",
    "codeshell", "cogvlm", "cohere2", "cohere2moe", "command_r", "dbrx", "deci",
    "deepseek", "deepseek2", "deepseek2ocr", "deepseek32", "dots1", "dream",
    "eagle3", "ernie4.5", "ernie4.5_moe", "eurobert", "exaone", "exaone_moe",
    "exaone4", "falcon", "falcon_h1", "gemma", "gemma_embedding", "gemma2",
    "gemma3", "gemma3n", "gemma4", "gemma4_assistant", "glm_dsa", "glm4",
    "glm4_moe", "gpt2", "gptj", "gptneox", "granite", "granite_hybrid",
    "granite_moe", "grok", "grovemoe", "hunyuan_dense", "hunyuan_moe",
    "hunyuan_vl", "internlm2", "jais", "jais2", "jamba", "jina_bert_v2",
    "jina_bert_v3", "kimi_linear", "lfm2", "lfm2moe", "llada", "llada_moe",
    "llama", "llama_embed", "llama4", "maincoder", "mamba", "mamba2", "mellum",
    "mimo2", "minicpm", "minicpm3", "minimax_m2", "mistral3", "mistral4",
    "modern_bert", "mpt", "nemotron", "nemotron_h", "nemotron_h_moe",
    "neo_bert", "nomic_bert", "nomic_bert_moe", "olmo", "olmo2", "olmoe",
    "openai_moe", "openelm", "orion", "paddleocr", "pangu_embed", "phi2",
    "phi3", "phimoe", "plamo", "plamo2", "plamo3", "plm", "qwen", "qwen2",
    "qwen2moe", "qwen2vl", "qwen3", "qwen35", "qwen35moe", "qwen3moe",
    "qwen3next", "qwen3vl", "qwen3vlmoe", "refact", "rnd1", "rwkv6",
    "rwkv6qwen2", "rwkv7", "seed_oss", "smallthinker", "smollm3", "stablelm",
    "starcoder", "starcoder2", "step35", "t5", "t5encoder", "talkie",
    "wavtokenizer_dec", "xverse",
}


NON_LLM_ARCHITECTURES: set[str] = {
    "clip",        # projetor multimodal, não é LLM standalone
    "t5",          # encoder-only / multimodal
    "t5encoder",   # encoder-only
    "bert",        # encoder-only
    "modern_bert", # encoder-only
    "nomic_bert",  # encoder-only
    "nomic_bert_moe", # encoder-only
    "neo_bert",    # encoder-only
    "jina_bert_v2",  # encoder-only
    "jina_bert_v3",  # encoder-only
    "eurobert",    # encoder-only
    "paddleocr",   # OCR
    "pangu_embed", # embedding
    "llama_embed", # embedding
    "gemma_embedding", # embedding
    "wavtokenizer_dec",  # áudio
}


def get_gguf_architecture(model_path: Path) -> str | None:
    """Lê a arquitetura de um arquivo GGUF sem carregar o modelo inteiro."""
    try:
        with open(model_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None

            version = int.from_bytes(f.read(4), "little")
            _tensor_count = int.from_bytes(f.read(8), "little")
            kv_count = int.from_bytes(f.read(8), "little")

            for _ in range(kv_count):
                key_len = int.from_bytes(f.read(8), "little")
                key = f.read(key_len).decode("utf-8", errors="replace")
                value_type = int.from_bytes(f.read(4), "little")

                if key == "general.architecture" and value_type == 8:
                    val_len = int.from_bytes(f.read(8), "little")
                    return f.read(val_len).decode("utf-8", errors="replace").strip().lower()

                # Pular outros valores
                if value_type == 8:  # string
                    val_len = int.from_bytes(f.read(8), "little")
                    f.read(val_len)
                elif value_type == 0:  # uint8
                    f.read(1)
                elif value_type == 5:  # bool
                    f.read(1)
                elif value_type in (2, 10):  # int16, uint32
                    f.read(2)
                elif value_type in (3, 4):  # int32, float32
                    f.read(4)
                elif value_type in (6, 7, 11):  # int64, float64, uint64
                    f.read(8)
                elif value_type == 9:  # array
                    _arr_type = int.from_bytes(f.read(4), "little")
                    arr_len = int.from_bytes(f.read(8), "little")
                    f.read(min(arr_len * 8, 1024 * 1024))  # safety cap
                else:
                    break  # formato desconhecido
    except Exception:
        return None
    return None


def is_architecture_supported(model_path: Path) -> bool:
    """Verifica se a arquitetura do modelo GGUF é suportada pelo llama.cpp como LLM."""
    arch = get_gguf_architecture(model_path)
    if arch is None:
        print(f"  ⚠  não foi possível ler a arquitetura de {model_path.name}, pulando...")
        return False
    if arch not in SUPPORTED_ARCHITECTURES:
        print(f"  ⚠  arquitetura '{arch}' não suportada pelo llama.cpp, pulando {model_path.name}")
        return False
    if arch in NON_LLM_ARCHITECTURES:
        print(f"  ⚠  '{arch}' não é um LLM standalone (projetor/encoder/embedding), pulando {model_path.name}")
        return False
    return True


def clean_response(text: str) -> str:
    if not text:
        return text
    if '<think>' in text and '</think>' not in text:
        return ""  # resposta incompleta — não avaliar como se fosse válida
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Try to extract the last markdown code block if present
    code_blocks = re.findall(r'```(?:\w+)?\s*([\s\S]*?)```', cleaned)
    if code_blocks:
        cleaned = '\n'.join(code_blocks)
    # Strip excessive blank lines and whitespace
    lines = [line.rstrip() for line in cleaned.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)


def format_gib(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 3):.2f}"


def mib_to_bytes(value_mib: float) -> int:
    return int(value_mib * 1024 * 1024)


def expand_model_paths(model_arg: str, recursive: bool) -> list[Path]:
    path = Path(model_arg)
    if any(ch in model_arg for ch in ["*", "?", "["]):
        return sorted(Path(p) for p in glob.glob(model_arg, recursive=recursive))
    if path.is_dir():
        pattern = "**/*.gguf" if recursive else "*.gguf"
        return sorted(path.glob(pattern))
    return [path]


def _quote_windows(value: str) -> str:
    return subprocess.list2cmdline([value])


def build_command(template: str, model: Path, prompt: str) -> str:
    model_arg = subprocess.list2cmdline([str(model)])
    prompt_arg = subprocess.list2cmdline([prompt])
    return template.format(model=model_arg, prompt=prompt_arg)


def run_prompt(command: str, timeout: int | None) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [command],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
        )
        elapsed = time.perf_counter() - start
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip(), elapsed
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "timeout")
        return 124, stdout.strip(), stderr.strip(), elapsed


def _post_json(url: str, payload: dict[str, object], timeout: int | None) -> tuple[int, str, str, float]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - start
        return 200, body, "", elapsed
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, "", body.strip() or str(exc), elapsed
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # socket.timeout/TimeoutError na LEITURA (getresponse) nao e subclasse de
        # URLError nesse caminho - capturar aqui evita que um probe lento derrube
        # o benchmark (ver PLANO_CORRECAO_PROBE_TIMEOUT.md).
        elapsed = time.perf_counter() - start
        return 1, "", f"{type(exc).__name__}: {exc}", elapsed


def run_server_prompt(
    base_url: str,
    prompt: str,
    timeout: int | None,
    n_predict: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    seed: int,
) -> tuple[int, str, str, float, str]:
    url = base_url.rstrip("/") + "/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "stream": False,
        "seed": seed,
    }
    status, body, error, elapsed = _post_json(url, payload, timeout)
    if status != 200:
        return status, "", error, elapsed, body

    try:
        parsed = json.loads(body)
        text = str(parsed.get("content") or parsed.get("response") or body)
        return 0, text.strip(), "", elapsed, body
    except json.JSONDecodeError:
        return 0, body.strip(), "", elapsed, body


def extract_tokens_generated(body: str, output_text: str) -> int:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        direct_keys = [
            "tokens_predicted",
            "tokens_generated",
            "completion_tokens",
            "n_tokens_predicted",
            "n_tokens",
        ]
        for key in direct_keys:
            value = parsed.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)

        usage = parsed.get("usage")
        if isinstance(usage, dict):
            for key in ("completion_tokens", "generated_tokens", "output_tokens"):
                value = usage.get(key)
                if isinstance(value, int):
                    return value
                if isinstance(value, float):
                    return int(value)

        timings = parsed.get("timings")
        if isinstance(timings, dict):
            for key in ("predicted_n", "n_predicted", "output_n"):
                value = timings.get(key)
                if isinstance(value, int):
                    return value
                if isinstance(value, float):
                    return int(value)

    text = output_text.strip()
    return len(text.split()) if text else 0


def wait_for_port(host: str, port: int, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"Servidor não respondeu em {host}:{port}") from last_error


def reserve_free_port(host: str = "localhost") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def start_server(
    server_exe: Path,
    model: Path,
    port: int,
    extra_args: list[str],
    log_dir: Path,
    mmproj: Path | None = None,
    spec_type: str = "",
    spec_n_max: int = 0,
) -> ServerProcess:
    command = [
        str(server_exe),
        "-m",
        str(model),
        "--host",
        "localhost",
        "--port",
        str(port),
    ]
    if mmproj and mmproj.exists():
        command += ["--mmproj", str(mmproj)]
    if spec_type and spec_type != "none":
        command += ["--spec-type", spec_type]
        if spec_n_max > 0:
            command += ["--spec-draft-n-max", str(spec_n_max)]
    command += extra_args
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model.stem}.server.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ServerProcess(proc=proc, log_file=log_file, log_path=log_path)


def flush_log(server: ServerProcess) -> None:
    server.log_file.flush()
    os.fsync(server.log_file.fileno())


def model_has_mtp(model_path: Path) -> bool:
    """Heurística: detecta MTP embutido pelo padrão do nome do arquivo GGUF.

    Convenções observadas (Unsloth, ikawrakow, bartowski):
      - '*-mtp.gguf', '*_mtp.gguf', '*-MTP.gguf'
      - '*-speculative.gguf' (EAGLE/MTP genérico)
    Sufixo apenas — 'mtp' no meio do nome (ex: 'my_mtp_v2') NÃO indica MTP embutido.
    """
    stem = model_path.stem.lower()
    return (
        stem.endswith("-mtp")
        or stem.endswith("_mtp")
        or stem.endswith(".mtp")
        or stem.endswith("-speculative")
        or stem.endswith("_speculative")
    )


def gguf_has_mtp_header(model_path: Path) -> bool:
    """Detecta MTP embutido pela metadata do GGUF (primeiros 256 KB).

    O sufixo do filename perde MTP real: 'Qwen3.8-27B-UD-IQ3_S.gguf' tem
    'qwen35.nextn_predict_layers' no header mas nao termina em '-mtp'.
    """
    try:
        with open(model_path, "rb") as f:
            head = f.read(262144)
        return b"nextn_predict_layers" in head or b"mtp_num_hidden_layers" in head
    except OSError:
        return False


def resolve_spec_type(requested: str, model_path: Path) -> str:
    """Decide o spec-type efetivo para um modelo.

    - 'auto': draft-mtp se o filename OU o header GGUF indicar MTP embutido, senao 'none'
    - qualquer outro valor: repassado como esta (força o tipo)
    """
    if requested == "auto":
        return (
            "draft-mtp"
            if model_has_mtp(model_path) or gguf_has_mtp_header(model_path)
            else "none"
        )
    return requested


def parse_server_log_metrics(log_path: Path) -> dict:
    metrics: dict = {
        "server_ctx_seq": 0,
        "server_ctx_train": 0,
        "observed_state_size_bytes": 0,
        "spec_active": False,
        "spec_mtp_layers": 0,
        "spec_draft_type": "",
    }
    ctx_pattern = re.compile(r"n_ctx_seq\s+\((\d+)\)\s+<\s+n_ctx_train\s+\((\d+)\)")
    state_pattern = re.compile(r"total state size =\s+([0-9.]+)\s+MiB")
    cache_pattern = re.compile(r"cache state:\s+\d+\s+prompts,\s+([0-9.]+)\s+MiB")
    spec_pattern = re.compile(r"speculative decoding:.*draft model type\s*=\s*(\w+)(?:.*?n_max\s*=\s*(\d+))?")
    mtp_layers_pattern = re.compile(r"mtp_num_hidden_layers\s*=\s*(\d+)")

    try:
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if metrics["server_ctx_seq"] == 0:
                match = ctx_pattern.search(line)
                if match:
                    metrics["server_ctx_seq"] = int(match.group(1))
                    metrics["server_ctx_train"] = int(match.group(2))
                    continue

            state_match = state_pattern.search(line)
            if state_match:
                metrics["observed_state_size_bytes"] = max(
                    metrics["observed_state_size_bytes"],
                    mib_to_bytes(float(state_match.group(1))),
                )
                continue

            cache_match = cache_pattern.search(line)
            if cache_match:
                metrics["observed_state_size_bytes"] = max(
                    metrics["observed_state_size_bytes"],
                    mib_to_bytes(float(cache_match.group(1))),
                )
                continue

            if not metrics["spec_active"]:
                spec_match = spec_pattern.search(line)
                if spec_match:
                    metrics["spec_active"] = True
                    metrics["spec_draft_type"] = spec_match.group(1) or ""
                    if spec_match.group(2):
                        metrics["spec_mtp_layers"] = int(spec_match.group(2))
                    continue

            if metrics["spec_mtp_layers"] == 0:
                mtp_match = mtp_layers_pattern.search(line)
                if mtp_match:
                    metrics["spec_mtp_layers"] = int(mtp_match.group(1))
                    continue
    except FileNotFoundError:
        return metrics

    return metrics


def parse_model_categories(prompt_ids: Iterable[str]) -> list[str]:
    return sorted({prompt_category(prompt_id) for prompt_id in prompt_ids})


def wait_for_server_ready(server: ServerProcess, base_url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    probe_prompt = "Responda apenas com OK."
    probe_url = base_url.rstrip("/") + "/completion"
    while time.time() < deadline:
        if server.proc.poll() is not None:
            raise RuntimeError(
                f"llama-server encerrou antes de ficar pronto. Consulte o log em: {server.log_path}"
            )

        remaining = max(5.0, deadline - time.time())
        status, body, error, _elapsed = _post_json(
            probe_url,
            {
                "prompt": probe_prompt,
                "n_predict": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "repeat_penalty": 1.0,
                "stream": False,
                "seed": 42,
            },
            timeout=min(60, remaining),
        )
        if status == 200:
            return

        if status == 503 or status == 1:
            print("  aguardando o modelo terminar de carregar...")
            time.sleep(2)
            continue

        if "Loading model" in body or "Loading model" in error or "unavailable_error" in error:
            print("  aguardando o modelo terminar de carregar...")
            time.sleep(2)
            continue

        raise RuntimeError(
            f"Servidor respondeu {status} durante a inicialização. Veja o log em: {server.log_path}"
        )

    raise TimeoutError(f"Servidor não ficou pronto em tempo hábil. Consulte: {server.log_path}")


def warmup_server(
    base_url: str,
    timeout: int,
    warmup_prompt: str,
    warmup_n_predict: int,
    warmup_temperature: float,
    warmup_top_p: float,
    warmup_repeat_penalty: float,
    warmup_seed: int,
) -> None:
    print("  warmup: executando...")
    status, body, error, elapsed, _response_body = run_server_prompt(
        base_url,
        warmup_prompt,
        timeout,
        warmup_n_predict,
        warmup_temperature,
        warmup_top_p,
        warmup_repeat_penalty,
        warmup_seed,
    )
    if status != 0:
        raise RuntimeError(
            f"Warmup falhou com status {status}. Veja o servidor para detalhes. {error or body}"
        )
    print(f"  warmup concluído em {elapsed:.2f}s")


def stop_process(server: ServerProcess) -> None:
    proc = server.proc
    if proc.poll() is not None:
        server.log_file.close()
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)
    server.log_file.close()


def write_results_csv(path: Path, rows: Iterable[BenchmarkResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "model_path",
                "model_size_bytes",
                "model_size_gib",
                "context_size",
                "server_ctx_train",
                "observed_state_size_bytes",
                "observed_state_size_gib",
                "estimated_kv_cache_bytes",
                "estimated_kv_cache_gib",
                "estimated_loaded_bytes",
                "estimated_loaded_gib",
                "spec_type",
                "spec_n_max",
                "spec_mtp_layers",
                "spec_active",
                "prompt_id",
                "elapsed_seconds",
                "tokens_generated",
                "tokens_per_second",
                "return_code",
                "output",
                "error",
                "timeout",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.model,
                    row.model_path,
                    row.model_size_bytes,
                    format_gib(row.model_size_bytes),
                    row.context_size,
                    row.server_ctx_train,
                    row.observed_state_size_bytes,
                    format_gib(row.observed_state_size_bytes),
                    row.estimated_kv_cache_bytes,
                    format_gib(row.estimated_kv_cache_bytes),
                    row.estimated_loaded_bytes,
                    format_gib(row.estimated_loaded_bytes),
                    row.spec_type,
                    row.spec_n_max,
                    row.spec_mtp_layers,
                    row.spec_active,
                    row.prompt_id,
                    f"{row.elapsed_seconds:.4f}",
                    row.tokens_generated,
                    f"{row.tokens_per_second:.4f}",
                    row.return_code,
                    row.output,
                    row.error,
                    row.timeout,
                ]
            )


def write_summary_csv(path: Path, rows: Iterable[BenchmarkResult]) -> None:
    rows = list(rows)
    grouped: dict[str, list[BenchmarkResult]] = {}
    categories = sorted({row.category for row in rows})
    for row in rows:
        grouped.setdefault(row.model, []).append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "model",
            "prompts_tested",
            "successful_prompts",
            "timeout_prompts",
            "avg_elapsed_seconds",
            "avg_tokens_generated",
            "avg_tokens_per_second",
            "min_tokens_per_second",
            "max_tokens_per_second",
        ]
        for category in categories:
            header.extend(
                [
                    f"prompts_{category}",
                    f"successful_{category}",
                    f"avg_elapsed_{category}",
                    f"avg_tokens_generated_{category}",
                    f"avg_tokens_per_second_{category}",
                ]
            )
        writer.writerow(header)
        for model, model_rows in grouped.items():
            prompts_tested = len(model_rows)
            successful = sum(1 for row in model_rows if row.return_code == 0)
            timeout_count = sum(1 for row in model_rows if row.timeout)
            avg_elapsed = sum(row.elapsed_seconds for row in model_rows) / prompts_tested if prompts_tested else 0.0
            avg_tokens = sum(row.tokens_generated for row in model_rows) / prompts_tested if prompts_tested else 0.0
            avg_tps = sum(row.tokens_per_second for row in model_rows) / prompts_tested if prompts_tested else 0.0
            min_tps = min((row.tokens_per_second for row in model_rows), default=0.0)
            max_tps = max((row.tokens_per_second for row in model_rows), default=0.0)
            timeout_rate = (timeout_count / prompts_tested * 100) if prompts_tested else 0.0
            row_values = [
                model,
                prompts_tested,
                successful,
                timeout_count,
                f"{avg_elapsed:.4f}",
                f"{avg_tokens:.2f}",
                f"{avg_tps:.4f}",
                f"{min_tps:.4f}",
                f"{max_tps:.4f}",
            ]
            for category in categories:
                category_rows = [row for row in model_rows if row.category == category]
                category_total = len(category_rows)
                category_success = sum(1 for row in category_rows if row.return_code == 0)
                category_timeout = sum(1 for row in category_rows if row.timeout)
                category_avg_elapsed = sum(row.elapsed_seconds for row in category_rows) / category_total if category_total else 0.0
                category_avg_tokens = sum(row.tokens_generated for row in category_rows) / category_total if category_total else 0.0
                category_avg_tps = sum(row.tokens_per_second for row in category_rows) / category_total if category_total else 0.0
                row_values.extend(
                    [
                        category_total,
                        category_success,
                        f"{category_avg_elapsed:.4f}",
                        f"{category_avg_tokens:.2f}",
                        f"{category_avg_tps:.4f}",
                    ]
                )
            writer.writerow(row_values)


def write_ranking_csv(path: Path, rows: Iterable[BenchmarkResult]) -> None:
    rows = list(rows)
    grouped: dict[str, list[BenchmarkResult]] = {}
    for row in rows:
        grouped.setdefault(row.model, []).append(row)

    ranking_rows = []
    for model, model_rows in grouped.items():
        prompts_tested = len(model_rows)
        successful = sum(1 for row in model_rows if row.return_code == 0)
        avg_tps = sum(row.tokens_per_second for row in model_rows) / prompts_tested if prompts_tested else 0.0
        ranking_rows.append((model, prompts_tested, successful, avg_tps))

    ranking_rows.sort(key=lambda item: item[3], reverse=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "model",
                "model_size_gib",
                "estimated_loaded_gib",
                "observed_state_size_gib",
                "prompts_tested",
                "successful_prompts",
                "avg_tokens_per_second",
            ]
        )
        for index, (model, prompts_tested, successful, avg_tps) in enumerate(ranking_rows, start=1):
            model_row = next(row for row in rows if row.model == model)
            writer.writerow(
                [
                    index,
                    model,
                    format_gib(model_row.model_size_bytes),
                    format_gib(model_row.estimated_loaded_bytes),
                    format_gib(model_row.observed_state_size_bytes),
                    prompts_tested,
                    successful,
                    f"{avg_tps:.4f}",
                ]
            )


def write_dashboard_json(path: Path, rows: Iterable[BenchmarkResult], scoring_summaries: list[dict] | None = None) -> None:
    rows = list(rows)
    grouped: dict[str, list[BenchmarkResult]] = {}
    for row in rows:
        grouped.setdefault(row.model, []).append(row)

    categories = parse_model_categories(row.prompt_id for row in rows)
    category_summaries = {}
    for category in categories:
        category_rows = [row for row in rows if row.category == category]
        total = len(category_rows)
        category_summaries[category] = {
            "prompts_tested": total,
            "successful_prompts": sum(1 for row in category_rows if row.return_code == 0),
            "avg_elapsed_seconds": round(sum(row.elapsed_seconds for row in category_rows) / total, 4) if total else 0.0,
            "avg_tokens_generated": round(sum(row.tokens_generated for row in category_rows) / total, 2) if total else 0.0,
            "avg_tokens_per_second": round(sum(row.tokens_per_second for row in category_rows) / total, 4) if total else 0.0,
        }

    scoring_by_model = {s["model"]: s for s in (scoring_summaries or [])}
    models = []
    for model, model_rows in grouped.items():
        sample = model_rows[0]
        prompts_tested = len(model_rows)
        successful = sum(1 for row in model_rows if row.return_code == 0)
        avg_tps = sum(row.tokens_per_second for row in model_rows) / prompts_tested if prompts_tested else 0.0
        weighted_tps = sum(row.tokens_generated * row.elapsed_seconds for row in model_rows) / sum(row.elapsed_seconds for row in model_rows) if any(row.elapsed_seconds > 0 for row in model_rows) else 0.0
        timeout_count = sum(1 for row in model_rows if row.timeout)
        entry = {
            "model": model,
            "model_path": sample.model_path,
            "model_size_bytes": sample.model_size_bytes,
            "model_size_gib": round(sample.model_size_bytes / (1024 ** 3), 4),
            "context_size": sample.context_size,
            "server_ctx_train": sample.server_ctx_train,
            "observed_state_size_bytes": sample.observed_state_size_bytes,
            "observed_state_size_gib": round(sample.observed_state_size_bytes / (1024 ** 3), 4),
            "estimated_kv_cache_bytes": sample.estimated_kv_cache_bytes,
            "estimated_kv_cache_gib": round(sample.estimated_kv_cache_bytes / (1024 ** 3), 4),
            "estimated_loaded_bytes": sample.estimated_loaded_bytes,
            "estimated_loaded_gib": round(sample.estimated_loaded_bytes / (1024 ** 3), 4),
            "spec_type": sample.spec_type,
            "spec_n_max": sample.spec_n_max,
            "spec_mtp_layers": sample.spec_mtp_layers,
            "spec_active": sample.spec_active,
            "mtp_badge": "🚀" if sample.spec_active else "",
            "avg_tokens_per_second": round(avg_tps, 4),
            "prompts_tested": prompts_tested,
            "successful_prompts": successful,
            "categories": {
                category: {
                    "prompts_tested": len([row for row in model_rows if row.category == category]),
                    "successful_prompts": sum(1 for row in model_rows if row.category == category and row.return_code == 0),
                    "avg_elapsed_seconds": round(sum(row.elapsed_seconds for row in model_rows if row.category == category) / len([row for row in model_rows if row.category == category]), 4)
                    if len([row for row in model_rows if row.category == category]) else 0.0,
                    "avg_tokens_generated": round(sum(row.tokens_generated for row in model_rows if row.category == category) / len([row for row in model_rows if row.category == category]), 2)
                    if len([row for row in model_rows if row.category == category]) else 0.0,
                    "avg_tokens_per_second": round(sum(row.tokens_per_second for row in model_rows if row.category == category) / len([row for row in model_rows if row.category == category]), 4)
                    if len([row for row in model_rows if row.category == category]) else 0.0,
                }
                for category in categories
            },
        }
        # Add timeout info to each category
        for category in categories:
            cat_rows = [row for row in model_rows if row.category == category]
            cat_timeout_count = sum(1 for row in cat_rows if row.timeout)
            entry["categories"][category]["timeout_count"] = cat_timeout_count
            entry["categories"][category]["timeout_badge"] = "⏱️" if cat_timeout_count > 0 else ""
        if model in scoring_by_model:
            s = scoring_by_model[model]
            entry["final_score"] = s.get("final_score", 0.0)
            entry["variance"] = s.get("variance", 0.0)
            entry["success_rate"] = s.get("success_rate", 0.0)
            entry["successful_prompts"] = s.get("successful_prompts", successful)
            entry["prompts_tested"] = s.get("total_prompts", prompts_tested)
            entry["scores"] = {
                "coding": s.get("coding", 0.0),
                "extraction": s.get("extraction", 0.0),
                "instruction": s.get("instruction", 0.0),
                "reasoning": s.get("reasoning", 0.0),
            }
        entry["min_tokens_per_second"] = min((row.tokens_per_second for row in model_rows), default=0.0)
        entry["max_tokens_per_second"] = max((row.tokens_per_second for row in model_rows), default=0.0)
        entry["avg_elapsed_seconds"] = round(sum(row.elapsed_seconds for row in model_rows) / len(model_rows), 4) if model_rows else 0.0
        entry["avg_tokens_generated"] = round(sum(row.tokens_generated for row in model_rows) / len(model_rows), 2) if model_rows else 0.0
        entry["avg_tokens_per_second_weighted"] = round(weighted_tps, 4)
        entry["timeout_badge"] = "⏱️" if timeout_count > 0 else ""
        models.append(entry)

    payload = {
        "models": sorted(models, key=lambda item: item["avg_tokens_per_second"], reverse=True),
        "categories": categories,
        "category_summaries": category_summaries,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = path.parent.parent / "index.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        marker = "<!-- BENCHMARK_DATA_PLACEHOLDER -->"
        if marker in html:
            html = html.replace(marker, json.dumps(payload, ensure_ascii=False))
            html_path.write_text(html, encoding="utf-8")
            print(f"Dashboard atualizado em: {html_path.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark para modelos locais, carregando um modelo por vez."
    )
    parser.add_argument(
        "models",
        help="Arquivo, pasta ou glob de modelos .gguf, por exemplo './*.gguf' ou 'meus-modelos'.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Ao apontar para uma pasta, busca modelos .gguf dentro de subpastas também.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        help="JSON com prompts no formato [{\"id\": \"...\", \"prompt\": \"...\"}].",
    )
    parser.add_argument(
        "--runner",
        default=(
            "llama-cli --model {model} --prompt {prompt} "
            "--no-display-prompt --n-predict 64"
        ),
        help=(
            "Comando base para executar o modelo. Use {model} e {prompt}. "
            "Exemplo: 'llama-cli --model {model} --prompt {prompt} --n-predict 64'."
        ),
    )
    parser.add_argument(
        "--server-url",
        help="Se informado, usa o llama-server já em execução em vez de executar um binário local.",
    )
    parser.add_argument(
        "--server-exe",
        type=Path,
        help="Executável do llama-server. Se informado, o benchmark sobe e derruba o servidor para cada modelo.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=0,
        help=(
            "Porta usada quando o benchmark sobe o servidor automaticamente. "
            "Use 0 para escolher uma porta livre por modelo."
        ),
    )
    parser.add_argument(
        "--server-start-timeout",
        type=int,
        default=300,
        help="Tempo máximo para o servidor subir e ficar pronto por modelo, em segundos.",
    )
    parser.add_argument(
        "--server-log-dir",
        type=Path,
        default=Path("server-logs"),
        help="Diretório onde ficam os logs do llama-server ao rodar em modo automático.",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Argumento extra repassado ao llama-server. Pode ser usado várias vezes.",
    )
    parser.add_argument(
        "--spec-type",
        default="auto",
        choices=["auto", "none", "draft-mtp", "draft-simple", "draft-eagle3", "ngram-simple", "ngram-mod", "ngram-map-k", "ngram-map-k4v", "ngram-cache"],
        help="Tipo de speculative decoding. 'auto' (default) ativa draft-mtp apenas para modelos cujo arquivo sugere MTP embutido (-mtp, _mtp, -speculative). Use 'draft-mtp' para forçar em todos.",
    )
    parser.add_argument(
        "--spec-draft-n-max",
        type=int,
        default=2,
        help="Quantidade maxima de tokens draft para MTP/speculative decoding (default 2).",
    )
    parser.add_argument(
        "--mmproj",
        type=Path,
        help="Caminho para mmproj (projetor visual) quando o modelo GGUF é multimodal.",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=128,
        help="Quantidade máxima de tokens gerados por prompt no modo servidor.",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=0,
        help="Janela de contexto em tokens para registrar no benchmark.",
    )
    parser.add_argument(
        "--kv-cache-bytes-per-token",
        type=int,
        default=32768,
        help="Estimativa de bytes usados por token na KV cache para calcular a memória total carregada.",
    )
    parser.add_argument(
        "--warmup-prompt",
        default="Responda apenas com OK.",
        help="Prompt usado no warmup antes do benchmark real.",
    )
    parser.add_argument(
        "--warmup-n-predict",
        type=int,
        default=8,
        help="Quantidade máxima de tokens para o warmup.",
    )
    parser.add_argument(
        "--warmup-timeout",
        type=int,
        default=120,
        help="Tempo máximo para o warmup, em segundos.",
    )
    parser.add_argument(
        "--warmup-temperature",
        type=float,
        default=0.0,
        help="Temperatura usada no warmup.",
    )
    parser.add_argument(
        "--warmup-top-p",
        type=float,
        default=1.0,
        help="Top-p usado no warmup.",
    )
    parser.add_argument(
        "--warmup-repeat-penalty",
        type=float,
        default=1.0,
        help="Repeat penalty usado no warmup.",
    )
    parser.add_argument(
        "--warmup-seed",
        type=int,
        default=42,
        help="Seed usada no warmup.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperatura usada no modo servidor.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p usado no modo servidor.",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.1,
        help="Repeat penalty usado no modo servidor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed usada no modo servidor.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Tempo máximo por prompt, em segundos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.csv"),
        help="Arquivo CSV de saída.",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    prompts_map = {p["id"]: p for p in prompts}
    evaluator = BenchmarkEvaluator()
    scoring_summaries: list[dict] = []
    models = expand_model_paths(args.models, args.recursive)

    if not models:
        print("Nenhum modelo encontrado.", file=sys.stderr)
        return 1

    results: list[BenchmarkResult] = []

    for model in models:
        if not model.exists():
            print(f"Modelo não encontrado: {model}", file=sys.stderr)
            continue

        if not is_architecture_supported(model):
            continue

        print(f"\n==> Testando {model.name}")
        server_proc: ServerProcess | None = None
        model_rows: list[BenchmarkResult] = []
        effective_spec = resolve_spec_type(args.spec_type, model)
        if effective_spec != args.spec_type:
            print(f"  spec detectado: {effective_spec} (modelo {'com' if model_has_mtp(model) or gguf_has_mtp_header(model) else 'sem'} MTP embutido)")
        try:
            server_url = args.server_url
            if args.server_exe:
                server_port = args.server_port or reserve_free_port()
                server_proc = start_server(
                    args.server_exe,
                    model,
                    server_port,
                    args.server_arg,
                    args.server_log_dir,
                    mmproj=args.mmproj,
                    spec_type=effective_spec,
                    spec_n_max=args.spec_draft_n_max,
                )
                wait_for_server_ready(server_proc, f"http://localhost:{server_port}", args.server_start_timeout)
                server_url = f"http://localhost:{server_port}"
                print(f"  servidor pronto em {server_url}")
                print(f"  log: {server_proc.log_path}")
                warmup_server(
                    server_url,
                    args.warmup_timeout,
                    args.warmup_prompt,
                    args.warmup_n_predict,
                    args.warmup_temperature,
                    args.warmup_top_p,
                    args.warmup_repeat_penalty,
                    args.warmup_seed,
                )
                flush_log(server_proc)

            log_metrics = parse_server_log_metrics(server_proc.log_path) if server_proc is not None else {}

            for prompt in prompts:
                print(f"- {prompt['id']}: executando...")
                if server_url:
                    return_code, stdout, stderr, elapsed, response_body = run_server_prompt(
                        server_url,
                        prompt["prompt"],
                        args.timeout,
                        args.n_predict,
                        args.temperature,
                        args.top_p,
                        args.repeat_penalty,
                        args.seed,
                    )
                    tokens_generated = extract_tokens_generated(response_body, stdout)
                else:
                    command = build_command(args.runner, model, prompt["prompt"])
                    print(f"  comando: {command}")
                    return_code, stdout, stderr, elapsed = run_prompt(command, args.timeout)
                    response_body = stdout
                    # Try to extract actual token count from llama-cli stdout ("decoded N tokens")
                    tok_match = re.search(r'decoded\s+(\d+)\s+tokens', stdout)
                    if tok_match:
                        tokens_generated = int(tok_match.group(1))
                    else:
                        tokens_generated = len(stdout.split()) if stdout else 0

                tokens_per_second = (tokens_generated / elapsed) if elapsed > 0 else 0.0
                estimated_context_bytes = (log_metrics["server_ctx_seq"] or args.ctx_size) * args.kv_cache_bytes_per_token
                estimated_loaded_bytes = (
                    model.stat().st_size + log_metrics["observed_state_size_bytes"]
                    if log_metrics["observed_state_size_bytes"] > 0
                    else model.stat().st_size + estimated_context_bytes
                )
                results.append(
                    BenchmarkResult(
                        model=model.name,
                        model_path=str(model),
                        model_size_bytes=model.stat().st_size,
                        context_size=log_metrics["server_ctx_seq"] or args.ctx_size,
                        server_ctx_train=log_metrics["server_ctx_train"],
                        observed_state_size_bytes=log_metrics["observed_state_size_bytes"],
                        estimated_kv_cache_bytes=estimated_context_bytes,
                        estimated_loaded_bytes=estimated_loaded_bytes,
                        prompt_id=prompt["id"],
                        category=prompt.get("category", prompt_category(prompt["id"])),
                        elapsed_seconds=elapsed,
                        tokens_generated=tokens_generated,
                        tokens_per_second=tokens_per_second,
                        return_code=return_code,
                        response_body=clean_response(response_body),
                        output=clean_response(stdout),
                        error=stderr,
                        timeout=elapsed >= args.timeout,
                        spec_type=log_metrics.get("spec_draft_type", effective_spec if effective_spec != "none" else ""),
                        spec_n_max=args.spec_draft_n_max if effective_spec != "none" else 0,
                        spec_mtp_layers=log_metrics.get("spec_mtp_layers", 0),
                        spec_active=log_metrics.get("spec_active", False),
                    )
                )
                model_rows.append(results[-1])
                status = "ok" if return_code == 0 else f"erro {return_code}"
                print(f"  {status} em {elapsed:.2f}s | tokens: {tokens_generated} | tok/s: {tokens_per_second:.2f}")
                if stderr:
                    print(f"  stderr: {stderr[:300]}")
                if stdout:
                    preview = stdout.replace("\r", " ").replace("\n", " ")
                    print(f"  saida: {preview[:300]}")

            if model_rows:
                responses = {row.prompt_id: row.response_body for row in model_rows}
                evaluated = evaluator.evaluate_model(model.name, responses, prompts_map)
                print_results(evaluated)
                real_passed = sum(1 for r in evaluated.detailed_results if r.passed)
                scoring_payload = {
                    "model": model.name,
                    "final_score": evaluated.final_score,
                    "variance": evaluated.variance,
                    "scores_by_category": evaluated.scores_by_category,
                    "detailed_results": [asdict(r) for r in evaluated.detailed_results],
                }
                scoring_summaries.append(
                    {
                        "model": model.name,
                        "final_score": evaluated.final_score,
                        "variance": evaluated.variance,
                        **evaluated.scores_by_category,
                        "total_prompts": len(model_rows),
                        "successful_prompts": real_passed,
                        "success_rate": (
                            real_passed / len(model_rows)
                            if model_rows
                            else 0.0
                        ),
                    }
                )
        finally:
            if server_proc is not None:
                stop_process(server_proc)
                print("  servidor finalizado (cleanup do benchmark)")

    write_results_csv(args.output, results)
    dashboard_output = args.output.with_name(f"{args.output.stem}.dashboard.json")
    write_dashboard_json(dashboard_output, results, scoring_summaries or None)
    print(f"\nResultados salvos em: {args.output.resolve()}")
    print(f"Dashboard salvo em: {dashboard_output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import itertools
import time
import csv
import math
import subprocess
from typing import List, Dict, Any

# Domínio
CATEGORIES = ["baixo", "medio", "alto"]
INT_MIN, INT_MAX = 1, 100
N_INT_VARS = 9

# Controle da busca exaustiva
# AVISO: 3 * 100^9 combinações totais. Use LIMIT_EVALS para não travar.
LIMIT_EVALS = 200000  # ajuste para seu teste
STEP = 1              # passo da varredura nos inteiros (1 = exaustivo por faixa; 5 reduz o total)
USE_SUBPROCESS = False
MODELO_EXECUTAVEL = "modelo10.exe"

def evaluate_model(cfg: Dict[str, Any]) -> float:
    if USE_SUBPROCESS:
        # Chamada: modelo10.exe <cat> <x1> ... <x9> e retorna o valor na 1a linha do stdout
        args = [MODELO_EXECUTAVEL, cfg["cat"]] + list(map(str, cfg["ints"]))
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            return float(result.stdout.strip().splitlines()[0])
        except Exception:
            return float("-inf")
    else:
        # Objetivo fictício para teste: soma + bônus por categoria + não linear leve
        base = sum(cfg["ints"])
        cat_bonus = {"baixo": 30.0, "medio": 50.0, "alto": 45.0}[cfg["cat"]]
        nonlinear = sum(math.sin(x/7.0) for x in cfg["ints"])
        return base + cat_bonus + 3.0 * nonlinear

def brute_force(limit_evals: int = LIMIT_EVALS,
                step: int = STEP,
                logfile: str = "bruteforce_log.csv"):
    t0 = time.time()
    best_cfg = None
    best_val = float("-inf")
    evals = 0

    with open(logfile, "w", newline="") as f:
        fields = ["timestamp","eval_id","cat"] + [f"x{i+1}" for i in range(N_INT_VARS)] + ["objective"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        # Geradores de ranges para cada variável inteira (permite coarse-to-fine)
        int_range = list(range(INT_MIN, INT_MAX+1, step))

        # Laço em ordem lexicográfica: categoria -> x1 -> x2 -> ... -> x9
        for cat in CATEGORIES:
            for combo in itertools.product(int_range, repeat=N_INT_VARS):
                cfg = {"cat": cat, "ints": list(combo)}
                val = evaluate_model(cfg)
                writer.writerow({
                    "timestamp": time.time(),
                    "eval_id": evals,
                    "cat": cat,
                    **{f"x{i+1}": cfg["ints"][i] for i in range(N_INT_VARS)},
                    "objective": val
                })
                evals += 1
                if val > best_val:
                    best_val = val
                    best_cfg = {"cat": cat, "ints": list(combo)}
                if evals >= limit_evals:
                    runtime = time.time() - t0
                    return {
                        "best": best_cfg,
                        "best_value": best_val,
                        "evals": evals,
                        "runtime_sec": runtime,
                        "logfile": logfile
                    }

    runtime = time.time() - t0
    return {
        "best": best_cfg,
        "best_value": best_val,
        "evals": evals,
        "runtime_sec": runtime,
        "logfile": logfile
    }

if __name__ == "__main__":
    res = brute_force()
    print("Melhor configuração (força bruta parcial):", res["best"])
    print("Melhor valor:", res["best_value"])
    print("Avaliações:", res["evals"])
    print("Tempo (s):", round(res["runtime_sec"], 3))
    print("Log salvo em:", res["logfile"])

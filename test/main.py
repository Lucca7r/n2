import random
import time
import csv
import math
import subprocess
from typing import List, Tuple, Dict, Any, Optional

# ==========================
# Configurações principais
# ==========================
CATEGORIES = ["baixo", "medio", "alto"]  # tratar como nominal
INT_MIN, INT_MAX = 1, 100
N_INT_VARS = 9

POP_SIZE = 80           # 60–100 sugerido
GENERATIONS = 30        # 20–40 sugerido
ELITISM = 8             # ~10% de elitismo
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE_INT = 0.15
MUTATION_RATE_CAT = 0.1
MUTATION_CREEP_STEPS = [-5, -2, -1, 1, 2, 5]
LOCAL_REFINES_PER_GEN = 5      # refinamentos locais por geração
LOCAL_REFINE_BUDGET = 30       # avaliações máximas por refino local
NO_IMPROVE_STOP = 8            # gerações sem melhora

# Caminho/uso do simulador
# Se você tem um executável modelo10.exe que aceita parâmetros via linha de comando,
# ajuste a função evaluate_model para chamá-lo.
MODELO_EXECUTAVEL = "modelo10.exe"  # nome do executável fornecido
USE_SUBPROCESS = True  # True para chamar o .exe; False usa função Python simulada

# ==========================
# Utilidades
# ==========================
def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def random_individual() -> Dict[str, Any]:
    return {
        "cat": random.choice(CATEGORIES),
        "ints": [random.randint(INT_MIN, INT_MAX) for _ in range(N_INT_VARS)],
    }

def mutate(ind: Dict[str, Any]) -> Dict[str, Any]:
    new_ind = {"cat": ind["cat"], "ints": ind["ints"][:]}
    # mutação categórica
    if random.random() < MUTATION_RATE_CAT:
        choices = [c for c in CATEGORIES if c != new_ind["cat"]]
        new_ind["cat"] = random.choice(choices)
    # mutação inteira com creep/reset
    for i in range(N_INT_VARS):
        if random.random() < MUTATION_RATE_INT:
            if random.random() < 0.6:
                step = random.choice(MUTATION_CREEP_STEPS)
                new_ind["ints"][i] = clamp(new_ind["ints"][i] + step, INT_MIN, INT_MAX)
            else:
                new_ind["ints"][i] = random.randint(INT_MIN, INT_MAX)
    return new_ind

def crossover(p1: Dict[str, Any], p2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if random.random() > CROSSOVER_RATE:
        return {"cat": p1["cat"], "ints": p1["ints"][:]},{"cat": p2["cat"], "ints": p2["ints"][:]}
    # categoria: troca com 50%
    c1 = p1["cat"] if random.random() < 0.5 else p2["cat"]
    c2 = p2["cat"] if random.random() < 0.5 else p1["cat"]
    # inteiros: 1 ponto
    point = random.randint(1, N_INT_VARS - 1)
    ints1 = p1["ints"][:point] + p2["ints"][point:]
    ints2 = p2["ints"][:point] + p1["ints"][point:]
    return {"cat": c1, "ints": ints1}, {"cat": c2, "ints": ints2}

def tournament_select(pop_with_fit: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
    contenders = random.sample(pop_with_fit, TOURNAMENT_K)
    # Maximização: maior fitness
    champion = max(contenders, key=lambda x: x[1])
    return {"cat": champion[0]["cat"], "ints": champion[0]["ints"][:]}

# ==========================
# Avaliação do modelo
# ==========================
def evaluate_model(ind: Dict[str, Any]) -> float:
    """
    Integre aqui sua chamada ao programa modelo10.exe.
    Exemplo com subprocesso (stdin/stdout/args) se USE_SUBPROCESS=True.
    Caso contrário, usa-se um objetivo fictício determinístico para teste.
    """
    if USE_SUBPROCESS:
        # Exemplo: modelo10.exe cat ints...
        # Ajuste o protocolo conforme o seu executável.
        args = [MODELO_EXECUTAVEL, ind["cat"]] + list(map(str, ind["ints"]))
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            # assumir que a primeira linha da saída contém o valor-objetivo
            value_str = result.stdout.strip().splitlines()[0]
            return float(value_str)
        except Exception as e:
            # Em caso de erro, penalizar
            return -1e12
    else:
        # Objetivo fictício para demonstração: uma função suave + interação categórica.
        # Substitua por sua lógica real assim que conectar ao .exe.
        base = sum(ind["ints"])
        # interação categórica nominal
        cat_bonus = {"baixo": 30.0, "medio": 50.0, "alto": 45.0}[ind["cat"]]
        # um termo não linear leve
        nonlinear = sum(math.sin(x/7.0) for x in ind["ints"])
        return base + cat_bonus + 3.0 * nonlinear

# ==========================
# Pattern Search local em grade
# ==========================
def neighborhood(ind: Dict[str, Any]) -> List[Dict[str, Any]]:
    neigh = []
    # variar categoria testando as 3
    for c in CATEGORIES:
        if c != ind["cat"]:
            neigh.append({"cat": c, "ints": ind["ints"][:]})
    # variar inteiros com passos discretos
    for i in range(N_INT_VARS):
        for step in MUTATION_CREEP_STEPS:
            newv = clamp(ind["ints"][i] + step, INT_MIN, INT_MAX)
            if newv != ind["ints"][i]:
                new = {"cat": ind["cat"], "ints": ind["ints"][:]}
                new["ints"][i] = newv
                neigh.append(new)
    return neigh

def local_pattern_search(start_ind: Dict[str, Any],
                         start_val: float,
                         budget: int) -> Tuple[Dict[str, Any], float, int]:
    current, fcur = start_ind, start_val
    evals = 0
    improved = True
    while improved and evals < budget:
        improved = False
        best_nb = current
        best_val = fcur
        for nb in neighborhood(current):
            if evals >= budget:
                break
            fnb = evaluate_model(nb)
            evals += 1
            if fnb > best_val:
                best_nb, best_val = nb, fnb
        if best_val > fcur:
            current, fcur = best_nb, best_val
            improved = True
    return current, fcur, evals

# ==========================
# Loop do Algoritmo Genético
# ==========================
def genetic_optimize(seed: int = 42,
                     logfile: str = "avaliacoes_log.csv") -> Dict[str, Any]:
    random.seed(seed)
    t0 = time.time()
    # população inicial estratificada na categoria
    pop = []
    per_cat = POP_SIZE // len(CATEGORIES)
    for c in CATEGORIES:
        for _ in range(per_cat):
            ind = random_individual()
            ind["cat"] = c
            pop.append(ind)
    while len(pop) < POP_SIZE:
        pop.append(random_individual())

    # logging
    log_fields = ["timestamp","generation","eval_id","cat"] + [f"x{i+1}" for i in range(N_INT_VARS)] + ["objective","phase"]
    eval_id = 0
    best_overall = None
    best_value = -1e300
    gens_no_improve = 0

    with open(logfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        for gen in range(GENERATIONS):
            # avaliar população
            pop_fit: List[Tuple[Dict[str, Any], float]] = []
            for ind in pop:
                val = evaluate_model(ind)
                pop_fit.append((ind, val))
                writer.writerow({
                    "timestamp": time.time(),
                    "generation": gen,
                    "eval_id": eval_id,
                    "cat": ind["cat"],
                    **{f"x{i+1}": ind["ints"][i] for i in range(N_INT_VARS)},
                    "objective": val,
                    "phase": "GA"
                })
                eval_id += 1
            # atualizar melhor global
            gen_best_ind, gen_best_val = max(pop_fit, key=lambda x: x[1])
            if gen_best_val > best_value:
                best_value = gen_best_val
                best_overall = {"cat": gen_best_ind["cat"], "ints": gen_best_ind["ints"][:]}
                gens_no_improve = 0
            else:
                gens_no_improve += 1

            # intensificação local em top-k
            pop_fit_sorted = sorted(pop_fit, key=lambda x: x[1], reverse=True)
            k_local = min(LOCAL_REFINES_PER_GEN, len(pop_fit_sorted))
            for i in range(k_local):
                start_ind = pop_fit_sorted[i][0]
                start_val = pop_fit_sorted[i][1]
                loc_best, loc_val, extra_evals = local_pattern_search(start_ind, start_val, LOCAL_REFINE_BUDGET)
                # log das avaliações locais é feito dentro do PS; aqui só registramos o ponto final
                writer.writerow({
                    "timestamp": time.time(),
                    "generation": gen,
                    "eval_id": eval_id,
                    "cat": loc_best["cat"],
                    **{f"x{i+1}": loc_best["ints"][i] for i in range(N_INT_VARS)},
                    "objective": loc_val,
                    "phase": "PS_end"
                })
                eval_id += 1
                if loc_val > best_value:
                    best_value = loc_val
                    best_overall = {"cat": loc_best["cat"], "ints": loc_best["ints"][:]}

            # critério de parada antecipada
            if gens_no_improve >= NO_IMPROVE_STOP:
                break

            # seleção por elitismo
            elites = [ {"cat": ind["cat"], "ints": ind["ints"][:]} for ind, _ in pop_fit_sorted[:ELITISM] ]
            # reprodução
            new_pop = elites[:]
            while len(new_pop) < POP_SIZE:
                p1 = tournament_select(pop_fit)
                p2 = tournament_select(pop_fit)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < POP_SIZE:
                    new_pop.append(c2)
            pop = new_pop

    runtime = time.time() - t0
    return {
        "best": best_overall,
        "best_value": best_value,
        "runtime_sec": runtime,
        "logfile": logfile
    }

# ==========================
# Execução principal
# ==========================
if __name__ == "__main__":
    # Para usar o executável: defina USE_SUBPROCESS=True e garanta que modelo10.exe está no PATH.
    # Parâmetros podem ser ajustados via flags/ambiente conforme necessário.
    result = genetic_optimize(seed=123, logfile="avaliacoes_log.csv")
    print("Melhor configuração encontrada:")
    print(result["best"])
    print("Melhor valor:", result["best_value"])
    print("Tempo (s):", round(result["runtime_sec"], 3))
    print("Log salvo em:", result["logfile"])

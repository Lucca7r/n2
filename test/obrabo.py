import random
import time
import csv
import math
import subprocess
from typing import List, Tuple, Dict, Any, Optional
import sys

# ==========================
# Configurações do problema
# ==========================

# Aqui você define o "schema" das variáveis de entrada do executável.
# Tipos suportados:
#   - "int": variável inteira com min/max
#   - "cat": variável categórica com uma lista de strings possíveis
#
# EXEMPLO 1: 5 variáveis inteiras (comportamento similar ao seu script atual)
#
# VARIABLES = [
#     {"name": "x1", "type": "int", "min": 1, "max": 100},
#     {"name": "x2", "type": "int", "min": 1, "max": 100},
#     {"name": "x3", "type": "int", "min": 1, "max": 100},
#     {"name": "x4", "type": "int", "min": 1, "max": 100},
#     {"name": "x5", "type": "int", "min": 1, "max": 100},
# ]
#
# EXEMPLO 2: 1 categórica + 4 inteiras (x1 = "baixo/medio/alto")
# Descomente para usar:

VARIABLES = [
    #{"name": "x1", "type": "cat", "values": ["baixo", "medio", "alto"]},
    {"name": "x2", "type": "int", "min": 1, "max": 100},
    {"name": "x3", "type": "int", "min": 1, "max": 100},
    {"name": "x4", "type": "int", "min": 1, "max": 100},
    {"name": "x5", "type": "int", "min": 1, "max": 100},
    {"name": "x6", "type": "int", "min": 1, "max": 100},
    # {"name": "x7", "type": "int", "min": 1, "max": 100},
    # {"name": "x8", "type": "int", "min": 1, "max": 100},
    # {"name": "x9", "type": "int", "min": 1, "max": 100},
    # {"name": "x10", "type": "int", "min": 1, "max": 100},
    
]

N_VARS = len(VARIABLES)

# Passos discretos para mutação / vizinhança de inteiros
MUTATION_CREEP_STEPS = [-5, -2, -1, 1, 2, 5]

# ==========================
# Configurações de GA
# ==========================

POP_SIZE = 80
GENERATIONS = 30
ELITISM = 8
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE_INT = 0.5  # probabilidade de mutar cada gene

# ==========================
# Configurações do Híbrido
# ==========================

LOCAL_REFINES_PER_GEN = 5
LOCAL_REFINE_BUDGET = 30
NO_IMPROVE_STOP = 8

# ==========================
# Configurações de PSO
# ==========================

PSO_PARTICLES = 25
PSO_ITERATIONS_GLOBAL = 30
PSO_ITERATIONS_LOCAL = 5
PSO_INERTIA = 0.7
PSO_COGNITIVE = 1.4
PSO_SOCIAL = 1.4

# ==========================
# Timeout (por modo)
# ==========================

TIMEOUT_SEGUNDOS = 600

# ==========================
# Variáveis globais
# ==========================

START_TIME = None
TIMEOUT_GLOBAL = None
TIMEOUT_REACHED = False
EVAL_ID_COUNTER = 0
# Caminho do executável 
# caminhos C:/Users/adm/Documents/faculdade/2025_2/pesquisa_op/n2 ==== 
# C:/Users/aluno/Downloads/n2-main/n2-main/simulado.exe
MODELO_EXECUTAVEL = "C:/Users/adm/Documents/faculdade/2025_2/pesquisa_op/n2/model/simulado.exe"
USE_SUBPROCESS = True  # False para modo de simulação interna

# ==========================
# Utilidades de Log e Timeout
# ==========================

def check_timeout() -> bool:
    global TIMEOUT_REACHED, START_TIME, TIMEOUT_GLOBAL
    if START_TIME is not None and TIMEOUT_GLOBAL is not None:
        elapsed = time.time() - START_TIME
        if elapsed >= TIMEOUT_GLOBAL:
            TIMEOUT_REACHED = True
            return True
    return False

def set_timeout(seconds: int):
    global START_TIME, TIMEOUT_GLOBAL, TIMEOUT_REACHED
    START_TIME = time.time()
    TIMEOUT_GLOBAL = seconds
    TIMEOUT_REACHED = False
    print(f"⏱️ Orçamento de tempo: {seconds}s")

def get_elapsed_time() -> float:
    if START_TIME is not None:
        return time.time() - START_TIME
    return 0.0

def log_monitor(eval_id: int, phase: str, genes: List[int], val: float):
    """Imprime cada teste no terminal em tempo real."""
    print(f" > [{eval_id:04d}] {phase:12s} x={genes} -> Obj: {val:.4f}")

# ==========================
# Utilidades de genes
# ==========================

def random_individual() -> Dict[str, Any]:
    """Cria um indivíduo aleatório respeitando o schema VARIABLES."""
    genes: List[int] = []
    for var in VARIABLES:
        if var["type"] == "int":
            g = random.randint(var["min"], var["max"])
        elif var["type"] == "cat":
            g = random.randint(0, len(var["values"]) - 1)
        else:
            raise ValueError(f"Tipo de variável desconhecido: {var['type']}")
        genes.append(g)
    return {"genes": genes}

def clamp_gene(g: int, var: Dict[str, Any]) -> int:
    """Garante que o gene fique no domínio permitido (int ou cat)."""
    if var["type"] == "int":
        return max(var["min"], min(var["max"], g))
    elif var["type"] == "cat":
        return max(0, min(len(var["values"]) - 1, g))
    else:
        raise ValueError(f"Tipo de variável desconhecido: {var['type']}")

def mutate(ind: Dict[str, Any]) -> Dict[str, Any]:
    """Mutação por gene, tratando int e cat separadamente."""
    new_ind = {"genes": ind["genes"][:]}
    for i, var in enumerate(VARIABLES):
        if random.random() < MUTATION_RATE_INT:
            if var["type"] == "int":
                if random.random() < 0.6:
                    step = random.choice(MUTATION_CREEP_STEPS)
                    new_g = clamp_gene(new_ind["genes"][i] + step, var)
                else:
                    new_g = random.randint(var["min"], var["max"])
            else:  # categórica
                k = len(var["values"])
                if k <= 1:
                    new_g = 0
                else:
                    choices = [j for j in range(k) if j != new_ind["genes"][i]]
                    new_g = random.choice(choices)
            new_ind["genes"][i] = new_g
    return new_ind

def crossover(p1: Dict[str, Any], p2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Crossover de um ponto para o vetor de genes inteiro."""
    if random.random() > CROSSOVER_RATE:
        return {"genes": p1["genes"][:]}, {"genes": p2["genes"][:]}
    point = random.randint(1, N_VARS - 1)
    g1 = p1["genes"][:point] + p2["genes"][point:]
    g2 = p2["genes"][:point] + p1["genes"][point:]
    return {"genes": g1}, {"genes": g2}

def tournament_select(pop_with_fit: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
    """Seleção por torneio."""
    contenders = random.sample(pop_with_fit, TOURNAMENT_K)
    champion = max(contenders, key=lambda x: x[1])
    return {"genes": champion[0]["genes"][:]}

# ==========================
# Codificação p/ chamar o .exe
# ==========================

def decode_for_exe(ind: Dict[str, Any]) -> List[str]:
    """
    Converte o indivíduo interno (genes inteiros) para a lista de
    strings de argumentos que serão passados ao executável.
    """
    args: List[str] = []
    for g, var in zip(ind["genes"], VARIABLES):
        if var["type"] == "int":
            args.append(str(g))
        elif var["type"] == "cat":
            # índice -> string ("baixo", "medio", "alto", ...)
            args.append(var["values"][g])
        else:
            raise ValueError(f"Tipo de variável desconhecido: {var['type']}")
    return args

# ==========================
# Avaliação
# ==========================

def evaluate_model(ind: Dict[str, Any]) -> float:
    global EVAL_ID_COUNTER
    if check_timeout():
        return -1e12

    if USE_SUBPROCESS:
        args = [MODELO_EXECUTAVEL] + decode_for_exe(ind)
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=5
            )

            # Se o .exe retornou erro, penaliza
            if result.returncode != 0:
                return -1e12

            # Pega primeira linha não vazia da saída
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            if not lines:
                return -1e12

            first = lines[0]  # ex: "Valor de saída: 0.77"

            # Se tiver texto antes, como "Valor de saída: 0.77",
            # pega só a parte depois dos dois-pontos
            if ":" in first:
                first = first.split(":", 1)[1].strip()

            # Se algum dia vier com vírgula decimal, troca por ponto
            first = first.replace(",", ".")

            val = float(first)
            EVAL_ID_COUNTER += 1
            return val

        except Exception as e:
            # Opcional: debug
            print("DEBUG exception em evaluate_model:", repr(e))
            return -1e12

    else:
        # Simulação para teste sem .exe
        time.sleep(0.002)
        val = sum(ind["genes"]) + random.uniform(-5, 5)
        EVAL_ID_COUNTER += 1
        return val

# ==========================
# Buscas Locais (PS + PSO Refine)
# ==========================

def neighborhood(ind: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gera vizinhança: para int usa passos discretos, para cat muda o índice."""
    neigh: List[Dict[str, Any]] = []
    base_genes = ind["genes"]
    for i, var in enumerate(VARIABLES):
        if var["type"] == "int":
            for step in MUTATION_CREEP_STEPS:
                newv = clamp_gene(base_genes[i] + step, var)
                if newv != base_genes[i]:
                    new_genes = base_genes[:]
                    new_genes[i] = newv
                    neigh.append({"genes": new_genes})
        else:  # categórica
            k = len(var["values"])
            for new_idx in range(k):
                if new_idx != base_genes[i]:
                    new_genes = base_genes[:]
                    new_genes[i] = new_idx
                    neigh.append({"genes": new_genes})
    return neigh

def local_pattern_search(
    start_ind: Dict[str, Any],
    start_val: float,
    budget: int,
    logfile_writer: Optional[csv.DictWriter] = None
) -> Tuple[Dict[str, Any], float, int]:
    """Busca local em vizinhança (best improvement)."""
    current = {"genes": start_ind["genes"][:]}
    fcur = start_val
    evals = 0
    improved = True

    while improved and evals < budget and not check_timeout():
        improved = False
        best_nb = current
        best_val = fcur

        for nb in neighborhood(current):
            if evals >= budget or check_timeout():
                break
            fnb = evaluate_model(nb)
            evals += 1

            # LOG NO TERMINAL
            log_monitor(EVAL_ID_COUNTER, "PatSearch", nb["genes"], fnb)

            if logfile_writer:
                row = {
                    "timestamp": time.time(),
                    "eval_id": EVAL_ID_COUNTER,
                    "phase": "PS",
                    "cat": "MIX",
                    "objective": fnb,
                }
                for j in range(N_VARS):
                    row[f"x{j+1}"] = nb["genes"][j]
                logfile_writer.writerow(row)

            if fnb > best_val:
                best_nb, best_val = nb, fnb

        if best_val > fcur:
            current, fcur = best_nb, best_val
            improved = True

    return current, fcur, evals

def swarm_refine(
    start_ind: Dict[str, Any],
    start_val: float,
    n_particles: int,
    iterations: int,
    evaluations_limit: int,
    logfile_writer: Optional[csv.DictWriter] = None
) -> Tuple[Dict[str, Any], float, int]:
    """Pequeno PSO local para refinar ao redor de um indivíduo."""
    global EVAL_ID_COUNTER

    swarm: List[Dict[str, Any]] = []
    velocities: List[List[float]] = []
    evals = 0

    # Inicialização do enxame ao redor do start_ind
    for _ in range(n_particles):
        part = {"genes": start_ind["genes"][:]}
        for d, var in enumerate(VARIABLES):
            if random.random() < 0.6:
                if var["type"] == "int":
                    part["genes"][d] = clamp_gene(
                        part["genes"][d] + random.choice(MUTATION_CREEP_STEPS),
                        var
                    )
                else:  # categórica: pequena perturbação = outro índice aleatório
                    k = len(var["values"])
                    if k > 1:
                        choices = [j for j in range(k) if j != part["genes"][d]]
                        part["genes"][d] = random.choice(choices)
        swarm.append(part)
        velocities.append([random.uniform(-3.0, 3.0) for _ in range(N_VARS)])

    # pbest
    pbest = [{"genes": p["genes"][:]} for p in swarm]
    pbest_val: List[float] = []

    # Avaliação inicial
    for i in range(len(pbest)):
        if evals >= evaluations_limit or check_timeout():
            break
        val = evaluate_model(pbest[i])
        pbest_val.append(val)
        evals += 1

        # LOG NO TERMINAL
        log_monitor(EVAL_ID_COUNTER, "PSO_Ref_Ini", pbest[i]["genes"], val)

        if logfile_writer:
            row = {
                "timestamp": time.time(),
                "eval_id": EVAL_ID_COUNTER,
                "phase": "PSO_Ref_Init",
                "cat": "MIX",
                "objective": val,
            }
            for j in range(N_VARS):
                row[f"x{j+1}"] = pbest[i]["genes"][j]
            logfile_writer.writerow(row)

    if not pbest_val:
        return start_ind, start_val, evals

    best_idx = max(range(len(pbest_val)), key=lambda i: pbest_val[i])
    gbest = {"genes": pbest[best_idx]["genes"][:]}
    gbest_val = pbest_val[best_idx]

    w, c1, c2 = PSO_INERTIA, PSO_COGNITIVE, PSO_SOCIAL

    for it in range(iterations):
        if evals >= evaluations_limit or check_timeout():
            break

        for i in range(n_particles):
            if i >= len(pbest_val) or evals >= evaluations_limit or check_timeout():
                continue

            # Atualizar velocidade e posição
            for d, var in enumerate(VARIABLES):
                r1, r2 = random.random(), random.random()
                cog = c1 * r1 * (pbest[i]["genes"][d] - swarm[i]["genes"][d])
                soc = c2 * r2 * (gbest["genes"][d] - swarm[i]["genes"][d])
                velocities[i][d] = w * velocities[i][d] + cog + soc
                newv = swarm[i]["genes"][d] + int(round(velocities[i][d]))
                swarm[i]["genes"][d] = clamp_gene(newv, var)

            val = evaluate_model(swarm[i])
            evals += 1

            # LOG NO TERMINAL
            log_monitor(EVAL_ID_COUNTER, "PSO_Refine", swarm[i]["genes"], val)

            if logfile_writer:
                row = {
                    "timestamp": time.time(),
                    "eval_id": EVAL_ID_COUNTER,
                    "phase": "PSO_Refine",
                    "cat": "MIX",
                    "objective": val,
                }
                for j in range(N_VARS):
                    row[f"x{j+1}"] = swarm[i]["genes"][j]
                logfile_writer.writerow(row)

            if val > pbest_val[i]:
                pbest[i] = {"genes": swarm[i]["genes"][:]}
                pbest_val[i] = val

            if val > gbest_val:
                gbest = {"genes": pbest[i]["genes"][:]}
                gbest_val = val

    return gbest, gbest_val, evals

# ==========================
# 1) HÍBRIDO (GA + PS + PSO)
# ==========================

def genetic_hybrid(seed: int = 42, logfile: str = "avaliacoes_hybrid_mix.csv") -> Dict[str, Any]:
    global EVAL_ID_COUNTER
    random.seed(seed)
    t0 = time.time()

    pop = [random_individual() for _ in range(POP_SIZE)]
    best_overall, best_value = None, -1e300
    gens_no_improve = 0

    log_fields = ["timestamp", "eval_id", "phase", "cat"] \
                 + [f"x{i+1}" for i in range(N_VARS)] + ["objective"]

    with open(logfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        for gen in range(GENERATIONS):
            if check_timeout():
                break

            print(f"\n--- GERAÇÃO {gen+1}/{GENERATIONS} (HÍBRIDO) ---")

            pop_fit: List[Tuple[Dict[str, Any], float]] = []
            for ind in pop:
                val = evaluate_model(ind)
                pop_fit.append((ind, val))

                # LOG NO TERMINAL
                log_monitor(EVAL_ID_COUNTER, "GA_Hybrid", ind["genes"], val)

                row = {
                    "timestamp": time.time(),
                    "eval_id": EVAL_ID_COUNTER,
                    "phase": "GA",
                    "cat": "MIX",
                    "objective": val,
                }
                for i in range(N_VARS):
                    row[f"x{i+1}"] = ind["genes"][i]
                writer.writerow(row)

            if check_timeout() or not pop_fit:
                break

            gen_best_ind, gen_best_val = max(pop_fit, key=lambda x: x[1])

            if gen_best_val > best_value:
                best_value = gen_best_val
                best_overall = {"genes": gen_best_ind["genes"][:]}
                gens_no_improve = 0
            else:
                gens_no_improve += 1

            # Intensificação Local
            print(f"--- REFINAMENTO LOCAL (Top {LOCAL_REFINES_PER_GEN}) ---")
            pop_fit_sorted = sorted(pop_fit, key=lambda x: x[1], reverse=True)
            k_local = min(LOCAL_REFINES_PER_GEN, len(pop_fit_sorted))

            for i in range(k_local):
                if check_timeout():
                    break
                start_ind, start_val = pop_fit_sorted[i]

                loc_best, loc_val, _ = local_pattern_search(
                    start_ind, start_val, LOCAL_REFINE_BUDGET, writer
                )
                sw_best, sw_val, _ = swarm_refine(
                    loc_best, loc_val, PSO_PARTICLES, PSO_ITERATIONS_LOCAL,
                    LOCAL_REFINE_BUDGET * 2, writer
                )

                if sw_val > loc_val:
                    chosen_best, chosen_val = sw_best, sw_val
                    label = "PSO_End"
                else:
                    chosen_best, chosen_val = loc_best, loc_val
                    label = "PS_End"

                # Log do melhor escolhido (não conta avaliação nova)
                row = {
                    "timestamp": time.time(),
                    "eval_id": EVAL_ID_COUNTER,
                    "phase": label,
                    "cat": "MIX",
                    "objective": chosen_val,
                }
                for j in range(N_VARS):
                    row[f"x{j+1}"] = chosen_best["genes"][j]
                writer.writerow(row)

                if chosen_val > best_value:
                    best_value, best_overall = chosen_val, {"genes": chosen_best["genes"][:]}
                    gens_no_improve = 0

            if gens_no_improve >= NO_IMPROVE_STOP:
                print(">> Parada antecipada por falta de melhoria.")
                break

            # Reprodução (GA)
            pop_fit_sorted = sorted(pop_fit, key=lambda x: x[1], reverse=True)
            elites = [{"genes": ind["genes"][:]} for ind, _ in pop_fit_sorted[:ELITISM]]
            new_pop = elites[:]

            while len(new_pop) < POP_SIZE:
                c1, c2 = crossover(
                    tournament_select(pop_fit),
                    tournament_select(pop_fit)
                )
                new_pop.append(mutate(c1))
                if len(new_pop) < POP_SIZE:
                    new_pop.append(mutate(c2))

            pop = new_pop

    return {
        "best": best_overall,
        "best_value": best_value,
        "runtime_sec": time.time() - t0,
        "logfile": logfile,
    }

# ==========================
# 2) GA PURO
# ==========================

def genetic_pure(seed: int = 42, logfile: str = "avaliacoes_ga_puro_mix.csv") -> Dict[str, Any]:
    global EVAL_ID_COUNTER
    random.seed(seed)
    t0 = time.time()

    pop = [random_individual() for _ in range(POP_SIZE)]
    best_overall, best_value = None, -1e300
    gens_no_improve = 0

    log_fields = ["timestamp", "eval_id", "phase", "cat"] \
                 + [f"x{i+1}" for i in range(N_VARS)] + ["objective"]

    with open(logfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        for gen in range(GENERATIONS):
            if check_timeout():
                break

            print(f"\n--- GERAÇÃO {gen+1}/{GENERATIONS} (GA PURO) ---")

            pop_fit: List[Tuple[Dict[str, Any], float]] = []
            for ind in pop:
                val = evaluate_model(ind)
                pop_fit.append((ind, val))

                # LOG NO TERMINAL
                log_monitor(EVAL_ID_COUNTER, "GA_Pure", ind["genes"], val)

                row = {
                    "timestamp": time.time(),
                    "eval_id": EVAL_ID_COUNTER,
                    "phase": "GA_Pure",
                    "cat": "MIX",
                    "objective": val,
                }
                for i in range(N_VARS):
                    row[f"x{i+1}"] = ind["genes"][i]
                writer.writerow(row)

            if check_timeout() or not pop_fit:
                break

            gen_best_ind, gen_best_val = max(pop_fit, key=lambda x: x[1])

            if gen_best_val > best_value:
                best_value = gen_best_val
                best_overall = {"genes": gen_best_ind["genes"][:]}
                gens_no_improve = 0
            else:
                gens_no_improve += 1

            if gens_no_improve >= NO_IMPROVE_STOP:
                print(">> Parada antecipada por falta de melhoria.")
                break

            # Reprodução
            pop_fit_sorted = sorted(pop_fit, key=lambda x: x[1], reverse=True)
            elites = [{"genes": ind["genes"][:]} for ind, _ in pop_fit_sorted[:ELITISM]]
            new_pop = elites[:]

            while len(new_pop) < POP_SIZE:
                c1, c2 = crossover(
                    tournament_select(pop_fit),
                    tournament_select(pop_fit)
                )
                new_pop.append(mutate(c1))
                if len(new_pop) < POP_SIZE:
                    new_pop.append(mutate(c2))

            pop = new_pop

    return {
        "best": best_overall,
        "best_value": best_value,
        "runtime_sec": time.time() - t0,
        "logfile": logfile,
    }

# ==========================
# 3) PSO PURO
# ==========================

def pso_optimize(
    seed: int = 42,
    n_particles: int = PSO_PARTICLES,
    iterations: int = PSO_ITERATIONS_GLOBAL,
    logfile: str = "avaliacoes_pso_puro_mix.csv"
) -> Dict[str, Any]:
    global EVAL_ID_COUNTER
    random.seed(seed)
    t0 = time.time()

    particles: List[Dict[str, Any]] = []

    log_fields = ["timestamp", "eval_id", "phase", "cat"] \
                 + [f"x{i+1}" for i in range(N_VARS)] + ["objective"]

    with open(logfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        print("\n--- PSO INICIALIZAÇÃO ---")

        # Inicialização
        for _ in range(n_particles):
            if check_timeout():
                break

            ind = random_individual()
            val = evaluate_model(ind)

            # LOG NO TERMINAL
            log_monitor(EVAL_ID_COUNTER, "PSO_Init", ind["genes"], val)

            particle = {
                "pos": ind,  # {"genes": [...]}
                "vel": {"genes": [0.0] * N_VARS},
                "best_pos": {"genes": ind["genes"][:]},
                "best_val": val,
            }
            particles.append(particle)

            row = {
                "timestamp": time.time(),
                "eval_id": EVAL_ID_COUNTER,
                "phase": "PSO_Init",
                "cat": "MIX",
                "objective": val,
            }
            for i in range(N_VARS):
                row[f"x{i+1}"] = ind["genes"][i]
            writer.writerow(row)

        if not particles:
            return {
                "best": None,
                "best_value": -1e300,
                "runtime_sec": get_elapsed_time(),
                "logfile": logfile,
            }

        # Melhor global inicial
        best_particle = max(particles, key=lambda p: p["best_val"])
        gbest_pos = {"genes": best_particle["best_pos"]["genes"][:]}
        gbest_val = best_particle["best_val"]

        # Loop de iterações
        for it in range(iterations):
            if check_timeout():
                break

            print(f"\n--- PSO ITERAÇÃO {it+1}/{iterations} ---")

            for p in particles:
                if check_timeout():
                    break

                for d, var in enumerate(VARIABLES):
                    r1, r2 = random.random(), random.random()
                    cog = PSO_COGNITIVE * r1 * (p["best_pos"]["genes"][d] - p["pos"]["genes"][d])
                    soc = PSO_SOCIAL * r2 * (gbest_pos["genes"][d] - p["pos"]["genes"][d])
                    p["vel"]["genes"][d] = PSO_INERTIA * p["vel"]["genes"][d] + cog + soc
                    newv = p["pos"]["genes"][d] + int(round(p["vel"]["genes"][d]))
                    p["pos"]["genes"][d] = clamp_gene(newv, var)

                val = evaluate_model(p["pos"])

                # LOG NO TERMINAL
                log_monitor(EVAL_ID_COUNTER, "PSO_Iter", p["pos"]["genes"], val)

                row = {
                    "timestamp": time.time(),
                    "eval_id": EVAL_ID_COUNTER,
                    "phase": "PSO",
                    "cat": "MIX",
                    "objective": val,
                }
                for j in range(N_VARS):
                    row[f"x{j+1}"] = p["pos"]["genes"][j]
                writer.writerow(row)

                if val > p["best_val"]:
                    p["best_val"] = val
                    p["best_pos"] = {"genes": p["pos"]["genes"][:]}

                if val > gbest_val:
                    gbest_val = val
                    gbest_pos = {"genes": p["pos"]["genes"][:]}

    return {
        "best": gbest_pos,
        "best_value": gbest_val,
        "runtime_sec": time.time() - t0,
        "logfile": logfile,
    }

# ==========================
# MENU PRINCIPAL
# ==========================

if __name__ == "__main__":
    random.seed(123)

    print("\n" + "=" * 60)
    print("SELETOR DE MODO (MONITORAMENTO EM TEMPO REAL)")
    print("=" * 60)
    print("1 - Modo HÍBRIDO (GA + PS + PSO)")
    print("2 - Modo GA PURO")
    print("3 - Modo PSO PURO")

    escolha = input("\n>> Escolha: ").strip()
    print("\n" + "-" * 60)

    if escolha == "1":
        set_timeout(TIMEOUT_SEGUNDOS)
        try:
            res = genetic_hybrid(seed=123, logfile="avaliacoes_hybrid_mix.csv")
            print(f"\n✅ FIM. Melhor HÍBRIDO: {res['best_value']:.4f}")
            print("Melhor configuração (genes internos):", res["best"]["genes"])
        except Exception as e:
            print(f"❌ Erro: {e}")

    elif escolha == "2":
        set_timeout(TIMEOUT_SEGUNDOS)
        try:
            res = genetic_pure(seed=123, logfile="avaliacoes_ga_puro_mix.csv")
            print(f"\n✅ FIM. Melhor GA PURO: {res['best_value']:.4f}")
            print("Melhor configuração (genes internos):", res["best"]["genes"])
        except Exception as e:
            print(f"❌ Erro: {e}")

    elif escolha == "3":
        set_timeout(TIMEOUT_SEGUNDOS)
        try:
            res = pso_optimize(
                seed=123,
                n_particles=PSO_PARTICLES,
                iterations=PSO_ITERATIONS_GLOBAL,
                logfile="avaliacoes_pso_puro_mix.csv"
            )
            print(f"\n✅ FIM. Melhor PSO PURO: {res['best_value']:.4f}")
            print("Melhor configuração (genes internos):", res["best"]["genes"])
        except Exception as e:
            print(f"❌ Erro: {e}")

    else:
        print("Opção inválida.")

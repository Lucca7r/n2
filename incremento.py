# run_all_modes.py
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
MUTATION_RATE_INT = 0.5
MUTATION_RATE_CAT = 0.5
MUTATION_CREEP_STEPS = [-5, -2, -1, 1, 2, 5]

LOCAL_REFINES_PER_GEN = 5      # refinamentos locais por geração
LOCAL_REFINE_BUDGET = 30       # avaliações máximas por refino local
NO_IMPROVE_STOP = 8            # gerações sem melhora

# PSO defaults (ajuste conforme desejar)
PSO_PARTICLES = 25
PSO_ITERATIONS = 30
PSO_INERTIA = 0.7
PSO_COGNITIVE = 1.4
PSO_SOCIAL = 1.4
PSO_CAT_INJECT = 0.1  # probabilidade de tentar alterar categoria discretamente

# Caminho/uso do simulador
MODELO_EXECUTAVEL = "C:/Users/aluno/Downloads/n2-main/modelo10.exe"  # nome do executável fornecido
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
    Se USE_SUBPROCESS=True, o executável deve aceitar: modelo10.exe <cat> <x1> <x2> ...
    e a primeira linha da stdout deve conter o valor-objetivo (float).
    """
    if USE_SUBPROCESS:
            args = [MODELO_EXECUTAVEL, ind["cat"]] + list(map(str, ind["ints"]))
            print(">> Rodando:", args)  # DEBUG

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True
        )

        print("STDOUT:", repr(result.stdout))
        print("STDERR:", repr(result.stderr))
        print("Return code:", result.returncode)

        if result.returncode != 0:
            print("ERRO: Executável retornou código != 0")
            return -1e12

        value_str = result.stdout.strip().splitlines()[0]
        return float(value_str)

    except Exception as e:
        print("EXCEPTION:", e)
        return -1e12

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
    current, fcur = {"cat": start_ind["cat"], "ints": start_ind["ints"][:]}, start_val
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
# PSO (swarm) - função reutilizável
# ==========================
def pso_optimize(seed: int = 42,
                 n_particles: int = PSO_PARTICLES,
                 iterations: int = PSO_ITERATIONS,
                 logfile: str = "avaliacoes_pso.csv",
                 logfile_start_eval_id: int = 0) -> Dict[str, Any]:
    """
    PSO isolado (puro). Retorna dict com best, best_value, runtime_sec, logfile, last_eval_id.
    Faz log em CSV com phase='PSO'.
    """
    random.seed(seed)
    t0 = time.time()

    # inicializa partículas aleatórias
    particles: List[Dict[str, Any]] = []
    for _ in range(n_particles):
        ind = random_individual()
        val = evaluate_model(ind)
        particles.append({
            "pos": ind,
            "vel": {"ints": [0.0]*N_INT_VARS, "cat": 0},
            "best_pos": {"cat": ind["cat"], "ints": ind["ints"][:] },
            "best_val": val
        })

    # logging CSV
    log_fields = ["timestamp","eval_id","phase","cat"] + [f"x{i+1}" for i in range(N_INT_VARS)] + ["objective"]
    eval_id = logfile_start_eval_id
    with open(logfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        # escreve as avaliações iniciais
        for p in particles:
            writer.writerow({
                "timestamp": time.time(),
                "eval_id": eval_id,
                "phase": "PSO",
                "cat": p["pos"]["cat"],
                **{f"x{i+1}": p["pos"]["ints"][i] for i in range(N_INT_VARS)},
                "objective": p["best_val"]
            })
            eval_id += 1

        # melhor global
        best_particle = max(particles, key=lambda p: p["best_val"])
        gbest_pos = {"cat": best_particle["best_pos"]["cat"], "ints": best_particle["best_pos"]["ints"][:]}
        gbest_val = best_particle["best_val"]

        # iterações
        for it in range(iterations):
            for i, p in enumerate(particles):
                # atualizar velocidade dos inteiros
                for d in range(N_INT_VARS):
                    r1 = random.random()
                    r2 = random.random()
                    cognitive = PSO_COGNITIVE * r1 * (p["best_pos"]["ints"][d] - p["pos"]["ints"][d])
                    social = PSO_SOCIAL * r2 * (gbest_pos["ints"][d] - p["pos"]["ints"][d])
                    p["vel"]["ints"][d] = PSO_INERTIA * p["vel"]["ints"][d] + cognitive + social

                # atualizar posição (inteiros)
                for d in range(N_INT_VARS):
                    newv = p["pos"]["ints"][d] + int(round(p["vel"]["ints"][d]))
                    p["pos"]["ints"][d] = clamp(newv, INT_MIN, INT_MAX)

                # categoria (discreto): adotar pbest/gbest ou mutar com pequena prob
                if random.random() < 0.3:
                    if random.random() < 0.5:
                        p["pos"]["cat"] = p["best_pos"]["cat"]
                    else:
                        p["pos"]["cat"] = gbest_pos["cat"]
                else:
                    if random.random() < 0.05:
                        choices = [c for c in CATEGORIES if c != p["pos"]["cat"]]
                        p["pos"]["cat"] = random.choice(choices)

                # avaliar
                val = evaluate_model(p["pos"])
                writer.writerow({
                    "timestamp": time.time(),
                    "eval_id": eval_id,
                    "phase": "PSO",
                    "cat": p["pos"]["cat"],
                    **{f"x{i+1}": p["pos"]["ints"][i] for i in range(N_INT_VARS)},
                    "objective": val
                })
                eval_id += 1

                # atualizar pbest / gbest
                if val > p["best_val"]:
                    p["best_val"] = val
                    p["best_pos"] = {"cat": p["pos"]["cat"], "ints": p["pos"]["ints"][:] }
                if val > gbest_val:
                    gbest_val = val
                    gbest_pos = {"cat": p["pos"]["cat"], "ints": p["pos"]["ints"][:]}

        runtime = time.time() - t0
    return {"best": gbest_pos, "best_value": gbest_val, "runtime_sec": runtime, "logfile": logfile, "last_eval_id": eval_id}

# ==========================
# Swarm-refine (PSO usado dentro do GA hybrid)
# ==========================
def swarm_refine(start_ind: Dict[str, Any],
                 start_val: float,
                 n_particles: int = PSO_PARTICLES,
                 iterations: int = PSO_ITERATIONS,
                 evaluations_limit: int = 200,
                 logfile_writer: Optional[csv.DictWriter] = None,
                 eval_id_start: int = 0,
                 generation: int = 0) -> Tuple[Dict[str, Any], float, int]:
    """
    PSO local/global inicializado ao redor de start_ind.
    Retorna (best_individual, best_value, evals_done).
    Se logfile_writer fornecido, escreve linhas com phase='PSO'.
    """
    swarm: List[Dict[str, Any]] = []
    velocities: List[List[float]] = []

    for _ in range(n_particles):
        part = {"cat": start_ind["cat"], "ints": start_ind["ints"][:]}
        # pequena perturbação inicial
        for d in range(N_INT_VARS):
            if random.random() < 0.6:
                part["ints"][d] = clamp(part["ints"][d] + random.choice(MUTATION_CREEP_STEPS), INT_MIN, INT_MAX)
            else:
                if random.random() < 0.05:
                    part["ints"][d] = random.randint(INT_MIN, INT_MAX)
        if random.random() < 0.2:
            choices = [c for c in CATEGORIES if c != part["cat"]]
            part["cat"] = random.choice(choices)

        swarm.append(part)
        velocities.append([random.uniform(-3.0, 3.0) for _ in range(N_INT_VARS)])

    pbest = [{"cat": p["cat"], "ints": p["ints"][:]} for p in swarm]
    pbest_val = [evaluate_model(p) for p in pbest]
    evals = len(pbest_val)

    eval_id = eval_id_start
    if logfile_writer is not None:
        for i in range(len(swarm)):
            logfile_writer.writerow({
                "timestamp": time.time(),
                "eval_id": eval_id,
                "phase": "PSO",
                "cat": swarm[i]["cat"],
                **{f"x{j+1}": swarm[i]["ints"][j] for j in range(N_INT_VARS)},
                "objective": pbest_val[i]
            })
            eval_id += 1

    best_idx = max(range(len(pbest_val)), key=lambda i: pbest_val[i])
    gbest = {"cat": pbest[best_idx]["cat"], "ints": pbest[best_idx]["ints"][:]}
    gbest_val = pbest_val[best_idx]

    w = PSO_INERTIA
    c1 = PSO_COGNITIVE
    c2 = PSO_SOCIAL

    for it in range(iterations):
        for i in range(n_particles):
            for d in range(N_INT_VARS):
                r1 = random.random()
                r2 = random.random()
                cognitive = c1 * r1 * (pbest[i]["ints"][d] - swarm[i]["ints"][d])
                social = c2 * r2 * (gbest["ints"][d] - swarm[i]["ints"][d])
                velocities[i][d] = w * velocities[i][d] + cognitive + social

            for d in range(N_INT_VARS):
                newv = swarm[i]["ints"][d] + int(round(velocities[i][d]))
                swarm[i]["ints"][d] = clamp(newv, INT_MIN, INT_MAX)

            # categoria: adota pbest/gbest ou muta
            if random.random() < 0.3:
                if random.random() < 0.5:
                    swarm[i]["cat"] = pbest[i]["cat"]
                else:
                    swarm[i]["cat"] = gbest["cat"]
            else:
                if random.random() < 0.05:
                    choices = [c for c in CATEGORIES if c != swarm[i]["cat"]]
                    swarm[i]["cat"] = random.choice(choices)

            val = evaluate_model(swarm[i])
            evals += 1

            if logfile_writer is not None:
                logfile_writer.writerow({
                    "timestamp": time.time(),
                    "eval_id": eval_id,
                    "phase": "PSO",
                    "cat": swarm[i]["cat"],
                    **{f"x{j+1}": swarm[i]["ints"][j] for j in range(N_INT_VARS)},
                    "objective": val
                })
                eval_id += 1

            if val > pbest_val[i]:
                pbest[i] = {"cat": swarm[i]["cat"], "ints": swarm[i]["ints"][:]}
                pbest_val[i] = val
                if val > gbest_val:
                    gbest = {"cat": pbest[i]["cat"], "ints": pbest[i]["ints"][:]}
                    gbest_val = val

            if evals >= evaluations_limit:
                return gbest, gbest_val, evals

    return gbest, gbest_val, evals

# ==========================
# Loop do Algoritmo Genético (Híbrido: GA -> Pattern Search -> PSO)
# ==========================
def genetic_hybrid(seed: int = 42, logfile: str = "avaliacoes_hybrid.csv") -> Dict[str, Any]:
    """
    GA com Pattern Search + PSO (aplica PSO após o Pattern Search nos top-k).
    """
    random.seed(seed)
    t0 = time.time()

    # população inicial estratificada na categoria
    pop: List[Dict[str, Any]] = []
    per_cat = POP_SIZE // len(CATEGORIES)
    for c in CATEGORIES:
        for _ in range(per_cat):
            ind = random_individual()
            ind["cat"] = c
            pop.append(ind)
    while len(pop) < POP_SIZE:
        pop.append(random_individual())

    # logging
    log_fields = ["timestamp","eval_id","phase","cat"] + [f"x{i+1}" for i in range(N_INT_VARS)] + ["objective"]
    eval_id = 0
    best_overall: Optional[Dict[str, Any]] = None
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
                    "eval_id": eval_id,
                    "phase": "GA",
                    "cat": ind["cat"],
                    **{f"x{i+1}": ind["ints"][i] for i in range(N_INT_VARS)},
                    "objective": val
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

            # intensificação local em top-k: Pattern Search seguido de PSO
            pop_fit_sorted = sorted(pop_fit, key=lambda x: x[1], reverse=True)
            k_local = min(LOCAL_REFINES_PER_GEN, len(pop_fit_sorted))
            for i in range(k_local):
                start_ind = {"cat": pop_fit_sorted[i][0]["cat"], "ints": pop_fit_sorted[i][0]["ints"][:]}
                start_val = pop_fit_sorted[i][1]

                # 1) Pattern Search
                loc_best, loc_val, extra_evals = local_pattern_search(start_ind, start_val, LOCAL_REFINE_BUDGET)
                # log do resultado do PS
                writer.writerow({
                    "timestamp": time.time(),
                    "eval_id": eval_id,
                    "phase": "PS_end",
                    "cat": loc_best["cat"],
                    **{f"x{j+1}": loc_best["ints"][j] for j in range(N_INT_VARS)},
                    "objective": loc_val
                })
                eval_id += 1

                # 2) PSO refinamento a partir do resultado do Pattern Search
                # passamos o writer para que as avaliações do PSO sejam logadas no mesmo arquivo
                sw_best, sw_val, sw_evals = swarm_refine(loc_best, loc_val,
                                                        n_particles=PSO_PARTICLES,
                                                        iterations=PSO_ITERATIONS,
                                                        evaluations_limit=LOCAL_REFINE_BUDGET * 4,
                                                        logfile_writer=writer,
                                                        eval_id_start=eval_id,
                                                        generation=gen)
                eval_id += sw_evals

                # escolher o melhor entre PS e PSO
                if sw_val > loc_val:
                    chosen_best, chosen_val = sw_best, sw_val
                    phase_label = "PSO_end"
                else:
                    chosen_best, chosen_val = loc_best, loc_val
                    phase_label = "PS_end"

                writer.writerow({
                    "timestamp": time.time(),
                    "eval_id": eval_id,
                    "phase": phase_label,
                    "cat": chosen_best["cat"],
                    **{f"x{j+1}": chosen_best["ints"][j] for j in range(N_INT_VARS)},
                    "objective": chosen_val
                })
                eval_id += 1

                if chosen_val > best_value:
                    best_value = chosen_val
                    best_overall = {"cat": chosen_best["cat"], "ints": chosen_best["ints"][:]}

            # critério de parada antecipada
            if gens_no_improve >= NO_IMPROVE_STOP:
                break

            # seleção por elitismo e reprodução (GA standard)
            elites = [ {"cat": ind["cat"], "ints": ind["ints"][:]} for ind, _ in pop_fit_sorted[:ELITISM] ]
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
    return {"best": best_overall, "best_value": best_value, "runtime_sec": runtime, "logfile": logfile}

# ==========================
# GA puro (sem Pattern Search nem PSO)
# ==========================
def genetic_pure(seed: int = 42, logfile: str = "avaliacoes_ga_puro.csv") -> Dict[str, Any]:
    random.seed(seed)
    t0 = time.time()

    pop: List[Dict[str, Any]] = []
    per_cat = POP_SIZE // len(CATEGORIES)
    for c in CATEGORIES:
        for _ in range(per_cat):
            ind = random_individual()
            ind["cat"] = c
            pop.append(ind)
    while len(pop) < POP_SIZE:
        pop.append(random_individual())

    log_fields = ["timestamp","eval_id","phase","cat"] + [f"x{i+1}" for i in range(N_INT_VARS)] + ["objective"]
    eval_id = 0
    best_overall: Optional[Dict[str, Any]] = None
    best_value = -1e300
    gens_no_improve = 0

    with open(logfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        for gen in range(GENERATIONS):
            pop_fit: List[Tuple[Dict[str, Any], float]] = []
            for ind in pop:
                val = evaluate_model(ind)
                pop_fit.append((ind, val))
                writer.writerow({
                    "timestamp": time.time(),
                    "eval_id": eval_id,
                    "phase": "GA",
                    "cat": ind["cat"],
                    **{f"x{i+1}": ind["ints"][i] for i in range(N_INT_VARS)},
                    "objective": val
                })
                eval_id += 1

            gen_best_ind, gen_best_val = max(pop_fit, key=lambda x: x[1])
            if gen_best_val > best_value:
                best_value = gen_best_val
                best_overall = {"cat": gen_best_ind["cat"], "ints": gen_best_ind["ints"][:]}
                gens_no_improve = 0
            else:
                gens_no_improve += 1

            if gens_no_improve >= NO_IMPROVE_STOP:
                break

            pop_fit_sorted = sorted(pop_fit, key=lambda x: x[1], reverse=True)
            elites = [ {"cat": ind["cat"], "ints": ind["ints"][:]} for ind, _ in pop_fit_sorted[:ELITISM] ]
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
    return {"best": best_overall, "best_value": best_value, "runtime_sec": runtime, "logfile": logfile}

# ==========================
# Execução automática: roda NA ORDEM
# 1) híbrido (GA + PS + PSO)
# 2) GA puro
# 3) PSO puro
# ==========================
if __name__ == "__main__":
    random.seed(123)

    print("1) Rodando modo HÍBRIDO: GA + PatternSearch + PSO")
    res_h = genetic_hybrid(seed=123, logfile="avaliacoes_hybrid.csv")
    print("Híbrido -> Melhor:", res_h["best"], "valor:", res_h["best_value"], "tempo(s):", round(res_h["runtime_sec"],3))

    print("\n2) Rodando modo GA PURO")
    res_g = genetic_pure(seed=123, logfile="avaliacoes_ga_puro.csv")
    print("GA puro -> Melhor:", res_g["best"], "valor:", res_g["best_value"], "tempo(s):", round(res_g["runtime_sec"],3))

    print("\n3) Rodando modo PSO PURO")
    res_p = pso_optimize(seed=123, n_particles=PSO_PARTICLES, iterations=PSO_ITERATIONS, logfile="avaliacoes_pso_puro.csv")
    print("PSO puro -> Melhor:", res_p["best"], "valor:", res_p["best_value"], "tempo(s):", round(res_p["runtime_sec"],3))

    print("\nExecuções completas. Logs: avaliacoes_hybrid.csv, avaliacoes_ga_puro.csv, avaliacoes_pso_puro.csv")
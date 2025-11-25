# run_verbose_mode.py
import random
import time
import csv
import math
import subprocess
from typing import List, Tuple, Dict, Any, Optional
import sys

# ==========================
# Configurações principais
# ==========================
INT_MIN, INT_MAX = 1, 100
N_INT_VARS = 5  # 5 variáveis inteiras

# Configurações de GA
POP_SIZE = 80
GENERATIONS = 30
ELITISM = 8
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE_INT = 0.5
MUTATION_CREEP_STEPS = [-5, -2, -1, 1, 2, 5]

# Configurações do Híbrido
LOCAL_REFINES_PER_GEN = 5
LOCAL_REFINE_BUDGET = 30
NO_IMPROVE_STOP = 8

# Configurações de PSO
PSO_PARTICLES = 25
PSO_ITERATIONS_GLOBAL = 30
PSO_ITERATIONS_LOCAL = 5
PSO_INERTIA = 0.7
PSO_COGNITIVE = 1.4
PSO_SOCIAL = 1.4

# Timeout (por modo)
TIMEOUT_SEGUNDOS = 600 

# Variáveis globais
START_TIME = None
TIMEOUT_GLOBAL = None
TIMEOUT_REACHED = False
EVAL_ID_COUNTER = 0 

# Caminho do executável
MODELO_EXECUTAVEL = "C:/Users/aluno/Downloads/n2-main/n2-main/simulado.exe" 
USE_SUBPROCESS = True 

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
    print(f"⏱️  Orçamento de tempo: {seconds}s")

def get_elapsed_time() -> float:
    if START_TIME is not None: return time.time() - START_TIME
    return 0.0

def log_monitor(eval_id: int, phase: str, ints: List[int], val: float):
    """Imprime cada teste no terminal em tempo real."""
    # Formatação: ID com 4 digitos | Fase com 10 caracteres | Vetor | Valor
    print(f" > [{eval_id:04d}] {phase:12s} x={ints} -> Obj: {val:.4f}")

# ==========================
# Utilidades GA
# ==========================
def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def random_individual() -> Dict[str, Any]:
    return {"ints": [random.randint(INT_MIN, INT_MAX) for _ in range(N_INT_VARS)]}

def mutate(ind: Dict[str, Any]) -> Dict[str, Any]:
    new_ind = {"ints": ind["ints"][:]}
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
        return {"ints": p1["ints"][:]}, {"ints": p2["ints"][:]}
    point = random.randint(1, N_INT_VARS - 1)
    ints1 = p1["ints"][:point] + p2["ints"][point:]
    ints2 = p2["ints"][:point] + p1["ints"][point:]
    return {"ints": ints1}, {"ints": ints2}

def tournament_select(pop_with_fit: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
    contenders = random.sample(pop_with_fit, TOURNAMENT_K)
    champion = max(contenders, key=lambda x: x[1])
    return {"ints": champion[0]["ints"][:]}

# ==========================
# Avaliação
# ==========================
def evaluate_model(ind: Dict[str, Any]) -> float:
    global EVAL_ID_COUNTER
    if check_timeout(): return -1e12 
        
    if USE_SUBPROCESS:
        args = [MODELO_EXECUTAVEL] + list(map(str, ind["ints"]))
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=5)
            if result.returncode != 0: return -1e12
            value_str = result.stdout.strip().splitlines()[0]
            val = float(value_str)
            EVAL_ID_COUNTER += 1
            return val
        except Exception: return -1e12
    else:
        # Simulação para teste sem .exe
        time.sleep(0.002) 
        val = sum(ind["ints"]) + random.uniform(-5, 5)
        EVAL_ID_COUNTER += 1
        return val

# ==========================
# Buscas Locais (PS + PSO Refine)
# ==========================
def neighborhood(ind: Dict[str, Any]) -> List[Dict[str, Any]]:
    neigh = []
    for i in range(N_INT_VARS):
        for step in MUTATION_CREEP_STEPS:
            newv = clamp(ind["ints"][i] + step, INT_MIN, INT_MAX)
            if newv != ind["ints"][i]:
                new = {"ints": ind["ints"][:]}
                new["ints"][i] = newv
                neigh.append(new)
    return neigh

def local_pattern_search(start_ind: Dict[str, Any], start_val: float, budget: int, logfile_writer: Optional[csv.DictWriter] = None) -> Tuple[Dict[str, Any], float, int]:
    current, fcur = {"ints": start_ind["ints"][:]}, start_val
    evals = 0
    improved = True
    
    while improved and evals < budget and not check_timeout():
        improved = False
        best_nb = current
        best_val = fcur
        
        for nb in neighborhood(current):
            if evals >= budget or check_timeout(): break
            fnb = evaluate_model(nb)
            evals += 1
            
            # LOG NO TERMINAL
            log_monitor(EVAL_ID_COUNTER, "PatSearch", nb["ints"], fnb)
            
            if logfile_writer:
                logfile_writer.writerow({
                    "timestamp": time.time(), "eval_id": EVAL_ID_COUNTER, "phase": "PS", "cat": "INT", 
                    **{f"x{j+1}": nb["ints"][j] for j in range(N_INT_VARS)}, "objective": fnb
                })
            
            if fnb > best_val: best_nb, best_val = nb, fnb
                
        if best_val > fcur:
            current, fcur = best_nb, best_val
            improved = True
            
    return current, fcur, evals

def swarm_refine(start_ind: Dict[str, Any], start_val: float, n_particles: int, iterations: int, evaluations_limit: int, logfile_writer: Optional[csv.DictWriter] = None) -> Tuple[Dict[str, Any], float, int]:
    global EVAL_ID_COUNTER
    swarm = []
    velocities = []
    evals = 0

    for _ in range(n_particles):
        part = {"ints": start_ind["ints"][:]}
        for d in range(N_INT_VARS):
            if random.random() < 0.6:
                part["ints"][d] = clamp(part["ints"][d] + random.choice(MUTATION_CREEP_STEPS), INT_MIN, INT_MAX)
        swarm.append(part)
        velocities.append([random.uniform(-3.0, 3.0) for _ in range(N_INT_VARS)])

    pbest = [{"ints": p["ints"][:]} for p in swarm]
    pbest_val = []
    
    # Avaliação Inicial Refinamento
    for i in range(len(pbest)):
        if evals >= evaluations_limit or check_timeout(): break
        val = evaluate_model(pbest[i])
        pbest_val.append(val)
        evals += 1
        
        # LOG NO TERMINAL
        log_monitor(EVAL_ID_COUNTER, "PSO_Ref_Ini", pbest[i]["ints"], val)

        if logfile_writer:
            logfile_writer.writerow({
                "timestamp": time.time(), "eval_id": EVAL_ID_COUNTER, "phase": "PSO_Ref_Init", "cat": "INT",
                **{f"x{j+1}": pbest[i]["ints"][j] for j in range(N_INT_VARS)}, "objective": val
            })
        
    if not pbest_val: return start_ind, start_val, evals

    best_idx = max(range(len(pbest_val)), key=lambda i: pbest_val[i])
    gbest = pbest[best_idx]
    gbest_val = pbest_val[best_idx]
    w, c1, c2 = PSO_INERTIA, PSO_COGNITIVE, PSO_SOCIAL

    for it in range(iterations):
        if evals >= evaluations_limit or check_timeout(): break
        for i in range(n_particles):
            if i >= len(pbest_val) or evals >= evaluations_limit or check_timeout(): continue 
            for d in range(N_INT_VARS):
                r1, r2 = random.random(), random.random()
                cog = c1 * r1 * (pbest[i]["ints"][d] - swarm[i]["ints"][d])
                soc = c2 * r2 * (gbest["ints"][d] - swarm[i]["ints"][d])
                velocities[i][d] = w * velocities[i][d] + cog + soc
                newv = swarm[i]["ints"][d] + int(round(velocities[i][d]))
                swarm[i]["ints"][d] = clamp(newv, INT_MIN, INT_MAX)

            val = evaluate_model(swarm[i])
            evals += 1
            
            # LOG NO TERMINAL
            log_monitor(EVAL_ID_COUNTER, "PSO_Refine", swarm[i]["ints"], val)

            if logfile_writer:
                logfile_writer.writerow({
                    "timestamp": time.time(), "eval_id": EVAL_ID_COUNTER, "phase": "PSO_Refine", "cat": "INT",
                    **{f"x{j+1}": swarm[i]["ints"][j] for j in range(N_INT_VARS)}, "objective": val
                })

            if val > pbest_val[i]:
                pbest[i] = {"ints": swarm[i]["ints"][:]}
                pbest_val[i] = val
                if val > gbest_val:
                    gbest, gbest_val = {"ints": pbest[i]["ints"][:]}, val

    return gbest, gbest_val, evals

# ==========================
# 1) HÍBRIDO (GA + PS + PSO)
# ==========================
def genetic_hybrid(seed: int = 42, logfile: str = "avaliacoes_hybrid_int.csv") -> Dict[str, Any]:
    global EVAL_ID_COUNTER
    random.seed(seed)
    t0 = time.time()
    pop = [random_individual() for _ in range(POP_SIZE)]
    best_overall, best_value = None, -1e300
    gens_no_improve = 0

    log_fields = ["timestamp","eval_id","phase","cat"] + [f"x{i+1}" for i in range(N_INT_VARS)] + ["objective"]
    with open(logfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        for gen in range(GENERATIONS):
            if check_timeout(): break
            
            print(f"\n--- GERAÇÃO {gen+1}/{GENERATIONS} (HÍBRIDO) ---")
            
            pop_fit = []
            for ind in pop:
                val = evaluate_model(ind)
                pop_fit.append((ind, val))
                
                # LOG NO TERMINAL
                log_monitor(EVAL_ID_COUNTER, "GA_Hybrid", ind["ints"], val)

                writer.writerow({
                    "timestamp": time.time(), "eval_id": EVAL_ID_COUNTER, "phase": "GA", "cat": "INT",
                    **{f"x{i+1}": ind["ints"][i] for i in range(N_INT_VARS)}, "objective": val
                })
                if check_timeout(): break
            if check_timeout() or not pop_fit: break

            gen_best_ind, gen_best_val = max(pop_fit, key=lambda x: x[1])
            if gen_best_val > best_value:
                best_value = gen_best_val
                best_overall = {"ints": gen_best_ind["ints"][:]}
                gens_no_improve = 0
            else:
                gens_no_improve += 1

            # Intensificação Local
            print(f"--- REFINAMENTO LOCAL (Top {LOCAL_REFINES_PER_GEN}) ---")
            pop_fit_sorted = sorted(pop_fit, key=lambda x: x[1], reverse=True)
            k_local = min(LOCAL_REFINES_PER_GEN, len(pop_fit_sorted))
            
            for i in range(k_local):
                if check_timeout(): break
                start_ind, start_val = pop_fit_sorted[i]
                
                loc_best, loc_val, _ = local_pattern_search(start_ind, start_val, LOCAL_REFINE_BUDGET, writer)
                sw_best, sw_val, _ = swarm_refine(loc_best, loc_val, PSO_PARTICLES, PSO_ITERATIONS_LOCAL, LOCAL_REFINE_BUDGET*2, writer)
                
                chosen_best, chosen_val = (sw_best, sw_val) if sw_val > loc_val else (loc_best, loc_val)
                label = "PSO_End" if sw_val > loc_val else "PS_End"
                
                # O log para CSV do 'melhor escolhido' não conta como avaliação nova, é apenas registro
                writer.writerow({
                    "timestamp": time.time(), "eval_id": EVAL_ID_COUNTER, "phase": label, "cat": "INT",
                    **{f"x{j+1}": chosen_best["ints"][j] for j in range(N_INT_VARS)}, "objective": chosen_val
                })
                
                if chosen_val > best_value:
                    best_value, best_overall = chosen_val, chosen_best
                    gens_no_improve = 0

            if gens_no_improve >= NO_IMPROVE_STOP: 
                print(">> Parada antecipada por falta de melhoria.")
                break
            
            # Reprodução
            pop_fit_sorted = sorted(pop_fit, key=lambda x: x[1], reverse=True)
            elites = [{"ints": ind["ints"][:]} for ind, _ in pop_fit_sorted[:ELITISM]]
            new_pop = elites[:]
            while len(new_pop) < POP_SIZE:
                c1, c2 = crossover(tournament_select(pop_fit), tournament_select(pop_fit))
                new_pop.append(mutate(c1))
                if len(new_pop) < POP_SIZE: new_pop.append(mutate(c2))
            pop = new_pop

    return {"best": best_overall, "best_value": best_value, "runtime_sec": time.time()-t0, "logfile": logfile}

# ==========================
# 2) GA PURO
# ==========================
def genetic_pure(seed: int = 42, logfile: str = "avaliacoes_ga_puro_int.csv") -> Dict[str, Any]:
    global EVAL_ID_COUNTER
    random.seed(seed)
    t0 = time.time()
    pop = [random_individual() for _ in range(POP_SIZE)]
    best_overall, best_value = None, -1e300
    gens_no_improve = 0

    log_fields = ["timestamp","eval_id","phase","cat"] + [f"x{i+1}" for i in range(N_INT_VARS)] + ["objective"]
    with open(logfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        for gen in range(GENERATIONS):
            if check_timeout(): break
            print(f"\n--- GERAÇÃO {gen+1}/{GENERATIONS} (GA PURO) ---")

            pop_fit = []
            for ind in pop:
                val = evaluate_model(ind)
                pop_fit.append((ind, val))
                
                # LOG NO TERMINAL
                log_monitor(EVAL_ID_COUNTER, "GA_Pure", ind["ints"], val)

                writer.writerow({
                    "timestamp": time.time(), "eval_id": EVAL_ID_COUNTER, "phase": "GA_Pure", "cat": "INT",
                    **{f"x{i+1}": ind["ints"][i] for i in range(N_INT_VARS)}, "objective": val
                })
                if check_timeout(): break
            if check_timeout() or not pop_fit: break

            gen_best_ind, gen_best_val = max(pop_fit, key=lambda x: x[1])
            if gen_best_val > best_value:
                best_value = gen_best_val
                best_overall = {"ints": gen_best_ind["ints"][:]}
                gens_no_improve = 0
            else:
                gens_no_improve += 1
            if gens_no_improve >= NO_IMPROVE_STOP: 
                print(">> Parada antecipada por falta de melhoria.")
                break

            # Reprodução
            pop_fit_sorted = sorted(pop_fit, key=lambda x: x[1], reverse=True)
            elites = [{"ints": ind["ints"][:]} for ind, _ in pop_fit_sorted[:ELITISM]]
            new_pop = elites[:]
            while len(new_pop) < POP_SIZE:
                c1, c2 = crossover(tournament_select(pop_fit), tournament_select(pop_fit))
                new_pop.append(mutate(c1))
                if len(new_pop) < POP_SIZE: new_pop.append(mutate(c2))
            pop = new_pop

    return {"best": best_overall, "best_value": best_value, "runtime_sec": time.time()-t0, "logfile": logfile}

# ==========================
# 3) PSO PURO
# ==========================
def pso_optimize(seed: int = 42, n_particles: int = PSO_PARTICLES, iterations: int = PSO_ITERATIONS_GLOBAL, logfile: str = "avaliacoes_pso_puro_int.csv") -> Dict[str, Any]:
    global EVAL_ID_COUNTER
    random.seed(seed)
    t0 = time.time()
    particles = []
    
    log_fields = ["timestamp","eval_id","phase","cat"] + [f"x{i+1}" for i in range(N_INT_VARS)] + ["objective"]
    with open(logfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        
        print("\n--- PSO INICIALIZAÇÃO ---")
        for _ in range(n_particles):
            if check_timeout(): break
            ind = random_individual()
            val = evaluate_model(ind)
            
            # LOG NO TERMINAL
            log_monitor(EVAL_ID_COUNTER, "PSO_Init", ind["ints"], val)
            
            particles.append({"pos": ind, "vel": {"ints": [0.0]*N_INT_VARS}, "best_pos": {"ints": ind["ints"][:]}, "best_val": val})
            writer.writerow({
                "timestamp": time.time(), "eval_id": EVAL_ID_COUNTER, "phase": "PSO_Init", "cat": "INT",
                **{f"x{i+1}": ind["ints"][i] for i in range(N_INT_VARS)}, "objective": val
            })
        
        if not particles: return {"best": None, "best_value": -1e300, "runtime_sec": get_elapsed_time(), "logfile": logfile}

        best_particle = max(particles, key=lambda p: p["best_val"])
        gbest_pos, gbest_val = best_particle["best_pos"], best_particle["best_val"]

        for it in range(iterations):
            if check_timeout(): break
            print(f"\n--- PSO ITERAÇÃO {it+1}/{iterations} ---")
            
            for i, p in enumerate(particles):
                if check_timeout(): break
                for d in range(N_INT_VARS):
                    r1, r2 = random.random(), random.random()
                    cog = PSO_COGNITIVE * r1 * (p["best_pos"]["ints"][d] - p["pos"]["ints"][d])
                    soc = PSO_SOCIAL * r2 * (gbest_pos["ints"][d] - p["pos"]["ints"][d])
                    p["vel"]["ints"][d] = PSO_INERTIA * p["vel"]["ints"][d] + cog + soc
                    newv = p["pos"]["ints"][d] + int(round(p["vel"]["ints"][d]))
                    p["pos"]["ints"][d] = clamp(newv, INT_MIN, INT_MAX)

                val = evaluate_model(p["pos"])
                
                # LOG NO TERMINAL
                log_monitor(EVAL_ID_COUNTER, "PSO_Iter", p["pos"]["ints"], val)

                writer.writerow({
                    "timestamp": time.time(), "eval_id": EVAL_ID_COUNTER, "phase": "PSO", "cat": "INT",
                    **{f"x{i+1}": p["pos"]["ints"][i] for i in range(N_INT_VARS)}, "objective": val
                })

                if val > p["best_val"]: p["best_val"], p["best_pos"] = val, {"ints": p["pos"]["ints"][:]}
                if val > gbest_val: gbest_val, gbest_pos = val, {"ints": p["pos"]["ints"][:]}

    return {"best": gbest_pos, "best_value": gbest_val, "runtime_sec": time.time()-t0, "logfile": logfile}

# ==========================
# MENU PRINCIPAL
# ==========================
if __name__ == "__main__":
    random.seed(123)
    
    print("\n" + "="*60)
    print("SELETOR DE MODO (MONITORAMENTO EM TEMPO REAL)")
    print("="*60)
    print("1 - Modo HÍBRIDO (GA + PS + PSO)")
    print("2 - Modo GA PURO")
    print("3 - Modo PSO PURO")
    
    escolha = input("\n>> Escolha: ").strip()
    
    print("\n" + "-"*60)
    
    if escolha == "1":
        set_timeout(TIMEOUT_SEGUNDOS)
        try:
            res = genetic_hybrid(seed=123, logfile="avaliacoes_hybrid_int.csv")
            print(f"\n✅ FIM. Melhor HÍBRIDO: {res['best_value']:.4f}")
        except Exception as e: print(f"❌ Erro: {e}")

    elif escolha == "2":
        set_timeout(TIMEOUT_SEGUNDOS)
        try:
            res = genetic_pure(seed=123, logfile="avaliacoes_ga_puro_int.csv")
            print(f"\n✅ FIM. Melhor GA PURO: {res['best_value']:.4f}")
        except Exception as e: print(f"❌ Erro: {e}")

    elif escolha == "3":
        set_timeout(TIMEOUT_SEGUNDOS)
        try:
            res = pso_optimize(seed=123, n_particles=PSO_PARTICLES, iterations=PSO_ITERATIONS_GLOBAL, logfile="avaliacoes_pso_puro_int.csv")
            print(f"\n✅ FIM. Melhor PSO PURO: {res['best_value']:.4f}")
        except Exception as e: print(f"❌ Erro: {e}")

    else:
        print("Opção inválida.")
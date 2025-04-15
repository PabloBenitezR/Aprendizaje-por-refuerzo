import numpy as np
import matplotlib.pyplot as plt
from pymoo.problems.many.wfg import WFG1
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from scipy.linalg import LinAlgError
from scipy.spatial.distance import cdist
from pymoo.problems.many.wfg import WFG1


# ----------------------------------------------------------
# Utilidades matemáticas
# ----------------------------------------------------------

def factorial(n):
    return 1 if n in (0, 1) else n * factorial(n - 1)

def combination(n, m):
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))

# ----------------------------------------------------------
# Generación de puntos de referencia (Das & Dennis)
# ----------------------------------------------------------

def reference_points(npop, nvar):
    h1 = 0
    while combination(h1 + nvar, nvar - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) -
              np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < nvar:
        h2 = 0
        while combination(h1 + nvar - 1, nvar - 1) + combination(h2 + nvar, nvar - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) -
                           np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * nvar)
            points = np.concatenate((points, temp_points), axis=0)
    return points

# ----------------------------------------------------------
# Normalización de objetivos
# ----------------------------------------------------------

def normalize_objectives(F, front0_indices):
    z_min = F.min(axis=0)
    z_max = F[front0_indices].max(axis=0)
    return (F - z_min) / (z_max - z_min + 1e-12)

# ----------------------------------------------------------
# Fast Non-Dominated Sorting (NSGA-II style)
# ----------------------------------------------------------

def nd_sort(objs):
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)
    s = []
    rank = np.zeros(npop, dtype=int)
    ind = 0
    pfs = {ind: []}
    
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind

    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1

    pfs.pop(ind)
    return pfs, rank



def niching_selection(F_norm, ref_dirs, front_indices, N_remaining):
    """
    Selecciona N_remaining soluciones del último frente (parcial),
    asignándolas a nichos definidos por los reference directions.

    Parámetros:
        F_norm: matriz (n_total, n_objetivos), objetivos normalizados
        ref_dirs: matriz (n_refs, n_objetivos), puntos de referencia
        front_indices: índices de soluciones que pertenecen al último frente parcial
        N_remaining: cuántas soluciones faltan por seleccionar

    Retorna:
        selected_indices: índices de las soluciones seleccionadas del último frente
    """

    # Extraer solo las soluciones del último frente (parcial)
    F_partial = F_norm[front_indices]

    # Calcular distancias angulares (perpendiculares) entre soluciones y reference directions
    cosine = 1 - cdist(F_partial, ref_dirs, metric='cosine')  # Similaridad coseno
    norm = np.linalg.norm(F_partial, axis=1).reshape(-1, 1)
    distance = norm * np.sqrt(1 - cosine ** 2)  # Distancia perpendicular

    # Asociar cada solución al punto de referencia más cercano
    assigned_refs = np.argmin(distance, axis=1)
    assigned_distances = np.min(distance, axis=1)

    # Inicializar conteo de cuántas soluciones hay por nicho
    niche_counts = np.zeros(ref_dirs.shape[0], dtype=int)
    for ref in assigned_refs:
        niche_counts[ref] += 1

    # Empieza a seleccionar soluciones
    selected_flags = np.full(len(F_partial), False)
    ref_flags = np.full(len(ref_dirs), True)  # true = el nicho aún es candidato
    selected_indices = []

    while len(selected_indices) < N_remaining:
        # Encuentra el nicho con menos soluciones asignadas (de los habilitados)
        candidate_refs = np.where(ref_flags)[0]
        min_count = np.min(niche_counts[candidate_refs])
        candidates = candidate_refs[niche_counts[candidate_refs] == min_count]
        chosen_ref = np.random.choice(candidates)

        # Encuentra soluciones no seleccionadas asociadas a ese nicho
        sol_idxs = np.where((assigned_refs == chosen_ref) & (~selected_flags))[0]
        if sol_idxs.size > 0:
            # Selecciona la más cercana al punto de referencia
            best = sol_idxs[np.argmin(assigned_distances[sol_idxs])]
            selected_flags[best] = True
            selected_indices.append(front_indices[best])
            niche_counts[chosen_ref] += 1
        else:
            # Ya no hay soluciones en este nicho → desactívalo
            ref_flags[chosen_ref] = False

    return selected_indices



def selection(pop, pc, rank, k=2):
    # binary tournament selection
    (npop, nvar) = pop.shape
    nm = int(npop * pc)
    nm = nm if nm % 2 == 0 else nm + 1
    mating_pool = np.zeros((nm, nvar))
    for i in range(nm):
        [ind1, ind2] = np.random.choice(npop, k, replace=False)
        if rank[ind1] <= rank[ind2]:
            mating_pool[i] = pop[ind1]
        else:
            mating_pool[i] = pop[ind2]
    return mating_pool


def crossover(mating_pool, lb, ub, pc, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > pc, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, pm, eta_m):
    # polynomial mutation
    (npop, nvar) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, nvar)) < pm / nvar
    mu = np.random.random((npop, nvar))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop

def environmental_selection(pop, objs, zmin, npop, V):
    # NSGA-III environmental selection
    pfs, rank = nd_sort(objs)
    nobj = objs.shape[1]
    selected = np.full(pop.shape[0], False)
    ind = 0
    while np.sum(selected) + len(pfs[ind]) <= npop:
        selected[pfs[ind]] = True
        ind += 1
    K = npop - np.sum(selected)

    # select the remaining K solutions
    objs1 = objs[selected]
    objs2 = objs[pfs[ind]]
    npop1 = objs1.shape[0]
    npop2 = objs2.shape[0]
    nv = V.shape[0]
    temp_objs = np.concatenate((objs1, objs2), axis=0)
    t_objs = temp_objs - zmin

    # extreme points
    extreme = np.zeros(nobj)
    w = 1e-6 + np.eye(nobj)
    for i in range(nobj):
        extreme[i] = np.argmin(np.max(t_objs / w[i], axis=1))

    # intercepts
    try:
        hyperplane = np.matmul(np.linalg.inv(t_objs[extreme.astype(int)]), np.ones((nobj, 1)))
        if np.any(hyperplane == 0):
            a = np.max(t_objs, axis=0)
        else:
            a = 1 / hyperplane
    except LinAlgError:
        a = np.max(t_objs, axis=0)
    t_objs /= a.reshape(1, nobj)

    # association
    cosine = 1 - cdist(t_objs, V, 'cosine')
    distance = np.sqrt(np.sum(t_objs ** 2, axis=1).reshape(npop1 + npop2, 1)) * np.sqrt(1 - cosine ** 2)
    dis = np.min(distance, axis=1)
    association = np.argmin(distance, axis=1)
    temp_rho = dict(Counter(association[: npop1]))
    rho = np.zeros(nv)
    for key in temp_rho.keys():
        rho[key] = temp_rho[key]

    # selection
    choose = np.full(npop2, False)
    v_choose = np.full(nv, True)
    while np.sum(choose) < K:
        temp = np.where(v_choose)[0]
        jmin = np.where(rho[temp] == np.min(rho[temp]))[0]
        j = temp[np.random.choice(jmin)]
        I = np.where(np.bitwise_and(~choose, association[npop1:] == j))[0]
        if I.size > 0:
            if rho[j] == 0:
                s = np.argmin(dis[npop1 + I])
            else:
                s = np.random.randint(I.size)
            choose[I[s]] = True
            rho[j] += 1
        else:
            v_choose[j] = False
    selected[np.array(pfs[ind])[choose]] = True
    return pop[selected], objs[selected], rank[selected]


def main(npop, iter, lb, ub, nobj=3, pc=1, pm=1, eta_c=30, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space
    :param pc: crossover probability (default = 1)
    :param pm: mutation probability (default = 1)
    :param eta_c: spread factor distribution index (default = 30)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # objectives
    V = reference_points(npop, nobj)  # reference vectors
    zmin = np.min(objs, axis=0)  # ideal points
    [pfs, rank] = nd_sort(objs)  # Pareto rank

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 50 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. Mating selection + crossover + mutation
        mating_pool = selection(pop, pc, rank)
        off = crossover(mating_pool, lb, ub, pc, eta_c)
        off = mutation(off, lb, ub, pm, eta_m)
        off_objs = cal_obj(off, nobj)

        # Step 2.2. Environmental selection
        zmin = np.min((zmin, np.min(off_objs, axis=0)), axis=0)
        pop, objs, rank = environmental_selection(np.concatenate((pop, off), axis=0), np.concatenate((objs, off_objs), axis=0), zmin, npop, V)

    # Step 3. Sort the results
    pf = objs[rank == 0]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of WFG1')
    plt.savefig('Pareto front')
    plt.show()




# Crear una instancia del problema WFG1
wfg = WFG1(n_var=24, n_obj=3)

# Definir la función de evaluación
def cal_obj(pop, nobj):
    return wfg.evaluate(pop)


lb = np.zeros(24)
ub = np.ones(24)


main(npop=200, iter=250, lb=lb, ub=ub, nobj=3)



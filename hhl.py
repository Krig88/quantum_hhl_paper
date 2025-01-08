import numpy as np
import scipy

a_matrix = np.array(
    [
        [0.135, -0.092, -0.011, -0.045, -0.026, -0.033, 0.03, 0.034],
        [-0.092, 0.115, 0.02, 0.017, 0.044, -0.009, -0.015, -0.072],
        [-0.011, 0.02, 0.073, -0.0, -0.068, -0.042, 0.043, -0.011],
        [-0.045, 0.017, -0.0, 0.043, 0.028, 0.027, -0.047, -0.005],
        [-0.026, 0.044, -0.068, 0.028, 0.21, 0.079, -0.177, -0.05],
        [-0.033, -0.009, -0.042, 0.027, 0.079, 0.121, -0.123, 0.021],
        [0.03, -0.015, 0.043, -0.047, -0.177, -0.123, 0.224, 0.011],
        [0.034, -0.072, -0.011, -0.005, -0.05, 0.021, 0.011, 0.076],
    ]
)

b_vector = np.array(
    [
        -0.00885448,
        -0.17725898,
        -0.15441119,
        0.17760157,
        0.41428775,
        0.44735303,
        -0.71137715,
        0.1878808,
    ]
)

# Классическое решение
sol_classical = np.linalg.solve(a_matrix, b_vector)  # classical solution
print("Классическое решение:\n", sol_classical, "\n")

# Число кубитов (для QPE и т.п.)
num_qubits = int(np.log2(len(b_vector)))

# Пример унитарного оператора (используется в HHL)
import scipy.linalg

my_unitary = scipy.linalg.expm(1j * 2 * np.pi * a_matrix)

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PhaseEstimation as PhaseEstimation_QISKIT
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.circuit.library import Initialize

# Настройки транспиляции
transpilation_options = {"classiq": "auto optimize", "qiskit": 1}


def get_qiskit_hhl_results(precision):
    """
    Создаёт схему HHL в Qiskit, запускает её симуляцию и возвращает
    ширину (число квбит), глубину, число CNOT и fidelity.

    Также выводит решение в 'классическом' векторном виде и
    сравнивает с классическим решением.
    """

    # Схема для вектора b
    vector_circuit = QuantumCircuit(num_qubits)
    initi_vec = Initialize(b_vector / np.linalg.norm(b_vector))
    vector_circuit.append(initi_vec, list(range(num_qubits)))

    # Унитарная схема из матрицы my_unitary
    q = QuantumRegister(num_qubits, "q")
    unitary_qc = QuantumCircuit(q)
    unitary_qc.unitary(my_unitary.tolist(), q)

    # QPE-схема
    qpe_qc = PhaseEstimation_QISKIT(precision, unitary_qc)

    # Схема обратного (ExactReciprocal) для 1/λ
    reciprocal_circuit = ExactReciprocal(
        num_state_qubits=precision, scaling=1 / 2 ** precision
    )

    # Регистры:
    qb = QuantumRegister(num_qubits, "qb")  # хранит вектор b и решение
    ql = QuantumRegister(precision, "ql")  # кубиты для QPE
    qf = QuantumRegister(1, "qf")  # флаг-кубит

    hhl_qc = QuantumCircuit(qb, ql, qf)

    # 1) Инициализация вектора b
    hhl_qc.append(vector_circuit, qb[:])
    # 2) QPE
    hhl_qc.append(qpe_qc, ql[:] + qb[:])
    # 3) Обратное преобразование (ExactReciprocal)
    hhl_qc.append(reciprocal_circuit, ql[::-1] + [qf[0]])
    # 4) Обратная QPE
    hhl_qc.append(qpe_qc.inverse(), ql[:] + qb[:])

    # Транспиляция
    tqc = transpile(
        hhl_qc,
        basis_gates=["u3", "cx"],
        optimization_level=transpilation_options["qiskit"],
    )
    depth = tqc.depth()
    cx_counts = tqc.count_ops().get("cx", 0)
    total_q = tqc.width()

    # Вычисляем statevector
    statevector = np.array(Statevector(tqc))

    # Индексы для извлечения |1>_flag |0..0>_qpe ...  (считаем, что там решение)
    all_entries = [np.binary_repr(k, total_q) for k in range(2 ** total_q)]
    sol_indices = [
        int(entry, 2)
        for entry in all_entries
        if entry[0] == "1" and entry[1: precision + 1] == "0" * precision
    ]

    # Собираем квантовое решение, учтя масштаб 1/(1/2^precision)
    qsol = statevector[sol_indices] / (1 / 2 ** precision)

    # --- NEW: пересчёт квантового решения в «классический» вид и сравнение ---
    # По умолчанию qsol имеет свою собственную норму. Чтобы сравнить
    # "как вектор" с классическим решением, масштабируем qsol
    # под норму классического решения (или можно было бы оставить единичную).
    scaled_qsol = qsol * (np.linalg.norm(sol_classical) / np.linalg.norm(qsol))

    print(f"== precision={precision} ==")
    print(f"Cubits (width): {total_q}")
    print(f"Circuit depth(depth): {depth}")
    print("Quantum solution:\n", scaled_qsol.real)
    print("Classical solution:\n", sol_classical)
    diff = np.linalg.norm(scaled_qsol.real - sol_classical)
    print("Difference (L2-норма) between solutions:", diff, "\n")
    # --- end NEW ---

    # Fidelity (как и было в исходном коде)
    # Для fidelity мы сравниваем направления (нормированные вектора)
    fidelity = (
            np.abs(
                np.dot(
                    sol_classical / np.linalg.norm(sol_classical),
                    qsol / np.linalg.norm(qsol),
                )
            )
            ** 2
    )

    return total_q, depth, cx_counts, fidelity


qiskit_widths = []
qiskit_depths = []
qiskit_cx_counts = []
qiskit_fidelities = []

for precision in range(2, 9):
    total_q, depth, cx_counts, fidelity = get_qiskit_hhl_results(precision)
    qiskit_widths.append(total_q)
    qiskit_depths.append(depth)
    qiskit_cx_counts.append(cx_counts)
    qiskit_fidelities.append(fidelity)

print("Собранные результаты:")
print("qiskit_widths:    ", qiskit_widths)
print("qiskit_depths:    ", qiskit_depths)
print("qiskit_cx_counts: ", qiskit_cx_counts)
print("qiskit_fidelities:", qiskit_fidelities)

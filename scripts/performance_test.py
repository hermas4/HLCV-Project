# stress_cpu.py
from multiprocessing import Process, cpu_count
import numpy as np

def cpu_stress_worker(matrix_size=2000):
    """
    Endlosschleife: erzeugt zufällige Matrizen und
    multipliziert sie, um die CPU maximal auszulasten.
    """
    # einmal initial Matrixen allokieren
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)
    while True:
        # heavy operation: Matrixmultiplikation
        C = A @ B
        # um Schreibzugriffe zu vermeiden, nur kurz das Ergebnis abfragen
        _ = C[0, 0]

def start_cpu_stress(matrix_size=2000):
    """
    Startet pro CPU-Kern einen Prozess, der jeweils
    Matrizen der Größe matrix_size x matrix_size multipliziert.
    """
    procs = []
    n_cpus = cpu_count()
    print(f"Starte {n_cpus} Stress-Prozesse mit Matrixgröße {matrix_size}×{matrix_size} …")
    for _ in range(n_cpus):
        p = Process(target=cpu_stress_worker, args=(matrix_size,), daemon=True)
        p.start()
        procs.append(p)
    print("Alle Stress-Prozesse laufen im Hintergrund.")
    return procs

if __name__ == "__main__":
    # Passe matrix_size nach Wunsch an (je größer, desto mehr RAM/CPU)
    procs = start_cpu_stress(matrix_size=3000)
    # Damit das Script im Vordergrund bleibt und du es abbrechen kannst:
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("Abbruch durch Benutzer, Prozesse werden terminiert.")

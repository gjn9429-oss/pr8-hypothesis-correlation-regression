import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from block2 import compare_means

def run():
    result = compare_means(100, 24, 4, 100, 42, 6)
    reject = result["pvalue"] < 0.01
    print("БЛОК 4. Київ мінус Франкфурт:", result)
    print("H0 відхиляємо при alpha=0.01" if reject else "H0 не відхиляємо")
    return result

if __name__ == "__main__":
    run()

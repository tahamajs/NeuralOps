import sys
import time

sys.path.insert(0, 'Codes')
sys.path.insert(0, '.')

from scratch.test_all_22_hardened import test_all as test_all_hardened
from scratch.test_group1_hardened import test_group1
from scratch.test_group2_hardened import test_group2
from scratch.test_group3_hardened import test_group3
from scratch.test_group4_hardened import test_group4
from scratch.test_group5_hardened import test_group5
from scratch.test_group6_hardened import test_group6
from scratch.test_group7_hardened import test_group7
from scratch.test_group8_hardened import test_group8
from scratch.test_group9_hardened import test_group9
from scratch.test_group10_hardened import test_group10
from scratch.test_group11_hardened import test_group11
from scratch.test_group12_hardened import test_group12

def master_test():
    start_time = time.time()
    print("=" * 90)
    print("     MASTER VERIFICATION RUNNER: ALL ATTENTION & HYBRID MODULES (109 MODULES)")
    print("=" * 90)

    suites = [
        ("Batch 0 (Core 23 modules)", test_all_hardened),
        ("Group 1 (9 modules)", test_group1),
        ("Group 2 (9 modules)", test_group2),
        ("Group 3 (6 modules)", test_group3),
        ("Group 4 (5 modules)", test_group4),
        ("Group 5 (4 modules)", test_group5),
        ("Group 6 (7 modules)", test_group6),
        ("Group 7 (7 modules)", test_group7),
        ("Group 8 (8 modules)", test_group8),
        ("Group 9 (8 modules)", test_group9),
        ("Group 10 (Minimal Family - 5 modules)", test_group10),
        ("Group 11 (AEGIS Family - 8 modules)", test_group11),
        ("Group 12 (Hybrid SSM/Mamba - 10 modules)", test_group12),
    ]

    all_passed = True
    suite_results = []
    for name, fn in suites:
        print(f"\n>>>> RUNNING SUITE: {name}")
        try:
            passed = fn()
            suite_results.append((name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"SUITE CRASHED: {name} with error {e}")
            suite_results.append((name, False))
            all_passed = False

    elapsed = time.time() - start_time
    print("\n" + "=" * 90)
    print(f"     MASTER VERIFICATION REPORT (Elapsed: {elapsed:.2f}s)")
    print("=" * 90)
    for name, passed in suite_results:
        status = "[PASS 100%]" if passed else "[FAIL]"
        print(f"  {status:<12} {name}")
    print("=" * 90)
    if all_passed:
        print("🎉 ALL 109 MODULES (433+ TESTS) PASSED 100% ACROSS ALL 4 ARCHITECTURES!")
    else:
        print("❌ SOME TEST SUITES FAILED.")
    print("=" * 90)
    return all_passed

if __name__ == '__main__':
    success = master_test()
    sys.exit(0 if success else 1)

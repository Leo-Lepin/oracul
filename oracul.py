import math
import random
from dataclasses import dataclass
from itertools import islice
from typing import Callable, Iterator

from twins import *

Case = Callable[[Twin], list[dict]]

CaseGen = Callable[[], Iterator[Case]]


@dataclass
class TestGroup:
    generator: CaseGen
    count: int | None = None


TESTS: dict[str, TestGroup] = {}


def bench_group(name: str, count: int | None = None):
    """Декоратор: регистрирует генератор кейсов в TESTS"""

    def deco(fn: CaseGen) -> CaseGen:
        TESTS[name] = TestGroup(generator=fn, count=count)
        return fn

    return deco


def command_case(mode: str, value) -> Case:
    """Одна команда -> один снимок"""

    def case(twin: Twin) -> list[dict]:
        d = Data()
        twin.command(d, mode, value)
        return [twin.state(d)]

    return case


def scenario_case(fn: Callable[[Twin, Data], Iterator[dict]]) -> Case:
    """Сценарий (генератор) -> полная траектория"""

    def case(twin: Twin) -> list[dict]:
        return list(fn(twin, Data()))

    return case


@bench_group("commands", count=20)
def gen_commands() -> Iterator[Case]:
    """Случайные команды: setpoint / heater"""
    rng = random.Random(0)
    while True:
        if rng.random() < 0.5:
            yield command_case("setpoint", rng.uniform(-20, 100))
        else:
            yield command_case("heater", rng.choice([True, False, 1, 0]))


@bench_group("edge_setpoint")
def gen_edge_setpoints() -> Iterator[Case]:
    for v in [-100.0, 0.0, 1e9, "50"]:
        yield command_case("setpoint", v)


@bench_group("edge_heater")
def gen_edge_heaters() -> Iterator[Case]:
    for v in [True, False, 1, 0, "", "on", "off"]:
        yield command_case("heater", v)


def scn_cold_start(twin: Twin, d: Data, steps: int = 300):
    twin.command(d, "heater", True)
    for _ in range(steps):
        yield twin.step(d)


def scn_cool_down(twin: Twin, d: Data, steps: int = 300):
    d.T = 80.0
    twin.command(d, "heater", False)
    for _ in range(steps):
        yield twin.step(d)


def scn_thermostat(twin: Twin, d: Data, steps: int = 400, switch: int = 200):
    d.setpoint = 40.0
    for i in range(steps):
        if i == switch:
            twin.command(d, "setpoint", 60.0)
        twin.command(d, "heater", d.T < d.setpoint)
        yield twin.step(d)


def scn_no_heat(twin: Twin, d: Data, steps: int = 100):
    for _ in range(steps):
        twin.command(d, "setpoint", 60.0)
        yield twin.step(d)


def scn_fahrenheit(twin: Twin, d: Data, steps: int = 200):
    twin.command(d, "setpoint", 70.0)
    for _ in range(steps):
        twin.command(d, "heater", d.T < d.setpoint)
        yield twin.step(d)


@bench_group("scenarios")
def gen_scenarios() -> Iterator[Case]:
    scenarios = [
        scn_cold_start,
        scn_cool_down,
        scn_thermostat,
        scn_no_heat,
        scn_fahrenheit,
    ]
    for fn in scenarios:
        yield scenario_case(fn)


class PhysicsViolation(Exception):
    pass


class PolyTwinBench:

    def __init__(
        self,
        twin: Twin,
        ref: Twin | None = None,
        tests: dict[str, TestGroup] | None = None,
        tol_T: float = 0.5,
        tol_sp: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        self.twin = twin
        self.ref = ref if ref is not None else ReferenceTwin()
        self.tests = tests if tests is not None else TESTS
        self.tol_T = tol_T
        self.tol_sp = tol_sp
        self.eps = eps

    def run(self) -> dict[str, int]:
        name = type(self.twin).__name__
        print(f"\n=== {name} vs {type(self.ref).__name__} ===")
        summary: dict[str, int] = {}

        for group_name, group in self.tests.items():
            cases = group.generator()
            if group.count is not None:
                cases = islice(cases, group.count)

            diverged = 0
            total = 0
            for i, case in enumerate(cases):
                total += 1
                if self._apply(case, group_name, i):
                    diverged += 1

            summary[group_name] = diverged
            print(f"  [{group_name}] {total - diverged}/{total} pass")

        return summary

    def _apply(self, case: Case, group: str, idx: int) -> bool:
        ref_status, ref_val = self._call(case, self.ref)
        twin_status, twin_val = self._call(case, self.twin)

        if ref_status != twin_status:
            print(
                f"  {group}[{idx}] status mismatch: "
                f"ref={ref_status} twin={twin_status}"
            )
            return True

        if ref_status != "ok":
            return False

        m = self._compare(twin_val, ref_val)  # type: ignore
        is_div = (not math.isclose(m["T_max"], 0.0, abs_tol=self.tol_T)) or (
            not math.isclose(m["setpoint_max"], 0.0, abs_tol=self.tol_sp)
        )
        if is_div:
            self._log(group, idx, m)
        return is_div

    @staticmethod
    def _call(case: Case, twin: Twin):
        try:
            traj = case(twin)

            for step in traj:
                T = step["T"]
                if not math.isfinite(T) or T > 1000.0 or T < -273.15:
                    raise PhysicsViolation(f"T={T}")

            return ("ok", traj)
        except Exception as e:
            return (f"err:{type(e).__name__}", None)

    def _compare(self, traj: list[dict], ref: list[dict]) -> dict:
        n = min(len(traj), len(ref))
        out = {"n": n}
        for f in ("T", "setpoint", "heater"):
            diffs, first = [], None
            for i in range(n):
                a, b = traj[i][f], ref[i][f]
                diff = (0.0 if a == b else 1.0) if f == "heater" else abs(a - b)
                diffs.append(diff)
                if first is None and diff > self.eps:
                    first = i
            out[f"{f}_max"] = (  # pyright: ignore[reportArgumentType]
                max(diffs) if diffs else 0.0
            ) 
            out[f"{f}_rmse"] = ( # pyright: ignore[reportArgumentType]
                math.sqrt(sum(x * x for x in diffs) / n) if n else 0.0
            )  
            out[f"{f}_first"] = first  # pyright: ignore[reportArgumentType]
        return out

    def _log(self, group: str, idx: int, m: dict) -> None:
        first = "-" if m["T_first"] is None else str(m["T_first"])
        print(
            f"  {group}[{idx}] max|ΔT|={m['T_max']:>8.3e} "
            f"rmseΔT={m['T_rmse']:>8.3e} 1stΔT={first:>4} "
            f"max|Δsp|={m['setpoint_max']:>8.3e}  DIVERGE"
        )


def main():
    print("=" * 60)
    print("Старт")
    print("=" * 60)

    models = {
        "Reference (эталон)": ReferenceTwin(),
        "BadSign (неверный знак)": BadSign(),
        "BadThermalMass (забыта теплоемкость)": BadThermalMass(),
        "BadInvertedActuator (инверсия)": BadInvertedActuator(),
        "BadStuckActuator (без heater)": BadStuckActuator(),
        "BadUnitsCommand (Фаренгейты)": BadUnitsCommand(),
    }

    all_passed = True

    for name, twin in models.items():
        print(f"\nПроверка: {name}...")
        bench = PolyTwinBench(twin)

        failures_dict = bench.run()

        total_failures = sum(failures_dict.values())

        is_good_model = name == "Reference (эталон)"
        oracle_says_good = total_failures == 0

        if oracle_says_good == is_good_model:
            if is_good_model:
                print(f"  -> [OK] Оракул пропустил эталон (0 ошибок).")
            else:
                print(
                    f"  -> [OK] Оракул поймал испорченную модель ({total_failures} провалов)."
                )
        else:
            print(
                f"  -> [FAIL] Оракул ОШИБСЯ! (Ошибок: {total_failures}, Ожидали хорошую: {is_good_model})"
            )
            all_passed = False

    print("\n" + "=" * 60)
    print("ИТОГ ТЕСТИРОВАНИЯ ОРАКУЛА:")
    if all_passed:
        print("УСПЕХ! Оракул корректно отличает правильную модель от испорченных.")
    else:
        print("ПРОВАЛ! Оракул не смог распознать некоторые ошибки.")
    print("=" * 60)


if __name__ == "__main__":
    main()

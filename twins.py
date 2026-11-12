from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any
import math

import pytest


# ==========================================================================
# Состояние
# ==========================================================================

@dataclass
class Data:
    T: float = 20.0
    t: float = 0.0
    heater: bool = False
    setpoint: float = 50.0
    dt: float = 1.0
    t_env: float = 20.0
    C: float = 4200.0
    k: float = 5.0
    P: float = 2000.0


# ==========================================================================
# Базовый твин: шаблонный метод + протокол команд
# ==========================================================================

class Twin(ABC):
    def step(self, d: Data) -> dict:
        self._physics(d)
        d.t += d.dt
        return self.state(d)

    def state(self, d: Data) -> dict:
        return self._snapshot(d)

    def command(self, d: Data, mode: str, value: Any) -> None:
        self._apply(d, mode, value)

    @abstractmethod
    def _physics(self, d: Data) -> None:
        """Шаг физики. Мутирует d in-place. Время не двигает."""

    def _apply(self, d: Data, mode: str, value: Any) -> None:
        if mode == "setpoint":
            d.setpoint = float(value)
        elif mode == "heater":
            d.heater = bool(value)
        else:
            raise KeyError(f"unknown command: {mode!r}")

    @staticmethod
    def _snapshot(d: Data) -> dict:
        return {"t": d.t, "T": d.T, "setpoint": d.setpoint, "heater": d.heater}



class ReferenceTwin(Twin):
    def _physics(self, d: Data) -> None:
        q = d.P if d.heater else 0.0
        d.T += d.dt * (q - d.k * (d.T - d.t_env)) / d.C


class BadStuckActuator(ReferenceTwin):
    """Команда 'heater' игнорируется."""

    def _apply(self, d: Data, mode: str, value: Any) -> None:
        if mode == "heater":
            return
        super()._apply(d, mode, value)


class BadUnitsCommand(ReferenceTwin):
    """Уставку принимаем в °F, храним в °C."""

    def _apply(self, d: Data, mode: str, value: Any) -> None:
        if mode == "setpoint":
            d.setpoint = (float(value) - 32) * 5 / 9
        else:
            super()._apply(d, mode, value)

class BadSign(ReferenceTwin):
    """Неверный знак в уравнении теплопотерь. 
    Бак не остывает, а саморазогревается от окружающей среды."""
    def _physics(self, d: Data) -> None:
        q = d.P if d.heater else 0.0
        d.T += d.dt * (q + d.k * (d.T - d.t_env)) / d.C


class BadThermalMass(ReferenceTwin):
    """Забыли поделить на теплоемкость (C).
    Из-за этого бак нагревается и остывает в тысячи раз быстрее (мгновенно)."""
    def _physics(self, d: Data) -> None:
        q = d.P if d.heater else 0.0
        d.T += d.dt * (q - d.k * (d.T - d.t_env))


class BadInvertedActuator(ReferenceTwin):
    """Инвертирована логика актуатора.
    При команде 'включить' (True) нагреватель выключается, и наоборот."""
    def _apply(self, d: Data, mode: str, value: Any) -> None:
        if mode == "heater":
            d.heater = not bool(value)
        else:
            super()._apply(d, mode, value)

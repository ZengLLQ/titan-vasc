"""Unit tests for command-line interface execution and subcommands."""

import sys
import pytest

from titan_vasc.cli import build_parser, main


def test_cli_parser_subcommands() -> None:
    parser = build_parser()
    subparsers = parser._subparsers._actions[1].choices

    assert "simulate" in subparsers
    assert "calibrate" in subparsers
    assert "benchmark" in subparsers
    assert "inspect" in subparsers
    assert "demo" in subparsers


def test_cli_subcommand_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(["benchmark"])
    ret = args.func(args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "NUMERICAL SCHEME & ACCURACY BENCHMARK" in captured.out


def test_cli_subcommand_inspect(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(["inspect", "--type", "fusiform", "--diameter", "5.2"])
    ret = args.func(args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "INFRARENAL ABDOMINAL AORTA" in captured.out


def test_cli_subcommand_calibrate(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(["calibrate", "--patient-id", "P-TEST", "--diameter", "5.4", "--samples", "20"])
    ret = args.func(args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "TITANVASC: CARDIOVASCULAR HEMODYNAMICS" in captured.out


def test_cli_subcommand_demo(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(["demo"])
    ret = args.func(args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "END-TO-END DEMO" in captured.out

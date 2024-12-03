#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: test_i.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-12-03 17:41:40

import subprocess
import os
import pytest

@pytest.fixture(scope="module")
def project_root():
    """
    Fixture to determine the project's root directory and change to it for the test.
    """
    script_dir = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(script_dir, "../"))
    os.chdir(root)
    return root

@pytest.fixture
def setup_files():
    """
    Fixture to create necessary files for the tests and clean up afterward.
    """
    files = [
        ("test.txt", "This is a test file."),
        ("config/config.yaml", """config_version: "1.0"\nlogging:\n  enabled: true"""),
    ]
    for file_path, content in files:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
    yield
    for file_path, _ in files:
        if os.path.exists(file_path):
            os.remove(file_path)
    if os.path.exists("config"):
        os.rmdir("config")

def run_command(command):
    """
    Executes a shell command and captures the output.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def test_help_message(project_root, setup_files):
    command = f"python {os.path.join(project_root, 'i/i.py')} -h"
    returncode, stdout, stderr = run_command(command)
    assert returncode == 0
    assert "usage" in stdout

def test_identify_object(project_root, setup_files):
    command = f"python {os.path.join(project_root, 'i/i.py')} test.txt identify"
    returncode, stdout, stderr = run_command(command)
    assert returncode == 0

def test_extract_email_piped_input(project_root, setup_files):
    command = f"echo 'test@example.com' | python {os.path.join(project_root, 'i/i.py')} extract email"
    returncode, stdout, stderr = run_command(command)
    assert returncode == 0
    assert "test@example.com" in stdout

def test_invalid_action(project_root, setup_files):
    command = f"python {os.path.join(project_root, 'i/i.py')} test.txt invalid_action"
    returncode, stdout, stderr = run_command(command)
    assert returncode != 0
    assert "Invalid action" in stderr or stdout

def test_diagnose_interaction(project_root, setup_files):
    command = f"echo 'Disk is almost full.' | python {os.path.join(project_root, 'i/i.py')} diagnose disk : Focus on optimization"
    returncode, stdout, stderr = run_command(command)
    assert returncode == 0
    assert "optimization" in stdout

def test_dynamic_argument_loading(project_root, setup_files):
    command = f"python {os.path.join(project_root, 'i/i.py')} test.txt extract email"
    returncode, stdout, stderr = run_command(command)
    assert returncode == 0

def test_missing_config_file(project_root, setup_files):
    config_path = "config/config.yaml"
    backup_path = "config/config_backup.yaml"
    if os.path.exists(config_path):
        os.rename(config_path, backup_path)
    try:
        command = f"python {os.path.join(project_root, 'i/i.py')} test.txt extract email"
        returncode, stdout, stderr = run_command(command)
        assert returncode != 0
        assert "Config file not found" in stderr or stdout
    finally:
        if os.path.exists(backup_path):
            os.rename(backup_path, config_path)


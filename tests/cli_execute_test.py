#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: test_cli.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-12-01 20:58:48
# Modified: 2024-12-01 23:47:01

import subprocess
import os

def run_command(command):
    """
    Executes a shell command and captures the output.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(f"Command: {command}")
        print(f"Output:\n{result.stdout}")
        if result.stderr:
            print(f"Error:\n{result.stderr}")
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        print(f"Error running command: {e}")
        return -1, "", str(e)

def ensure_file(file_path, content=""):
    """
    Ensures a file exists with the specified content.
    """
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(content)

if __name__ == "__main__":
    # Determine the script's directory as the base path
    script_dir = os.path.abspath(os.path.dirname(__file__))
    print(script_dir)
    project_root = os.path.abspath(os.path.join(script_dir, "../"))
    print(project_root)
    os.chdir(project_root)

    # Ensure necessary directories and files exist
    ensure_file("test.txt", "This is a test file.")
    ensure_file("config/config.yaml", """config_version: "1.0"\nlogging:\n  enabled: true""")

    # Test 1: Verify help message
    print("=== Test 1: Help Message ===")
    run_command(f"python {os.path.join(project_root, 'i/i.py')} -h")

    # Test 2: Identify object from file
    print("=== Test 2: Identify Object ===")
    run_command(f"python {os.path.join(project_root, 'i/i.py')} test.txt identify")

    # Test 3: Extract email from piped input
    print("=== Test 3: Extract Email from Piped Input ===")
    run_command(f"echo 'test@example.com' | python {os.path.join(project_root, 'i/i.py')} extract email")

    # Test 4: Invalid action
    print("=== Test 4: Invalid Action ===")
    run_command(f"python {os.path.join(project_root, 'i/i.py')} test.txt invalid_action")

    # Test 5: Diagnose interaction
    print("=== Test 5: Diagnose Interaction ===")
    run_command(f"echo 'Disk is almost full.' | python {os.path.join(project_root, 'i/i.py')} diagnose disk : Focus on optimization")

    # Test 6: Dynamic argument loading from actions.yaml
    print("=== Test 6: Dynamic Argument Loading ===")
    run_command(f"python {os.path.join(project_root, 'i/i.py')} test.txt extract email")

    # Test 7: Missing Config File
    print("=== Test 7: Missing Config File ===")
    config_backup = os.path.join("config", "config_backup.yaml")
    try:
        os.rename("config/config.yaml", config_backup)
        run_command(f"python {os.path.join(project_root, 'i/i.py')} test.txt extract email")
    except FileNotFoundError:
        print("Error: Config file not found.")
    finally:
        if os.path.exists(config_backup):
            os.rename(config_backup, "config/config.yaml")

    # Cleanup
    if os.path.exists("test.txt"):
        os.remove("test.txt")


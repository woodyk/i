#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: i.py
# Author: Wadih Khairallah
# Description: Dynamic CLI tool powered by configuration
# Created: 2024-12-01 17:19:19
# Modified: 2024-12-01 23:58:23

import os
import sys
import argparse
import yaml
from handler import Handler


def load_config():
    """
    Load configuration from the specified directories in order of priority:
    1. config/config.yaml
    2. config/default/*
    3. config/user/*
    """
    config_paths = [
        "config/config.yaml",
        "config/default/",
        "config/user/"
    ]

    merged_config = {}

    for path in config_paths:
        if os.path.isfile(path):
            # Load a single YAML file
            with open(path, "r") as file:
                config = yaml.safe_load(file)
                merged_config.update(config)
        elif os.path.isdir(path):
            # Load all YAML files in the directory
            for root, _, files in os.walk(path):
                for file_name in sorted(files):
                    if file_name.endswith(".yaml"):
                        full_path = os.path.join(root, file_name)
                        with open(full_path, "r") as file:
                            config = yaml.safe_load(file)
                            merged_config.update(config)

    if not merged_config:
        raise RuntimeError("No valid configuration files found.")

    return merged_config


def main():
    # Load configuration
    try:
        config = load_config()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Determine if input is piped
    piped_input = None
    if not sys.stdin.isatty():
        piped_input = sys.stdin.read().strip()

    # Extract available actions from the configuration
    available_actions = [action["name"] for action in config["actions"]]

    # Create the argument parser
    parser = argparse.ArgumentParser(prog="i", description="Configurable CLI tool.")
    parser.add_argument("object", nargs="?", help="Object to act on (file, URL, etc.).")
    parser.add_argument("action", choices=available_actions, help="Action to perform.")
    parser.add_argument("args", nargs="*", help="Additional arguments for the action.")
    parser.add_argument("interaction", nargs="?", help="LLM interaction prompt (use `:` to signal).")

    # Parse arguments
    args = parser.parse_args()

    # Initialize the handler with the loaded configuration
    handler = Handler(config)

    # Route command to the handler
    try:
        result = handler.process(args.action, args.args, piped_input or args.object)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: handler.py
# Author: Wadih Khairallah
# Description: Core routing and LLM integration for the 'i' CLI tool.
# Created: 2024-12-01 17:20:05
# Modified: 2024-12-02 00:01:41

import importlib

class Handler:
    """
    Core handler for routing objects and actions based on configuration.
    """

    def __init__(self, config):
        self.config = config

    def process(self, action_name, args=None, piped_input=None):
        """
        Process the input and route the action to the appropriate module.

        Args:
            action_name (str): The name of the action to perform.
            args (list): Additional arguments for the action.
            piped_input (str): Input provided via a pipe.

        Returns:
            str: The result of the action or interaction.
        """
        # Default args to an empty list
        args = args or []

        # Determine the primary object (from piped input or args)
        object_input = piped_input or (args[0] if args else None)
        if not object_input:
            raise ValueError("Error: Missing required argument: object")

        # Locate the action configuration
        actions = self.config.get("actions", [])
        action_config = next((action for action in actions if action["name"] == action_name), None)

        if not action_config:
            raise ValueError(f"Error: Action '{action_name}' is not defined in the configuration.")

        # Route based on action type
        action_type = action_config.get("type", "function")
        if action_type == "function":
            return self._execute_function(action_config, object_input, args)
        elif action_type == "interaction":
            return self._execute_interaction(action_config, object_input, args)
        else:
            raise ValueError(f"Unsupported action type: {action_type}")

    def _execute_function(self, action_config, object_input, args):
        """
        Execute a function-based action.

        Args:
            action_config (dict): Configuration for the action.
            object_input (str): The main object to process.
            args (list): Additional arguments for the action.

        Returns:
            str: Result of the action.
        """
        try:
            module_name = f"{action_config['module']}"
            function_name = action_config["function"]
            module = importlib.import_module(module_name)
            action_function = getattr(module, function_name)
            return action_function(object_input, *args)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Module '{module_name}' not found.")
        except AttributeError:
            raise AttributeError(f"Function '{function_name}' not found in module '{module_name}'.")

    def _execute_interaction(self, action_config, object_input, args):
        """
        Handle an interaction-based action.

        Args:
            action_config (dict): Configuration for the action.
            object_input (str): Input data for the interaction.
            args (list): Additional arguments for the action.

        Returns:
            str: Simulated or real interaction response.
        """
        prompt = action_config.get("prompt", "Provide a response based on the input.")
        arg_prompts = [
            action.get("prompt", "")
            for action in action_config.get("args", [])
        ]
        full_prompt = f"{prompt}\n\nObject: {object_input}\nArgs: {args}\n{' '.join(arg_prompts)}"
        print(f"Executing interaction with prompt:\n{full_prompt}")
        return "Simulated LLM Response"


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: action_identify.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-12-01 21:42:59
# Modified: 2024-12-02 00:04:56

def identify_object(input_data, *args, **kwargs):
    """
    Idntifys the type of object given to it.
    """
    return { "obj": input_data }

def validate_object(input_data, obj_type, **kwargs):
    """
    Validates an object against a particular type.
    """
    return { "obj": obj, "obj_type": obj_type }

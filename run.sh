#!/usr/bin/env bash

## Learn representations on dataset umls.
python learn_representation.py --dataset umls

## Instance type prediction on dataset umls.
python evaluate_instance_type.py --dataset umls
# Genetic Algorithm for AWS EBS Storage Optimization

This project implements a genetic algorithm to optimize the cost of Amazon Web Services (AWS) Elastic Block Store (EBS) storage. The genetic algorithm explores different volume sizes and AWS EBS pricing options to find the most cost-effective storage solution.

## Overview

The genetic algorithm is implemented using the DEAP (Distributed Evolutionary Algorithms in Python) library. It considers various EBS storage types, such as General Purpose SSD (gp3), Provisioned IOPS SSD (io2/io1), Throughput Optimized HDD (st1), and Cold HDD (sc1).

## Prerequisites

- Python 3.x
- DEAP library

Install the required library using the following command:

```bash
pip install deap

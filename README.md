# Energy Community Simulation & Recommendation System

A multi-agent energy management simulation built in Python that models energy sharing, battery storage optimization, and demand-response recommendation logic across a residential community.

## Overview
The system models a local energy community consisting of **9 valid households** and a shared **50 kWh Battery Energy Storage System (ESS)** coupled with a community photovoltaic (PV) array. It evaluates real-world consumption patterns, solar irradiance, and dynamic energy allocation under stochastic user behaviors.

## Key Features
- **Multi-Agent Simulation (Mesa Framework):** Simulates individual household agents reacting probabilistically to community-level signals.
- **Rule-Based Recommendation Engine:** Employs a traffic-light logic (surplus vs. deficit) to encourage self-consumption and peak shaving.
- **Dynamic Scoring Mechanism:** Evaluates agent compliance using a 60-day moving window deque implementation) over an 8,760-hour simulated year.
- **Data Preprocessing & Validation:** Cleans and processes raw electrical consumption data (IRISE dataset), handling sensor faults, missing timestamps, and outlier filtering.
- **Interactive Visualizations:** Generates comprehensive energy balance and State of Charge (SOC) tracking reports using Plotly.

## Tech Stack
- **Language:** Python
- **Libraries:** Mesa, Pandas, NumPy, Plotly

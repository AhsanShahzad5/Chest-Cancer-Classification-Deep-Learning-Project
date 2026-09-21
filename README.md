# Workflows

(VIMP)

we follow these steps in our Architecture , starting with data injestion we follow all of these steps

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml


## Easier more clear explanation

### what this project arcitecture is?

This project has a **Configuration-Driven Pipeline Architecture** (often referred to as Config-Driven Modular Design in modern MLOps).

Under the hood, this architecture combines several classic software engineering design patterns to build scalable, reproducible Machine Learning pipelines:

### Why This Is the Gold Standard for MLOps

When you combine this architecture with orchestration tools like **DVC** (Data Version Control) or **MLflow**, the pipeline becomes fully deterministic:

**DVC** tracks your config.yaml and params.yaml files.

If parameters in params.yaml change, **DVC** automatically detects which pipeline stage needs to re-run.

**MLflow** logs the parameters parsed by your *ConfigurationManager* directly to an experiment tracking dashboard.


### HOW It Works (Step-by-Step Workflow)

**Step 1: Store Plain Text Settings (config/config.yaml)**

You write human-readable paths and URLs in YAML:


**Step 2: Define the Data Schema (entity/config_entity.py)**

You define an immutable, strongly-typed Dataclass to enforce structure


**Step 3: Build the Factory (config/configuration.py)**

The ConfigurationManager reads the raw YAML, handles folder creation side-effects, and returns the populated Dataclass

The Configuration Manager pattern acts as the bridge between your raw, unstructured configuration files (config.yaml, params.yaml) and your actual Python pipeline components (DataIngestion, ModelTraining, etc.).

Instead of having every script open YAML files and parse dictionaries manually, the ConfigurationManager centralizes config parsing, validates data types, handles side effects (like creating artifact directories), and provides strongly typed objects to your pipeline components.

**Step 5: Orchestrate in the Pipeline** 

The pipeline runner ties all pieces together cleanly

**Step 6: Run the Pipeline**

Execute the entry point (main.py) to run the entire workflow end-to-end